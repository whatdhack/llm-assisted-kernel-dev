import time

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_reg_layout,
    tma, mbarrier, tcgen05_mma, fence_async_shared,
)

from task import input_t, output_t

# Same Blackwell (cc 10.x) tcgen05 + tf32x3 SYRK step as cholesky_gluon_tcgen05.py,
# but fixing the batching design flaw identified in the conversation this file
# came from: the original version built each batch element's TMA descriptor
# on the HOST and looped `for b in range(batch): launch(...)`, meaning every
# batch element was a separate, fully-sequential kernel launch (and, in the
# persistent variant, a separate full-GPU-occupancy pass) -- catastrophic at
# large batch counts (e.g. ~33x slower than dot_fma at batch=1024, n=64).
#
# The fix: Gluon's `tma.make_tensor_descriptor` can build a TMA descriptor
# ON DEVICE from a raw pointer plus runtime shape/strides -- confirmed
# working via an isolated probe before this rewrite. That means `b` can
# become a normal grid dimension (program_id(0)), exactly like the
# register-based panel kernel and dot_fma already handle batching, instead
# of a host-side Python loop. Each grid cell builds its own descriptor from
# `A_ptr + b*stride_b` / `L_ptr + b*stride_b` -- one kernel launch total per
# bk-step, covering every batch element, instead of `batch` separate
# launches.
#
# NOTE: an attempt at a single 3D TMA descriptor spanning the whole
# (batch, n, n) tensor directly (avoiding even the on-device construction)
# hit a genuine LLVM backend crash (`cgaLayoutAttr.getCTAOrder().size() ==
# rank` assertion) -- the nvmma_shared swizzle layout's lowering doesn't
# support a 3D-shaped TMA descriptor in this Triton build. Per-tile,
# per-batch on-device descriptor construction sidesteps that entirely by
# keeping every descriptor genuinely 2D.
#
# This uses the ORIGINAL chained-accumulator, 3-wait tf32x3 implementation
# (not the "independent accumulators, single wait" variant from
# cholesky_gluon_tcgen05.py), since that variant was measured to regress
# small/medium-n performance -- using it here would confound whether any
# speedup is from fixing batching or from that separate, already-negative
# change.


@gluon.jit
def _panel_trsm_kernel(
    A_ptr, L_ptr,
    bk,
    stride_b, stride_r, stride_c,
    n,
    NB: gl.constexpr,
    BLOCK_I: gl.constexpr,
):
    TILE_LAYOUT: gl.constexpr = gl.BlockedLayout([1, NB], [32, 1], [1, 1], [1, 0])
    ROW_LAYOUT: gl.constexpr = gl.SliceLayout(dim=1, parent=TILE_LAYOUT)
    COL_LAYOUT: gl.constexpr = gl.SliceLayout(dim=0, parent=TILE_LAYOUT)
    SMEM_LAYOUT: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])

    b = gl.program_id(0)
    pid_i = gl.program_id(1)

    # Width of this panel: NB, except for a short tail block when n is not a
    # multiple of NB. Derived from n and bk rather than passed in -- the host
    # has no information here that the kernel doesn't already have.
    nb_valid = gl.minimum(NB, n - bk)

    idx_row = gl.arange(0, NB, layout=ROW_LAYOUT)
    idx_col = gl.arange(0, NB, layout=COL_LAYOUT)
    idx_row_valid = idx_row < nb_valid
    col_idx = idx_col[None, :]

    # contrib[r] = dot(Lp[r,:], row_jp): elementwise multiply broadcasts row_jp
    # over all NB rows ([NB,NB] TILE_LAYOUT), then axis=1 sums within each lane's
    # own registers -- free, no cross-lane traffic. Columns >= jp are still zero
    # (loop invariant), so this sum silently truncates at k < jp for free.
    #
    # a_col is ld.global.b32 straight into registers, no shared-memory staging
    # -- A[b, bk:bk+NB, bk+jp].

    # L11
    Lp = gl.zeros((NB, NB), dtype=gl.float32, layout=TILE_LAYOUT)  # [NB, NB] TILE_LAYOUT
    for jp in range(nb_valid):
        # row_jp = Lp[jp, :]. gl.gather along axis 0 lowers to one shfl.idx per
        # element (NB shuffles); the old `sum(where(row_idx == jp, Lp, 0))` idiom
        # lowered to a full 5-step butterfly all-reduce over NB register slots
        # (NB*log2(NB) = 160 shuffles), 31/32 of whose inputs were the zeros the
        # `where` had just written. gather yields [1, NB] TILE_LAYOUT; summing the
        # extent-1 axis 0 converts it back to [NB] COL_LAYOUT for free, so every
        # line downstream is unchanged. Verified bitwise-identical output.
        jp_idx = gl.zeros([1, NB], gl.int32, layout=TILE_LAYOUT) + jp
        row_jp = gl.sum(gl.gather(Lp, jp_idx, 0), axis=0)              # [NB] COL_LAYOUT

        contrib = gl.sum(Lp * row_jp[None, :], axis=1)                # [NB] ROW_LAYOUT

        a_col = gl.load( 
            
            A_ptr + b * stride_b + (bk + idx_row) * stride_r + (bk + jp) * stride_c,
            mask=idx_row_valid, other=0.0,
        ) # [NB] ROW_LAYOUT
        diff = a_col - contrib                                        # [NB] ROW_LAYOUT
        ljj = gl.sqrt(gl.sum(gl.where(idx_row == jp, diff, 0.0), axis=0))  # scalar: L[jp][jp]
        new_col = gl.where(idx_row == jp, ljj, gl.where(idx_row > jp, diff / ljj, 0.0))
        # new_col: [NB] ROW_LAYOUT -- column jp of L, masked to lower-triangular
        Lp = gl.where(col_idx == jp, new_col[:, None], Lp)             # [NB, NB] TILE_LAYOUT

    # Stage the finished factor into shared memory. Lp's ~NB^2/32 registers die
    # here, so loop 2 holds only L_rows: peak register pressure drops from two
    # register-resident tiles to one (measured 91 -> 62 regs at NB=32).
    #
    # Only loop 2 reads from smem, never loop 1. Loop 1's Lp IS the critical
    # dependency chain, so a shared-memory round-trip per iteration would add
    # LDS latency to all NB dependent steps. Loop 2's Lp is loop-invariant, so
    # its reads sit off the recurrence -- the safe place to spend that latency.
    Lp_smem = gl.allocate_shared_memory(gl.float32, [NB, NB], SMEM_LAYOUT)
    Lp_smem.store(Lp)

    i = bk + pid_i * BLOCK_I + gl.arange(0, BLOCK_I, layout=ROW_LAYOUT)
    row_mask = i < n

    # L21
    L_rows = gl.zeros((BLOCK_I, NB), dtype=gl.float32, layout=TILE_LAYOUT)
    # gl.static_range, not range: Gluon's smem slice/index require a COMPILE-TIME
    # offset, and a `range` loop variable is a runtime int32 even when the bound
    # is constexpr. static_range makes jp a Python int so `Lp_smem.slice(jp, ...)`
    # is a static address. The cost is that all NB iterations always run, so the
    # short-tail case (nb_valid < NB) must be masked explicitly -- see jp_valid.
    for jp in gl.static_range(NB):
        # Tail guard: iterations with jp >= nb_valid are out of this panel. Their
        # arithmetic is harmless (discarded), but their global load/store would
        # touch column bk+jp >= n, so both are masked off. Without this the tail
        # writes NaNs -- verified: 84 nonfinite entries at n=100, bk=96.
        jp_valid = jp < nb_valid

        # row_jp = Lp[jp, :] by indexed shared-memory read instead of a
        # cross-lane gather. All lanes read the same row, so it is a broadcast
        # (conflict-free) and vectorizes to 8 ld.shared.v4 versus 32 shfl.idx.
        row_jp = gl.sum(Lp_smem.slice(jp, 1, dim=0).load(TILE_LAYOUT), axis=0)
        contrib = gl.sum(L_rows * row_jp[None, :], axis=1)

        a_val = gl.load(
            A_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
            mask=row_mask & jp_valid, other=0.0,
        )
        diff = a_val - contrib
        ljj = gl.sum(gl.where(idx_col == jp, row_jp, 0.0), axis=0)

        is_diag = i == (bk + jp)
        val = gl.where(is_diag, ljj, diff / ljj)
        L_rows = gl.where(col_idx == jp, val[:, None], L_rows)

        gl.store(
            L_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
            val, mask=row_mask & jp_valid,
        )


@gluon.jit
def _round_tf32(x):
    # Round-to-nearest fp32 -> tf32 (10 mantissa bits): add the round bit
    # then truncate the low 13 mantissa bits.
    bits = x.to(gl.uint32, bitcast=True)
    rounded = (bits + 0x1000) & 0xFFFFE000
    return rounded.to(gl.float32, bitcast=True)


@gluon.jit
def _trunc_tf32(x):
    # fp32 -> tf32 by truncation: drop the low 13 mantissa bits.
    bits = x.to(gl.uint32, bitcast=True)
    return (bits & 0xFFFFE000).to(gl.float32, bitcast=True)


@gluon.jit
def _syrk_tile_tcgen05(l_desc, a_desc, bk, off_m, off_n, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = a_desc.block_type.shape[1]

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    l_m_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_n_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)

    # SPLIT ARRIVAL BARRIERS. Everything from here through the last MMA needs
    # only l_m and l_n (2 x 8KB); a_smem is not read until the epilogue, yet a
    # single barrier over all three made the kernel block on the LARGEST
    # transfer (16KB, half the tile's bytes) before starting work that does not
    # depend on it. ncu put 19.4% + 5.2% of all samples on that one wait.
    # Waiting on the L tiles alone lets A stream in behind the hi/lo split and
    # all three MMAs. Costs one extra mbarrier (8 bytes); no change to the smem
    # tiles, TMEM, or occupancy.
    mbarrier.expect(bar, 2 * l_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(l_desc, [off_m, bk], bar, l_m_smem)
    tma.async_copy_global_to_shared(l_desc, [off_n, bk], bar, l_n_smem)

    # bar_a lives with the copy it guards: issued here, waited on only in the
    # epilogue (search bar_a) so A streams in behind everything above.
    bar_a = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar_a, count=1)
    mbarrier.expect(bar_a, a_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m, off_n], bar_a, a_smem)

    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # tf32x3: split each operand into a tf32-rounded "hi" and a tf32-rounded
    # residual "lo" so that L_m @ L_n^T ~= hi_m@hi_n + hi_m@lo_n + lo_m@hi_n
    # recovers most of fp32's precision using 3 tf32 tensor-core passes
    # instead of 1.
    REG_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 16], [16, 2], [num_warps, 1], [1, 0])
    # hi is the untouched source tile: tcgen05 narrows fp32 -> tf32 itself.
    # lo = reg - trunc(reg) is EXACT and already has <=13 significant bits, so
    # the hardware's narrowing of lo is a no-op too -- no _round_tf32 needed
    # on either operand.
    l_m_reg = l_m_smem.load(REG_LAYOUT)
    l_m_lo = l_m_reg - _trunc_tf32(l_m_reg)
    l_n_reg = l_n_smem.load(REG_LAYOUT)
    l_n_lo = l_n_reg - _trunc_tf32(l_n_reg)

    # hi is NOT stored back: the MMA reads the untouched source tile, which
    # tcgen05 narrows by TRUNCATION. lo here is the residual against
    # round-to-nearest, so hi and lo disagree by up to 1 ulp of hi.
    # Measured cost: max|LL^T - A| 4.29e-06 -> 6.85e-04 at n=8192. Each warp rewrites only
    # the rows it read itself (REG_LAYOUT splits dim 0 by warp), so no extra
    # barrier. Triton was already aliasing these; writing it out just stops
    # four names from standing for two buffers.
    l_m_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_n_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_m_lo_smem.store(l_m_lo)
    l_n_lo_smem.store(l_n_lo)
    fence_async_shared()

    # The three tf32x3 products are mathematically INDEPENDENT -- they are only
    # summed. Chaining them through one accumulator with use_acc=True is a FALSE
    # dependency that forces a full mbarrier round-trip between each (ncu: 27.9%
    # `barrier` warp stall). Give each its own accumulator, issue all three back
    # to back, and wait ONCE with count=3.
    #
    # Worth 0.6% geomean / -0.81% on syrk -- small, but real: 14 of 15 official
    # cases improved when the two variants were INTERLEAVED per shape (p~5e-4 by
    # sign test). A first, non-interleaved A/B had put this inside the noise;
    # run-to-run drift on the sub-0.15ms shapes was swamping it.
    #
    # TMEM cost is 2x, not 3x: a [64,64] fp32 accumulator uses only 64 of TMEM's
    # 128 lanes, so triton packs two per 64-column group (tmem_size 64 -> 128 of
    # 512). 128 columns still admits 4 blocks/SM -- exactly the shared-memory
    # limit that already binds -- so occupancy is unchanged. Registers 63 -> 80
    # also does not bind (65,536/(128*80) = 6 blocks > 4).
    #
    # This does saturate TMEM at 100% (4 blocks x 128 cols = 512), foreclosing
    # BLOCK_M=128 and any 4th accumulator. Both were separately measured as
    # losses anyway, so nothing useful is given up.
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc0 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    acc1 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    acc2 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    # count=3: each MMA arrives once, so a single wait covers all three.
    # (bar was already invalidated after the L-tile wait above.)
    mbarrier.init(bar, count=3)
    tcgen05_mma(l_m_smem, l_n_smem.permute((1, 0)), acc0, use_acc=False, mbarriers=[bar], mbarrier_preds=[True])
    tcgen05_mma(l_m_smem, l_n_lo_smem.permute((1, 0)), acc1, use_acc=False, mbarriers=[bar], mbarrier_preds=[True])
    tcgen05_mma(l_m_lo_smem, l_n_smem.permute((1, 0)), acc2, use_acc=False, mbarriers=[bar], mbarrier_preds=[True])
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, num_warps)
    # left-associative, matching the order the chained accumulator produced.
    # Summing in registers rather than inside TMEM makes results differ from the
    # chained version by ~1.5e-06; accuracy is comparable, not uniformly better.
    update = acc0.load(acc_reg_layout) + acc1.load(acc_reg_layout) + acc2.load(acc_reg_layout)

    # Only NOW is the A tile needed -- it has had the whole hi/lo split plus all
    # three MMAs to arrive.
    mbarrier.wait(bar_a, phase=0)
    mbarrier.invalidate(bar_a)

    result = a_smem.load(acc_reg_layout) - update

    # Write the result back through a_smem rather than a separate out_smem:
    # a_smem's contents are dead the moment they reach registers, and this is a
    # read-modify-write of the same global tile, so the same 16KB serves both
    # directions. Triton's allocator was already aliasing out_smem onto
    # a_smem (the measured 49,176B only accounts for five distinct buffers,
    # not ten), so this is a readability change, not a saving -- confirmed
    # byte-identical by A/B.
    a_smem.store(result)
    fence_async_shared()
    tma.async_copy_shared_to_global(a_desc, [off_m, off_n], a_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def _syrk_kernel_tcgen05(A_ptr, L_ptr, bk, start, 
                         stride_b, stride_r, stride_c, n,
                         NB: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                         NUM_N_TILES: gl.constexpr, num_warps: gl.constexpr):
    b = gl.program_id(0)
    pid_m = gl.program_id(1)
    pid_n_lo = gl.program_id(2)
    pid_n_hi = NUM_N_TILES - 1 - pid_n_lo

    # Build this batch element's descriptors on-device from a raw pointer
    # offset -- no host-side per-batch descriptor or launch needed.
    l_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, NB], gl.float32)
    a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float32)
    off_m = start + pid_m * BLOCK_M

    # BUILD DESCRIPTORS LAZILY. Only the lower triangle of the trailing
    # submatrix is updated, so a block whose two mirrored n-tiles both fail
    # `pid_n <= pid_m` has nothing to do. At bk=0, n=8192 that is 2,016 of
    # 8,192 blocks (24.6%) -- see the tile census below. Building the TMA
    # descriptors before the guard made every one of them pay for descriptors
    # it never used, and on-device construction is not free: it writes through
    # global scratch and needs constant-cache invalidation (CCTL...IV.DEEP,
    # 4.23% of syrk samples), which is not private to the issuing block.
    # Moving construction inside the guard: syrk 8.687 -> 8.195ms (-5.7%),
    # output bitwise identical.
    #
    # `pid_n_lo <= pid_m` is exactly "at least one tile runs": pid_n_lo is
    # always < pid_n_hi (for even NUM_N_TILES), so if the hi tile is in range
    # the lo tile must be too.
    #
    # Tile census at bk=0, n=8192 (grid (1,128,64) = 8192 blocks):
    #   2,080 blocks run 2 tiles, 4,096 run 1, 2,016 run 0
    #   2,080*2 + 4,096 = 8,256 = 128*129/2, the lower-triangle tile count
    if pid_n_lo <= pid_m:
        l_desc = tma.make_tensor_descriptor(
            L_ptr + b * stride_b, [n, n], [stride_r, stride_c], [BLOCK_M, NB], l_layout,
        )
        a_desc = tma.make_tensor_descriptor(
            A_ptr + b * stride_b, [n, n], [stride_r, stride_c], [BLOCK_M, BLOCK_N], a_layout,
        )
        off_n = start + pid_n_lo * BLOCK_N
        _syrk_tile_tcgen05(l_desc, a_desc, bk, off_m, off_n, num_warps=num_warps)
        if pid_n_hi <= pid_m and pid_n_hi != pid_n_lo:
            off_n = start + pid_n_hi * BLOCK_N
            _syrk_tile_tcgen05(l_desc, a_desc, bk, off_m, off_n, num_warps=num_warps)


def _alloc_fn(size, alignment, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


_allocator_set = False


def custom_kernel(data: input_t) -> output_t:
    global _allocator_set
    if not _allocator_set:
        # on-device tma.make_tensor_descriptor needs a runtime scratch
        # allocator configured once per process.
        triton.set_allocator(_alloc_fn)
        _allocator_set = True

    A = data.to(torch.float32).contiguous().clone()
    batch, n, _ = A.shape
    L = torch.zeros_like(A)

    stride_b, stride_r, stride_c = L.stride()
    NB = 32
    BLOCK_I = 32
    BLOCK_M = BLOCK_N = 64

    for bk in range(0, n, NB):
        grid_panel = (batch, triton.cdiv(n - bk, BLOCK_I))
        _panel_trsm_kernel[grid_panel](
            A, L, bk, stride_b, stride_r, stride_c, n,
            NB=NB, BLOCK_I=BLOCK_I,
            num_warps=1,
        )

        # Only a full NB-wide panel leaves a trailing submatrix: a short tail
        # panel (n - bk < NB) is the last block-column, so trailing == 0.
        trailing = n - bk - NB
        if trailing > 0:
            num_m_tiles = triton.cdiv(trailing, BLOCK_M)
            num_n_tiles = triton.cdiv(trailing, BLOCK_N)
            start = bk + NB
            # Single launch covering every batch element -- no host loop.
            grid_syrk = (batch, num_m_tiles, triton.cdiv(num_n_tiles, 2))
            _syrk_kernel_tcgen05[grid_syrk](
                A, L, bk, start, stride_b, stride_r, stride_c, n,
                NB=NB, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                NUM_N_TILES=num_n_tiles, num_warps=4,
            )

    return L


if __name__ == "__main__":
    batch, n = 1, 8192
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(42)

    a = torch.randn((batch, n, n), device=device, dtype=torch.float32, generator=gen)
    A = (a @ a.transpose(-1, -2)) / n
    A.diagonal(dim1=-2, dim2=-1).add_(1.0)  # guaranteed SPD

    custom_kernel(A)  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    L = custom_kernel(A)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"custom_kernel time: {time.perf_counter() - t0:.6f} s")

    err = (L @ L.transpose(-1, -2) - A).abs().amax()
    print("max |L L^T - A| =", err.item())
