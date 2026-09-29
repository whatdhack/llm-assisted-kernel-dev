import time

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_reg_layout,
    tma, mbarrier, tcgen05_mma, tcgen05_commit, fence_async_shared,
)

from task import input_t, output_t

# Same Blackwell (cc 10.x) tcgen05 + tf32x3 SYRK step as cholesky_gluon_tcgen05.py,
# but with the SYRK grid made *persistent*: instead of launching one thread
# block per (batch, pid_m, pid_n) tile (up to batch * num_m_tiles *
# ceil(num_n_tiles/2) blocks per bk-step), a fixed number of blocks (one per
# SM) each loop internally over a share of the tiles -- now with batch folded
# into that same flattened tile-index space (see below), not a separate host
# loop.
#
# This is deliberately NOT warp-specialized -- every warp in a persistent block
# still does the same work together (load, split, mma, store), just repeated
# for many tiles per block instead of one block per tile. Two distinct costs
# get amortized by this alone, before any producer/consumer overlap:
#   1. Fewer thread-block launches for a given bk-step's SYRK grid, so less
#      GPU-side block scheduling overhead.
#   2. Shared-memory and Tensor Memory buffers are allocated ONCE per
#      persistent block and reused across all the tiles it processes, instead
#      of being allocated and torn down by every short-lived one-shot block
#      (the original _syrk_tile_tcgen05 allocated fresh l_m_smem, l_n_smem,
#      the three acc_tmem buffers, etc. on every single tile).
#
# Warp specialization (overlapping the load of tile i+1 with the compute of
# tile i across dedicated warps) is a separate, larger change layered on top
# of a persistent kernel like this one -- see the module docstring discussion
# in the conversation this file came from.
#
# Batching: cholesky_gluon_tcgen05.py originally built each batch element's
# TMA descriptor on the HOST and looped `for b in range(batch): launch(...)`
# -- catastrophic at large batch counts, since every batch element was a
# separate, fully-sequential launch (and here, a separate full-GPU-occupancy
# persistent pass, since num_persistent already claims every SM). The fix,
# validated in the conversation this file came from: `tma.make_tensor_
# descriptor` builds a TMA descriptor ON DEVICE from a raw pointer plus
# runtime shape/strides. That lets `b` become part of the same flattened
# tile-index space the persistent loop already walks (batch * num_m_tiles *
# num_n_lo_tiles total tiles), decoded via plain integer div/mod inside the
# loop, with each iteration building its own descriptor from
# `A_ptr + b*stride_b` / `L_ptr + b*stride_b` -- no host-side loop, no
# per-batch descriptor object, one launch total per bk-step covering every
# batch element and every tile.
#
# (A single 3D TMA descriptor spanning the whole (batch, n, n) tensor
# directly -- avoiding even the on-device construction -- was tried first and
# hit a genuine LLVM backend crash: `cgaLayoutAttr.getCTAOrder().size() ==
# rank`. The nvmma_shared swizzle layout's lowering doesn't support a
# 3D-shaped TMA descriptor in this Triton build. Per-tile on-device
# construction of a genuinely 2D descriptor sidesteps that.)


@gluon.jit
def _panel_trsm_kernel(
    A_ptr, L_ptr,
    bk,
    stride_b, stride_r, stride_c,
    n,
    NB: gl.constexpr,
    nb_valid: gl.constexpr,
    BLOCK_I: gl.constexpr,
):
    TILE_LAYOUT: gl.constexpr = gl.BlockedLayout([1, NB], [32, 1], [1, 1], [1, 0])
    ROW_LAYOUT: gl.constexpr = gl.SliceLayout(dim=1, parent=TILE_LAYOUT)
    COL_LAYOUT: gl.constexpr = gl.SliceLayout(dim=0, parent=TILE_LAYOUT)

    b = gl.program_id(0)
    pid_i = gl.program_id(1)

    idx_row = gl.arange(0, NB, layout=ROW_LAYOUT)
    idx_col = gl.arange(0, NB, layout=COL_LAYOUT)
    idx_row_valid = idx_row < nb_valid
    row_idx = idx_row[:, None]
    col_idx = idx_col[None, :]

    Lp = gl.zeros((NB, NB), dtype=gl.float32, layout=TILE_LAYOUT)
    for jp in range(nb_valid):
        row_jp = gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)
        contrib = gl.sum(Lp * row_jp[None, :], axis=1)

        a_col = gl.load(
            A_ptr + b * stride_b + (bk + idx_row) * stride_r + (bk + jp) * stride_c,
            mask=idx_row_valid, other=0.0,
        )
        diff = a_col - contrib
        ljj = gl.sqrt(gl.sum(gl.where(idx_row == jp, diff, 0.0), axis=0))
        new_col = gl.where(idx_row == jp, ljj, gl.where(idx_row > jp, diff / ljj, 0.0))
        Lp = gl.where(col_idx == jp, new_col[:, None], Lp)

    i_idx = gl.arange(0, BLOCK_I, layout=ROW_LAYOUT)
    i = bk + pid_i * BLOCK_I + i_idx
    row_mask = i < n

    L_rows = gl.zeros((BLOCK_I, NB), dtype=gl.float32, layout=TILE_LAYOUT)
    for jp in range(nb_valid):
        row_jp = gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)
        contrib = gl.sum(L_rows * row_jp[None, :], axis=1)

        a_val = gl.load(
            A_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
            mask=row_mask, other=0.0,
        )
        diff = a_val - contrib
        ljj = gl.sum(gl.where(idx_col == jp, row_jp, 0.0), axis=0)

        is_diag = i == (bk + jp)
        val = gl.where(is_diag, ljj, diff / ljj)
        L_rows = gl.where(col_idx == jp, val[:, None], L_rows)

        gl.store(
            L_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
            val, mask=row_mask,
        )


@gluon.jit
def _round_tf32(x):
    # Round-to-nearest fp32 -> tf32 (10 mantissa bits): add the round bit
    # then truncate the low 13 mantissa bits.
    bits = x.to(gl.uint32, bitcast=True)
    rounded = (bits + 0x1000) & 0xFFFFE000
    return rounded.to(gl.float32, bitcast=True)


@gluon.jit
def _syrk_do_tile(l_desc, a_desc, bk, off_m, off_n,
                   bar, l_m_smem, l_n_smem, a_smem,
                   l_m_hi_smem, l_m_lo_smem, l_n_hi_smem, l_n_lo_smem, out_smem,
                   acc_tmem_hh, acc_tmem_hl, acc_tmem_lh,
                   REG_LAYOUT: gl.constexpr, acc_reg_layout: gl.constexpr, num_warps: gl.constexpr):
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, 2 * l_desc.block_type.nbytes + a_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(l_desc, [off_m, bk], bar, l_m_smem)
    tma.async_copy_global_to_shared(l_desc, [off_n, bk], bar, l_n_smem)
    tma.async_copy_global_to_shared(a_desc, [off_m, off_n], bar, a_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    mbarrier.init(bar, count=1)

    # tf32x3 hi/lo split -- see cholesky_gluon_tcgen05.py for the derivation.
    l_m_reg = l_m_smem.load(REG_LAYOUT)
    l_m_hi = _round_tf32(l_m_reg)
    l_m_lo = _round_tf32(l_m_reg - l_m_hi)
    l_n_reg = l_n_smem.load(REG_LAYOUT)
    l_n_hi = _round_tf32(l_n_reg)
    l_n_lo = _round_tf32(l_n_reg - l_n_hi)

    l_m_hi_smem.store(l_m_hi)
    l_m_lo_smem.store(l_m_lo)
    l_n_hi_smem.store(l_n_hi)
    l_n_lo_smem.store(l_n_lo)
    fence_async_shared()

    # Independent accumulators (use_acc=False for all three) so the three
    # passes have no data dependency on each other; one commit + one wait
    # covers all three, relying on tcgen05's implicit pipelining for
    # same-shape/same-dtype calls issued back-to-back.
    tcgen05_mma(l_m_hi_smem, l_n_hi_smem.permute((1, 0)), acc_tmem_hh, use_acc=False)
    tcgen05_mma(l_m_hi_smem, l_n_lo_smem.permute((1, 0)), acc_tmem_hl, use_acc=False)
    tcgen05_mma(l_m_lo_smem, l_n_hi_smem.permute((1, 0)), acc_tmem_lh, use_acc=False)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    update = acc_tmem_hh.load(acc_reg_layout) + acc_tmem_hl.load(acc_reg_layout) + acc_tmem_lh.load(acc_reg_layout)

    a_reg = a_smem.load(acc_reg_layout)
    result = a_reg - update

    out_smem.store(result)
    fence_async_shared()
    tma.async_copy_shared_to_global(a_desc, [off_m, off_n], out_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def _syrk_kernel_tcgen05_persistent(A_ptr, L_ptr, bk, start,
                                    stride_b, stride_r, stride_c, n, batch,
                                    NB: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                    NUM_M_TILES: gl.constexpr, NUM_N_TILES: gl.constexpr,
                                    num_warps: gl.constexpr):
    l_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, NB], gl.float32)
    a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float32)

    # Buffers, mbarrier, and TMEM accumulators allocated ONCE per persistent
    # block and reused for every tile the block processes -- see the module
    # docstring above. Only the TMA descriptors themselves (built per
    # iteration below, since the batch element can change) depend on `b`.
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    l_m_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, NB], l_layout)
    l_n_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, NB], l_layout)
    a_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, BLOCK_N], a_layout)
    l_m_hi_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, NB], l_layout)
    l_m_lo_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, NB], l_layout)
    l_n_hi_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, NB], l_layout)
    l_n_lo_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, NB], l_layout)
    out_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, BLOCK_N], a_layout)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem_hh = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    acc_tmem_hl = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    acc_tmem_lh = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    REG_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 16], [16, 2], [num_warps, 1], [1, 0])
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, num_warps)

    pid = gl.program_id(0)
    num_programs = gl.num_programs(0)
    NUM_N_LO_TILES: gl.constexpr = gl.cdiv(NUM_N_TILES, 2)
    TILES_PER_BATCH: gl.constexpr = NUM_M_TILES * NUM_N_LO_TILES
    total_tiles = batch * TILES_PER_BATCH

    for idx in range(pid, total_tiles, num_programs):
        b = idx // TILES_PER_BATCH
        local_idx = idx % TILES_PER_BATCH
        pid_m = local_idx // NUM_N_LO_TILES
        pid_n_lo = local_idx % NUM_N_LO_TILES
        pid_n_hi = NUM_N_TILES - 1 - pid_n_lo
        off_m = start + pid_m * BLOCK_M

        l_desc = tma.make_tensor_descriptor(
            L_ptr + b * stride_b, [n, n], [stride_r, stride_c], [BLOCK_M, NB], l_layout,
        )
        a_desc = tma.make_tensor_descriptor(
            A_ptr + b * stride_b, [n, n], [stride_r, stride_c], [BLOCK_M, BLOCK_N], a_layout,
        )

        if pid_n_lo <= pid_m:
            _syrk_do_tile(l_desc, a_desc, bk, off_m, start + pid_n_lo * BLOCK_N,
                         bar, l_m_smem, l_n_smem, a_smem,
                         l_m_hi_smem, l_m_lo_smem, l_n_hi_smem, l_n_lo_smem, out_smem,
                         acc_tmem_hh, acc_tmem_hl, acc_tmem_lh,
                         REG_LAYOUT=REG_LAYOUT, acc_reg_layout=acc_reg_layout, num_warps=num_warps)
        if pid_n_hi <= pid_m and pid_n_hi != pid_n_lo:
            _syrk_do_tile(l_desc, a_desc, bk, off_m, start + pid_n_hi * BLOCK_N,
                         bar, l_m_smem, l_n_smem, a_smem,
                         l_m_hi_smem, l_m_lo_smem, l_n_hi_smem, l_n_lo_smem, out_smem,
                         acc_tmem_hh, acc_tmem_hl, acc_tmem_lh,
                         REG_LAYOUT=REG_LAYOUT, acc_reg_layout=acc_reg_layout, num_warps=num_warps)


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

    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count

    for bk in range(0, n, NB):
        nb_valid = min(NB, n - bk)

        grid_panel = (batch, triton.cdiv(n - bk, BLOCK_I))
        _panel_trsm_kernel[grid_panel](
            A, L, bk, stride_b, stride_r, stride_c, n,
            NB=NB, nb_valid=nb_valid, BLOCK_I=BLOCK_I,
            num_warps=1,
        )

        trailing = n - (bk + nb_valid)
        if trailing > 0:
            assert nb_valid == NB, "SYRK only ever runs with a full NB-wide panel"
            num_m_tiles = triton.cdiv(trailing, BLOCK_M)
            num_n_tiles = triton.cdiv(trailing, BLOCK_N)
            num_n_lo_tiles = triton.cdiv(num_n_tiles, 2)
            total_tiles = batch * num_m_tiles * num_n_lo_tiles
            num_persistent = min(num_sms, total_tiles)
            start = bk + nb_valid
            # Single launch covering every batch element and every tile --
            # no host loop, batch is folded into the persistent tile index.
            _syrk_kernel_tcgen05_persistent[(num_persistent,)](
                A, L, bk, start, stride_b, stride_r, stride_c, n, batch,
                NB=NB, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                NUM_M_TILES=num_m_tiles, NUM_N_TILES=num_n_tiles, num_warps=4,
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
