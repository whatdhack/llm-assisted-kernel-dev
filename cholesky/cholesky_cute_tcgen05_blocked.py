import time

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute import nvgpu
from cutlass.cute.nvgpu import cpasync, tcgen05
# In-kernel event tracing. CuTe DSL strips every iket op at JIT time unless
# CUTE_DSL_COMPILER_OPT contains `iket` (run-iket sets it), so the events below
# cost nothing in a normal run. See modal/iket_cute_b200.py.
from cutlass.cute.experimental import iket
from cutlass.cute.runtime import from_dlpack

from task import input_t, output_t

# CuTe DSL port of cholesky_gluon_tcgen05_blocked.py -- same blocked
# right-looking Cholesky, same Blackwell (cc 10.x) tcgen05 + tf32x3 SYRK step,
# same NB=32 / BLOCK_M=BLOCK_N=64 tiling, restated in CUTLASS's Python DSL
# rather than Gluon.
#
# Three things change, all of them because CuTe DSL has facilities Gluon does
# not, not because the algorithm changed:
#
# 1. NO ON-DEVICE TMA DESCRIPTOR CONSTRUCTION. The Gluon version calls
#    `tma.make_tensor_descriptor` inside the kernel, per block, because that
#    Triton build could not lower a 3D-shaped descriptor spanning the whole
#    (batch, n, n) tensor -- the nvmma_shared swizzle hit an
#    `cgaLayoutAttr.getCTAOrder().size() == rank` assertion in LLVM. Its
#    comments record both the workaround and its cost: on-device construction
#    writes through global scratch and needs a constant-cache invalidation
#    (CCTL...IV.DEEP, 4.23% of syrk samples), which is why that version has to
#    build descriptors lazily inside the `pid_n_lo <= pid_m` guard.
#    CUTLASS's TMA atoms are genuinely 3-mode, so here the batch is just the
#    L-mode of one host-built descriptor and the guard is only a guard again.
#
# 2. TILE OFFSETS COME FROM `cute.domain_offset`, NOT FROM TILE INDICES.
#    off_m = start + pid_m*BLOCK_M with start = bk + NB is a multiple of 32 but
#    not of BLOCK_M=64, so `local_tile` with a tile index cannot address it.
#    Shifting the TMA tensor's origin by the runtime (off_m, off_n) and then
#    taking tile (0,0) gives the same arbitrary element coordinates Gluon
#    passes to `tma.async_copy_global_to_shared` directly.
#
# 3. THE hi/lo SPLIT RUNS OVER RAW SMEM ADDRESSES. tf32x3 needs
#    `lo = x - trunc_tf32(x)` written into a second smem tile with the same
#    swizzled layout as the first. Applying the same address permutation to
#    source and destination makes the logical-coordinate walk unnecessary, so
#    the split is a flat 1D pass over the 2048 floats of the tile -- fully
#    vectorized, and provably the same tile.
#
# The numerics match: the same chained-Lp panel recurrence, the same tf32x3
# products summed left-associatively in registers, the same truncation-based
# `lo` residual. Measured on a B200 against the Gluon original, max|LL^T - A|
# tracks it to within one fp32 ulp at every size tested (n = 64 .. 2048,
# batch 1..4): 4.8e-07 vs 4.8e-07 at n=128, 1.3e-06 vs 1.2e-06 at n=2048,
# with the two factors differing by 1.2e-07 to 2.4e-07 and no nonfinite
# entries. Both sit at torch.linalg.cholesky's own reconstruction error.
#
# Speed is not at parity. Over the 15 official leaderboard shapes the geomean
# is 2.61 ms against Gluon's 2.05 ms (1.28x) -- 1.08x at best, widening to
# 1.78x at the largest shape (batch 1, n 32768). Both beat cuSOLVER's 2.37 ms
# geomean.
#
# It is ALL the SYRK, and it is uniform. Kineto split at batch 1 / n 8192:
#
#                     cute       gluon     ratio
#     GPU busy      18.241 ms  12.481 ms   1.46x
#       panel (256)  4.661      4.407      1.06x
#       syrk  (255) 13.543      8.037      1.69x
#     host gap       1.217      0.936      1.30x
#
# SYRK is 5.51 ms of the 5.76 ms difference. Per launch it is 1.75x-1.83x
# slower on all of the first 24 (mean 1.78x) -- nothing alternates, nothing
# is at parity. The panel is level after _dot_tree below, and the host-side
# launch gap is 0.28 ms and cannot matter.
#
# Occupancy is NOT the cause: both run 4 blocks/SM bounded by registers and
# shared memory, and cutting TMEM use in half or to a quarter (NUM_ACC below)
# changes nothing. The port also executes 18% FEWER instructions than Gluon's
# SYRK (13.3M vs 16.2M), so it is stalling rather than doing more work.
# NOT YET EXPLAINED, and three plausible explanations have been measured and
# ruled out. The epilogue's A read-modify-write does 32 scalar shared loads +
# 32 scalar shared stores per warp per tile at a 59-61% bank-conflict rate,
# against Gluon's ~14 each at 12-20%, for 1.8x-2.1x the shared-memory
# wavefronts -- but every intervention on that has bought exactly nothing:
# vectorizing the store via CUTLASS's r2s/stmatrix path (store requests 4.8x
# fewer, wavefronts unchanged, conflict degree simply rises 4.1 -> 19.7-way),
# and six different sA layouts (all within +-0.1%). Treat the traffic
# difference as correlation, not cause.
#
# The strongest live lead is that ncu times the second syrk launch at 76-89us
# while Kineto times the SAME launch at 152us. ncu serializes kernels and
# flushes caches between them, so this kernel is near parity in ISOLATION and
# 1.75x slower back-to-back with its neighbours -- which would point between
# launches (TMEM alloc/dealloc, smem drain, launch tail) rather than anywhere
# in the code below, and would explain why three body-level fixes did nothing.
# See the leaderboard file for the full record.
#
# Three things were tried against that gap and are recorded here so they are
# not retried: NUM_ACC = 1/2/3 (~2%, see below), the `read` vs full variant of
# the TMA store wait (nothing), and an unswizzled full-width A tile that turns
# the two 64x32 TMA boxes into one 64x64 box (3.23 ms geomean -- clearly
# worse, the epilogue's shared-memory bank conflicts cost more than the extra
# TMA instruction saves).

NB = 32
BLOCK_I = 32
BLOCK_M = 64
BLOCK_N = 64
MMA_TILER = (BLOCK_M, BLOCK_N, NB)

PANEL_THREADS = 32
SYRK_THREADS = 128
SPLIT_PER_THREAD = (BLOCK_M * NB) // SYRK_THREADS

# How many TMEM accumulators the three tf32x3 products are spread over, and
# which accumulator each product lands in.
#
# The Gluon original uses three, to remove the false dependency between
# products that a single chained accumulator creates (it measured 27.9% of
# warp cycles stalled on `barrier` with one). That does not reproduce here:
# measured geomean over the 15 leaderboard shapes, twice, on a B200 --
#
#              1 acc      2 accs     3 accs
#   run A      2.833 ms   2.789 ms   2.848 ms
#   run B      2.788 ms   2.753 ms   2.814 ms
#
# -- so the chaining cost is inside the noise either way, and two wins by ~2%
# in both runs while halving TMEM (128 of 512 columns rather than 256). The
# TMEM saving turns out not to be why: the 1-accumulator variant frees even
# more and is no faster, and ncu reports 4 blocks/SM bounded by registers and
# shared memory, not by TMEM, in every variant.
NUM_ACC = 2
TERM_ACC = {1: (0, 0, 0), 2: (0, 0, 1), 3: (0, 1, 2)}[NUM_ACC]
# 4 x [BLOCK_M, NB] L tiles + 1 x [BLOCK_M, BLOCK_N] A tile, plus a 1KB head
# for the two mbarriers and the TMEM-allocation slot.
SYRK_SMEM_BYTES = 1024 + (4 * BLOCK_M * NB + BLOCK_M * BLOCK_N) * 4


# ---------------------------------------------------------------------------
# Panel factorization + TRSM (pure SIMT, one warp, no tensor cores)
# ---------------------------------------------------------------------------
@cute.jit
def _dot_tree(x: cute.Tensor, y: cute.Tensor, scratch: cute.Tensor):
    """sum_k x[k] * y[k], summed as a binary TREE rather than a running
    accumulator -- hence the name.

    The naive form is a chain, and every step waits on the one before it:

        contrib = 0
        for k in range(NB):
            contrib = contrib + x[k] * y[k]   # FMA k+1 needs FMA k's result

    That is NB dependent FMAs at ~4 cycles each, ~128 cycles of pure latency
    per panel iteration, and it sits directly on the recurrence the whole
    panel kernel is latency-bound on.

    The tree below does the identical arithmetic -- NB multiplies, NB-1 adds
    -- with a shorter critical path:

        scratch[k] = x[k] * y[k]                NB independent multiplies
        w = NB//2:  scratch[k] += scratch[k+w]  NB/2 independent adds
        w = NB//4:  scratch[k] += scratch[k+w]  NB/4
        ... down to w == 1, then scratch[0] is the sum

    Each pass halves the live width, so the depth is log2(NB) = 5 dependent
    adds instead of NB-1 = 31, and the adds within a pass are independent of
    each other, so they pipeline.

    `scratch` (rTmp at the call sites) exists for exactly this: the tree has
    to hold the NB products and fold them in place, where an accumulator
    would have needed one register. Gluon gets the same shape for free from
    `gl.sum(..., axis=1)`; in CuTe DSL it is written out.
    """
    for k in cutlass.range_constexpr(NB):
        scratch[k] = x[k] * y[k]
    w: cutlass.Constexpr = NB
    while cutlass.const_expr(w > 1):
        w = w // 2
        for k in cutlass.range_constexpr(w):
            scratch[k] = scratch[k] + scratch[k + w]
    return scratch[0]


@cute.kernel
def _panel_trsm_kernel(
    mA: cute.Tensor,
    mL: cute.Tensor,
    bk: cutlass.Int32,
    n: cutlass.Int32,
):
    """One warp per BLOCK_I rows. Thread t owns row t of the panel, all NB
    columns of it, exactly as Gluon's BlockedLayout([1, NB], [32, 1], [1, 1])
    gives every lane one row and NB registers."""
    tid, _, _ = cute.arch.thread_idx()
    pid_i, b, _ = cute.arch.block_idx()

    # Width of this panel: NB, except for a short tail block when n is not a
    # multiple of NB. Derived from n and bk rather than passed in -- the host
    # has no information here that the kernel doesn't already have.
    nb_valid = cute.min(cutlass.Int32(NB), n - bk)
    last = nb_valid - 1

    # L11: factor the NB x NB diagonal block. rLp[k] is Lp[tid, k].
    #
    # Fully unrolled (range_constexpr) rather than Gluon's dynamic
    # `range(nb_valid)`: every index into rLp is then a Python int, so the
    # panel stays in registers instead of spilling to local memory the way a
    # runtime-indexed array would. Iterations past nb_valid are masked to zero
    # rather than skipped, which is what makes the unroll legal.
    iket.range_push("panel_L11")
    rLp = cute.make_rmem_tensor(NB, cutlass.Float32)
    rRow = cute.make_rmem_tensor(NB, cutlass.Float32)
    rTmp = cute.make_rmem_tensor(NB, cutlass.Float32)
    for k in cutlass.range_constexpr(NB):
        rLp[k] = cutlass.Float32(0.0)

    for jp in cutlass.range_constexpr(NB):
        jp_valid = cutlass.Int32(jp) < nb_valid

        # rRow = lane jp's whole rLp array, copied into EVERY lane. Since
        # rLp[k] is Lp[tid, k], lane jp's rLp IS the pivot row Lp[jp, :].
        # Needed because _dot_tree below mixes two rows: rLp is this lane's
        # own row, while the pivot row lives only in lane jp's registers.
        #
        # shuffle_sync is a register-to-register exchange across the warp --
        # no memory, no separate barrier. Its `_sync` half IS a warp-scoped
        # barrier fused into the instruction: the lanes in the member mask
        # converge before the exchange, which is what makes this well defined
        # under Volta-and-later independent thread scheduling, where lanes
        # have their own program counters and are NOT guaranteed to run in
        # lockstep. It orders no memory -- only register values move. The two
        # shuffles in this loop (rRow here, ljj below) therefore have to sit
        # OUTSIDE the `tid == jp` / `tid > jp` divergence below: a masked-in
        # lane that never arrives is undefined behaviour, typically a hang.
        #
        # Defaults taken (cutlass/cute/arch/nvvm_wrappers.py): mask=FULL_MASK
        # so all 32 lanes participate, kind=idx so `jp` is a source lane and
        # not an offset, mask_and_clamp=31 so no sub-warp segmentation --
        # i.e. exactly __shfl_sync(0xffffffff, rLp[k], jp).
        #
        # jp is a Python int here, so each of these is one shfl.idx with a
        # static lane -- the same NB shuffles Gluon's `gl.gather(Lp, jp_idx,
        # 0)` lowers to, and not the NB*log2(NB) butterfly that the older
        # `sum(where(row_idx == jp, Lp, 0))` idiom cost.
        for k in cutlass.range_constexpr(NB):
            rRow[k] = cute.arch.shuffle_sync(rLp[k], jp)

        # contrib = dot(rLp, rRow) = sum_k Lp[tid, k] * Lp[jp, k], with rTmp
        # as the tree's scratch. rLp[k] for k >= jp is still zero (loop
        # invariant), so this truncates at k < jp for free.
        contrib = _dot_tree(rLp, rRow, rTmp)

        # Clamp instead of predicate: the row/column are pulled inside the
        # panel and the value selected away, so the load is always in bounds
        # and stays a plain ld.global.b32 with no branch.
        row = bk + cute.min(cutlass.Int32(tid), last)
        col = bk + cute.min(cutlass.Int32(jp), last)
        a_col = mA[(row, col, b)]
        a_col = cutlass.Float32(0.0) if cutlass.Int32(tid) >= nb_valid else a_col

        diff = a_col - contrib # diff[tid] = A[tid,jp] - Σ_k L[tid,k]·L[jp,k]
        # ljj = L[jp, jp] = sqrt(lane jp's diff), broadcast to every lane so
        # each one can scale its own diff below. Same full-mask shfl.idx as
        # the rRow loop above, one value instead of NB.
        ljj = cute.math.sqrt(cute.arch.shuffle_sync(diff, jp))
        val = cutlass.Float32(0.0)
        if cutlass.Int32(tid) == cutlass.Int32(jp):
            val = ljj
        elif cutlass.Int32(tid) > cutlass.Int32(jp):
            val = diff / ljj
        rLp[jp] = val if jp_valid else cutlass.Float32(0.0)

    iket.range_pop()

    # Stage the finished factor into shared memory. rLp dies here, so loop 2
    # holds only rL: peak register pressure is one register-resident tile, not
    # two. Only loop 2 reads from smem, never loop 1 -- loop 1's Lp IS the
    # critical dependency chain, so an LDS round-trip per iteration would land
    # on all NB dependent steps, while loop 2's Lp is loop-invariant and its
    # reads sit off the recurrence.
    iket.range_push("panel_stage")
    smem = utils.SmemAllocator()
    sLp = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((NB, NB), stride=(NB, 1)), byte_alignment=16
    )
    # One 128-byte row per lane: 8 x st.shared.v4, conflict-free in 4 phases.
    # Storing element-by-element would be 32 scalar stores at a 128B stride --
    # a 32-way bank conflict every time.
    cute.autovec_copy(rLp, sLp[(cutlass.Int32(tid), None)])
    cute.arch.barrier()
    iket.range_pop()

    # L21: the rows below the diagonal block.
    iket.range_push("panel_L21")
    i = bk + cutlass.Int32(pid_i) * BLOCK_I + cutlass.Int32(tid)
    row_ok = i < n
    i_cl = cute.min(i, n - 1)

    rL = cute.make_rmem_tensor(NB, cutlass.Float32)
    for k in cutlass.range_constexpr(NB):
        rL[k] = cutlass.Float32(0.0)

    for jp in cutlass.range_constexpr(NB):
        jp_valid = cutlass.Int32(jp) < nb_valid

        # rRow = the pivot row Lp[jp, :] again, but by indexed shared-memory
        # read from sLp instead of the cross-lane shuffle loop 1 uses: every
        # lane reads the same row, so it is a broadcast (conflict-free) and
        # vectorizes to 8 ld.shared.v4 rather than 32 shfl.idx.
        cute.autovec_copy(sLp[(jp, None)], rRow)

        # contrib = dot(rL, rRow); rL is this lane's row of L21, not rLp.
        contrib = _dot_tree(rL, rRow, rTmp)

        col = bk + cute.min(cutlass.Int32(jp), last)
        a_val = mA[(i_cl, col, b)]
        keep = row_ok and jp_valid
        a_val = a_val if keep else cutlass.Float32(0.0)

        diff = a_val - contrib
        ljj = rRow[jp]                                  # Lp[jp, jp]
        val = ljj if i == bk + jp else diff / ljj
        rL[jp] = val if jp_valid else cutlass.Float32(0.0)

        # Tail guard: iterations with jp >= nb_valid are out of this panel.
        # Their arithmetic is discarded, but the store would touch column
        # bk+jp >= n, so it is masked off -- without this the tail writes NaNs.
        if keep:
            mL[(i, bk + jp, b)] = val
    iket.range_pop()


@cute.jit
def _panel_launcher(
    mA: cute.Tensor,
    mL: cute.Tensor,
    bk: cutlass.Int32,
    n: cutlass.Int32,
    batch: cutlass.Int32,
):
    num_i = cute.ceil_div(n - bk, BLOCK_I)
    _panel_trsm_kernel(mA, mL, bk, n).launch(
        grid=(num_i, batch, 1),
        block=(PANEL_THREADS, 1, 1),
    )


# ---------------------------------------------------------------------------
# Trailing-submatrix SYRK update: tcgen05 tf32x3
# ---------------------------------------------------------------------------
def _epi_view(acc: cute.Tensor) -> cute.Tensor:
    """(MMA, MMA_M, MMA_N) TMEM accumulator -> the plain (M, N) view the TMEM
    load atom is built and sliced against."""
    return cute.flat_divide(acc[((None, None), 0, 0)], (BLOCK_M, BLOCK_N))

@cute.jit
def _split_hi_lo(sSrc_flat: cute.Tensor, sDst_flat: cute.Tensor, tidx: cutlass.Int32):
    """lo = x - trunc_tf32(x), written into a second tile with the SAME
    swizzled layout as the first.

    Because the destination repeats the source's address permutation exactly,
    walking logical (m, k) coordinates is unnecessary: the two tiles are
    element-for-element identical under any single address map. So this is a
    flat pass over the tile's raw floats, 16 per thread, which vectorizes to
    4 x ld.shared.v4 + 4 x st.shared.v4 and no swizzle arithmetic at all.

    `hi` is never materialized: tcgen05 narrows the untouched source tile
    fp32 -> tf32 by truncation itself, so `hi` IS the source. And lo, being
    exact and already <= 13 significant bits wide, survives that same
    narrowing unchanged -- which is why neither operand needs rounding.
    """
    rU = cute.make_rmem_tensor(SPLIT_PER_THREAD, cutlass.Uint32)
    rX = cute.recast_tensor(rU, cutlass.Float32)         # same registers, float view
    rHiU = cute.make_rmem_tensor(SPLIT_PER_THREAD, cutlass.Uint32)
    rHi = cute.recast_tensor(rHiU, cutlass.Float32)
    rLo = cute.make_rmem_tensor(SPLIT_PER_THREAD, cutlass.Float32)

    cute.autovec_copy(sSrc_flat[(None, tidx)], rU)
    for i in cutlass.range_constexpr(SPLIT_PER_THREAD):
        rHiU[i] = rU[i] & cutlass.Uint32(0xFFFFE000)     # drop the low 13 mantissa bits
    for i in cutlass.range_constexpr(SPLIT_PER_THREAD):
        rLo[i] = rX[i] - rHi[i]
    cute.autovec_copy(rLo, sDst_flat[(None, tidx)])


@cute.kernel
def _syrk_kernel_tcgen05(
    tma_atom_lm: cute.CopyAtom,
    mL_a: cute.Tensor,
    tma_atom_ln: cute.CopyAtom,
    mL_b: cute.Tensor,
    tma_atom_a_ld: cute.CopyAtom,
    mA_ld: cute.Tensor,
    tma_atom_a_st: cute.CopyAtom,
    mA_st: cute.Tensor,
    sLa_layout: cute.ComposedLayout,
    sLb_layout: cute.ComposedLayout,
    sA_layout: cute.ComposedLayout,
    tiled_mma: cute.TiledMma,
    bk: cutlass.Int32,
    start: cutlass.Int32,
    num_n_tiles: cutlass.Int32,
    num_tmem_cols: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    pid_n_lo, pid_m, b = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    iket.mark("syrk_start", cutlass.Int32(pid_m))

    # Mirror the n-tile pair so one block covers a low and a high column tile;
    # `pid_n_lo <= pid_m` is exactly "at least one of them is in the lower
    # triangle", since pid_n_lo < pid_n_hi whenever the two differ.
    pid_n_hi = num_n_tiles - 1 - cutlass.Int32(pid_n_lo)
    off_m = start + cutlass.Int32(pid_m) * BLOCK_M

    smem = utils.SmemAllocator()
    l_elems = BLOCK_M * NB
    a_elems = BLOCK_M * BLOCK_N

    p_lm = smem.allocate_array(cutlass.Float32, l_elems, byte_alignment=1024)
    p_ln = smem.allocate_array(cutlass.Float32, l_elems, byte_alignment=1024)
    p_lm_lo = smem.allocate_array(cutlass.Float32, l_elems, byte_alignment=1024)
    p_ln_lo = smem.allocate_array(cutlass.Float32, l_elems, byte_alignment=1024)
    p_a = smem.allocate_array(cutlass.Float32, a_elems, byte_alignment=1024)
    mbar = smem.allocate_array(cutlass.Int64, 2, byte_alignment=8)
    tmem_holding = smem.allocate_array(cutlass.Int32, 1, byte_alignment=8)

    # Two views of each L tile over the very same bytes: the swizzled one the
    # MMA consumes, and a flat (16, 128) one the hi/lo split walks.
    flat_layout: cutlass.Constexpr = cute.make_layout(
        (SPLIT_PER_THREAD, SYRK_THREADS), stride=(1, SPLIT_PER_THREAD)
    )
    # The MMA reads through the swizzle, so it gets a pointer carrying it; the
    # hi/lo split reads raw addresses, so it gets the bare pointer. Same bytes.
    def _mma_view(ptr, lay):
        return cute.make_tensor(
            cute.recast_ptr(ptr, lay.inner, dtype=cutlass.Float32), lay.outer
        )

    sLm = _mma_view(p_lm, sLa_layout)
    sLn = _mma_view(p_ln, sLb_layout)
    sLm_lo = _mma_view(p_lm_lo, sLa_layout)
    sLn_lo = _mma_view(p_ln_lo, sLb_layout)
    sA = _mma_view(p_a, sA_layout)

    fLm = cute.make_tensor(cute.recast_ptr(p_lm, dtype=cutlass.Uint32), flat_layout)
    fLn = cute.make_tensor(cute.recast_ptr(p_ln, dtype=cutlass.Uint32), flat_layout)
    fLm_lo = cute.make_tensor(p_lm_lo, flat_layout)
    fLn_lo = cute.make_tensor(p_ln_lo, flat_layout)

    # SPLIT ARRIVAL BARRIERS. Everything from the split through the last MMA
    # needs only the two L tiles (2 x 8KB); the A tile is not read until the
    # epilogue, yet one barrier over all three would block on the LARGEST
    # transfer (16KB, half the tile's bytes) before starting work that does not
    # depend on it -- 19.4% + 5.2% of all ncu samples sat on that one wait.
    # Waiting on the L tiles alone lets A arrive in the background, behind the
    # split and all three MMAs, for the price of one extra mbarrier (8 bytes).
    bar_l = mbar + 0
    bar_a = mbar + 1
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(bar_l, 1)
            cute.arch.mbarrier_init(bar_a, 1)
    cute.arch.mbarrier_init_fence()

    # NUM_ACC accumulators, allocated once for the whole block rather than per
    # tile. Each [64, 64] fp32 accumulator claims 64 of TMEM's 512 columns and
    # the allocation rounds up to a power of two, so 1/2/3 of them cost
    # 64/128/256 columns. See NUM_ACC at the top for what that measured.
    acc_shape = tiled_mma.partition_shape_C(MMA_TILER[:2])
    acc_fake = tiled_mma.make_fragment_C(acc_shape)
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1, num_threads=SYRK_THREADS
    )
    tmem = utils.TmemAllocator(tmem_holding, barrier_for_retrieve=tmem_alloc_barrier)
    # Wraps the TMEM permit wait: a block that cannot get a permit shows here.
    iket.range_push("tmem_alloc")
    tmem.allocate(num_tmem_cols)
    tmem.wait_for_alloc()
    iket.range_pop()
    acc_ptr = tmem.retrieve_ptr(cutlass.Float32)
    acc_cols = tcgen05.find_tmem_tensor_col_offset(acc_fake)
    accs = [
        cute.make_tensor(acc_ptr + i * acc_cols, acc_fake.layout)
        for i in range(NUM_ACC)
    ]

    # The epilogue tile is the whole MMA tile, so the (EPI_M, EPI_N) modes of
    # the flat_divide are both 1 and get sliced straight back off; the divide
    # is still needed to hand make_tmem_copy a plain (M, N) view.
    op_t2r = sm100_utils.get_tmem_load_op(
        MMA_TILER, utils.LayoutEnum.ROW_MAJOR, cutlass.Float32, cutlass.Float32,
        (BLOCK_M, BLOCK_N), False,
    )
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        op_t2r, _epi_view(accs[0])[(None, None, 0, 0)]
    )
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)


    if pid_n_lo <= cutlass.Int32(pid_m):
        _syrk_tile(
            tma_atom_lm, mL_a, tma_atom_ln, mL_b,
            tma_atom_a_ld, mA_ld, tma_atom_a_st, mA_st,
            sLm, sLn, sLm_lo, sLn_lo, sA,
            fLm, fLn, fLm_lo, fLn_lo,
            tiled_mma, tiled_copy_t2r, thr_copy_t2r, accs,
            bar_l, bar_a, tidx, warp_idx, b, bk, off_m,
            start + pid_n_lo * BLOCK_N, 0,
        )
        if pid_n_hi <= cutlass.Int32(pid_m) and pid_n_hi != cutlass.Int32(pid_n_lo):
            _syrk_tile(
                tma_atom_lm, mL_a, tma_atom_ln, mL_b,
                tma_atom_a_ld, mA_ld, tma_atom_a_st, mA_st,
                sLm, sLn, sLm_lo, sLn_lo, sA,
                fLm, fLn, fLm_lo, fLn_lo,
                tiled_mma, tiled_copy_t2r, thr_copy_t2r, accs,
                bar_l, bar_a, tidx, warp_idx, b, bk, off_m,
                start + pid_n_hi * BLOCK_N, 1,
            )

    # Give up the allocation permit before freeing, so the next CTA scheduled
    # on this SM can allocate: dealloc alone does not release it.
    iket.range_push("tmem_free")
    tmem.relinquish_alloc_permit()
    cute.arch.barrier()
    tmem.free(acc_ptr, num_tmem_cols)
    iket.range_pop()


@cute.jit
def _syrk_tile(
    tma_atom_lm, mL_a, tma_atom_ln, mL_b,
    tma_atom_a_ld, mA_ld, tma_atom_a_st, mA_st,
    sLm, sLn, sLm_lo, sLn_lo, sA,
    fLm, fLn, fLm_lo, fLn_lo,
    tiled_mma, tiled_copy_t2r, thr_copy_t2r, accs,
    bar_l, bar_a, tidx, warp_idx, b, bk, off_m, off_n, a_phase: cutlass.Constexpr,
):
    """One [BLOCK_M, BLOCK_N] tile of A -= L_m @ L_n^T, in tf32x3."""
    # Shift each TMA tensor's origin to this tile's element coordinates, then
    # take tile (0, 0): off_m and off_n are multiples of NB but not of
    # BLOCK_M/BLOCK_N, so a tile-index coordinate could not name them.
    gLm = cute.local_tile(
        cute.domain_offset((off_m, bk, b), mL_a),
        cute.slice_(MMA_TILER, (None, 0, None)), (0, 0, 0),
    )
    gLn = cute.local_tile(
        cute.domain_offset((off_n, bk, b), mL_b),
        cute.slice_(MMA_TILER, (0, None, None)), (0, 0, 0),
    )
    gA_ld = cute.local_tile(
        cute.domain_offset((off_m, off_n, b), mA_ld),
        cute.slice_(MMA_TILER, (None, None, 0)), (0, 0, 0),
    )
    gA_st = cute.local_tile(
        cute.domain_offset((off_m, off_n, b), mA_st),
        cute.slice_(MMA_TILER, (None, None, 0)), (0, 0, 0),
    )

    # The TMA source has to be partitioned by the same MMA the smem
    # destination was laid out for, so both sides agree on (MMA, MMA_M, MMA_K).
    thr_mma = tiled_mma.get_slice(0)
    tCgLm = thr_mma.partition_A(gLm)
    tCgLn = thr_mma.partition_B(gLn)
    tSLm, tGLm = cpasync.tma_partition(
        tma_atom_lm, 0, cute.make_layout(1),
        cute.group_modes(sLm, 0, 3), cute.group_modes(tCgLm, 0, 3),
    )
    tSLn, tGLn = cpasync.tma_partition(
        tma_atom_ln, 0, cute.make_layout(1),
        cute.group_modes(sLn, 0, 3), cute.group_modes(tCgLn, 0, 3),
    )
    tSA_ld, tGA_ld = cpasync.tma_partition(
        tma_atom_a_ld, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 2), cute.group_modes(gA_ld, 0, 2),
    )
    tSA_st, tGA_st = cpasync.tma_partition(
        tma_atom_a_st, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 2), cute.group_modes(gA_st, 0, 2),
    )

    l_bytes = cute.size_in_bytes(cutlass.Float32, sLm.layout)
    a_bytes = cute.size_in_bytes(cutlass.Float32, sA.layout)

    iket.mark("tile", a_phase)
    iket.range_push("tma_issue")
    cute.arch.barrier()
    # One warp issues the loads. The `expect_tx` arrivals need an explicit
    # single-thread election, but `cute.copy` on a TMA atom elects internally
    # -- putting it inside `elect_one` too would leave its own `elect.sync`
    # waiting on the 31 lanes that election had already switched off, and the
    # copy would never issue.
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar_l, 2 * l_bytes)
        cute.copy(tma_atom_lm, tGLm, tSLm, tma_bar_ptr=bar_l)
        cute.copy(tma_atom_ln, tGLn, tSLn, tma_bar_ptr=bar_l)
        # bar_a lives with the copy it guards: issued here, waited on only
        # in the epilogue, so A arrives in the background behind everything
        # above.
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar_a, a_bytes)
        cute.copy(tma_atom_a_ld, tGA_ld, tSA_ld, tma_bar_ptr=bar_a)
    iket.range_pop()

    iket.range_push("wait_L")
    cute.arch.mbarrier_wait(bar_l, 0)
    iket.range_pop()

    # tf32x3: L_m @ L_n^T ~= hi_m@hi_n + hi_m@lo_n + lo_m@hi_n recovers most of
    # fp32's precision in 3 tf32 tensor-core passes instead of 1.
    iket.range_push("split")
    _split_hi_lo(fLm, fLm_lo, tidx)
    _split_hi_lo(fLn, fLn_lo, tidx)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()
    iket.range_pop()

    tCrLm = tiled_mma.make_fragment_A(sLm)
    tCrLn = tiled_mma.make_fragment_B(sLn)
    tCrLm_lo = tiled_mma.make_fragment_A(sLm_lo)
    tCrLn_lo = tiled_mma.make_fragment_B(sLn_lo)

    # hi_m@hi_n + hi_m@lo_n + lo_m@hi_n. `hi` is the untouched source tile:
    # tcgen05 narrows it to tf32 by truncation itself, so it is never
    # materialized.
    terms = ((tCrLm, tCrLn), (tCrLm, tCrLn_lo), (tCrLm_lo, tCrLn))

    # One warp issues the MMAs, for the same reason as the loads: `cute.gemm`
    # elects a thread of whatever warp reaches it, so leaving all four warps
    # here would issue the instruction group four times over.
    num_kblocks: cutlass.Constexpr = cute.size(tCrLm, mode=[2])
    iket.range_push("mma")
    if warp_idx == 0:
        written = [False] * NUM_ACC
        acc_flag = None
        for kb in cutlass.range_constexpr(num_kblocks):
            kc = (None, None, kb)
            for t in cutlass.range_constexpr(3):
                dst: cutlass.Constexpr = TERM_ACC[t]
                # ACCUMULATE is a field of the shared TiledMma, so it is only
                # re-set when it actually has to change.
                want: cutlass.Constexpr = written[dst]
                if cutlass.const_expr(acc_flag != want):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, want)
                    acc_flag = want
                a, bb = terms[t]
                cute.gemm(tiled_mma, accs[dst], a[kc], bb[kc], accs[dst])
                written[dst] = True

        # Every MMA lands on one arrival: a single commit after the last one
        # covers the group, so the epilogue waits exactly once.
        with cute.arch.elect_one():
            tcgen05.commit(bar_l)
    cute.arch.mbarrier_wait(bar_l, 1)
    iket.range_pop()

    # The epilogue tile is the whole MMA tile, so both (EPI_M, EPI_N) modes
    # are 1 and get sliced back off.
    epi: cutlass.Constexpr = (None, None, None, 0, 0)
    tTR_acc = [thr_copy_t2r.partition_S(_epi_view(a))[epi] for a in accs]
    # Register fragments are sized from the t2r DESTINATION partition: the TMEM
    # side of a t2r copy is addressed per warp, not per thread, so its shape is
    # not the per-thread register count. sA is only used for the shape here --
    # the actual shared-memory accesses go through the r2s partition below.
    tTR_shape: cutlass.Constexpr = thr_copy_t2r.partition_D(sA).shape

    r = [cute.make_rmem_tensor(tTR_shape, cutlass.Float32)
         for _ in range(NUM_ACC)]
    rA = cute.make_rmem_tensor(tTR_shape, cutlass.Float32)
    iket.range_push("t2r_load")
    for i in cutlass.range_constexpr(NUM_ACC):
        cute.copy(tiled_copy_t2r, tTR_acc[i], r[i])
    # tcgen05.ld is asynchronous with respect to the registers it fills, so the
    # reads below (and the next tile's MMAs writing these same accumulators)
    # have to be ordered behind it explicitly.
    cute.arch.fence_view_async_tmem_load()
    # left-associative, matching the order a chained accumulator produced
    update = r[0].load()
    for i in cutlass.range_constexpr(1, NUM_ACC):
        update = update + r[i].load()
    iket.range_pop()

    # Only NOW is the A tile needed -- it has had the whole hi/lo split plus
    # all three MMAs to arrive.
    iket.range_push("wait_A")
    cute.arch.mbarrier_wait(bar_a, a_phase)
    iket.range_pop()

    # The A tile is read and written in the accumulator's own register layout,
    # so nothing repartitions between the subtraction and either memory.
    #
    # This costs 32 scalar ld.shared.b32 + 32 st.shared.b32 per warp per tile,
    # against Gluon's ~14 each, and runs at a 59-61% bank-conflict rate. That
    # extra shared-memory traffic IS the 1.69x SYRK gap -- but routing the
    # store through CUTLASS's r2s/stmatrix path to vectorize it was measured
    # and does NOT help: it cuts store requests 4.8x while leaving the
    # wavefront count identical (5.53M either way), because the conflict
    # degree rises 4.1-way -> 19.7-way to compensate. The addresses, not the
    # instruction width, are what set the wavefront count. See the leaderboard
    # file for the counters.
    iket.range_push("epilogue")
    tTR_sA = thr_copy_t2r.partition_D(sA)
    cute.autovec_copy(tTR_sA, rA)
    result = rA.load() - update

    # Write the result back through sA rather than a separate out tile: sA's
    # contents are dead the moment they reach registers, and this is a
    # read-modify-write of the same global tile, so the same 16KB serves both
    # directions.
    rA.store(result)
    cute.autovec_copy(rA, tTR_sA)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()
    iket.range_pop()

    iket.range_push("tma_store")
    if warp_idx == 0:
        cute.copy(tma_atom_a_st, tSA_st, tGA_st)
        cute.arch.cp_async_bulk_commit_group()
    # The `read` variant: this only has to guarantee that sA is free to be
    # overwritten by the next tile. Global visibility is the kernel boundary's
    # job -- a CTA cannot retire with a bulk copy still outstanding -- which is
    # why CUTLASS's own TmaStoreFence uses read=True even for its producer
    # tail. Waiting for full completion instead serializes the second tile of
    # the pair behind the first tile's store reaching memory.
    cute.arch.cp_async_bulk_wait_group(0, read=True)
    iket.range_pop()


@cute.jit
def _syrk_launcher(
    mL: cute.Tensor,
    mA: cute.Tensor,
    bk: cutlass.Int32,
    start: cutlass.Int32,
    num_m_tiles: cutlass.Int32,
    num_n_tiles: cutlass.Int32,
    batch: cutlass.Int32,
):
    # Both MMA operands are tiles of the SAME row-major L, so both are
    # K-major: A is L[off_m : +BLOCK_M, bk : +NB] and B is
    # L[off_n : +BLOCK_N, bk : +NB]. C = A @ B^T is exactly the SYRK update.
    major: cutlass.Constexpr = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        cutlass.TFloat32, cutlass.TFloat32, major, major,
        cutlass.Float32, tcgen05.CtaGroup.ONE, MMA_TILER[:2],
    )
    sLa_layout = cute.slice_(
        sm100_utils.make_smem_layout_a(tiled_mma, MMA_TILER, cutlass.TFloat32, 1),
        (None, None, None, 0),
    )
    sLb_layout = cute.slice_(
        sm100_utils.make_smem_layout_b(tiled_mma, MMA_TILER, cutlass.TFloat32, 1),
        (None, None, None, 0),
    )
    sA_layout = cute.slice_(
        sm100_utils.make_smem_layout_epi(
            cutlass.Float32, utils.LayoutEnum.ROW_MAJOR, (BLOCK_M, BLOCK_N), 1
        ),
        (None, None, 0),
    )

    # One descriptor per tensor for the whole (n, n, batch) problem: the batch
    # is the L-mode, so it costs a coordinate, not a kernel launch.
    #
    # NO internal_type=TFloat32 HERE, even though the MMA operands are tf32.
    # That option makes the TMA unit itself round fp32 -> tf32 in flight, so
    # shared memory receives values whose low 13 mantissa bits are already
    # gone -- and `lo = x - trunc_tf32(x)` then comes out identically zero,
    # silently reducing tf32x3 to plain tf32. Measured on a single [64,32] x
    # [64,32]^T product against a float64 reference (scale 23.87):
    #
    #   internal_type=TFloat32, 1 term   8.1e-03    3 terms  8.1e-03  (!)
    #   no internal_type,       1 term   1.7e-02    3 terms  1.1e-05
    #
    # The same probe settles which residual to use: with the bits intact, the
    # 1-term result matches a truncated-tf32 reference to 3.6e-06 and an
    # RN-tf32 one only to 2.0e-02, so tcgen05 narrows by TRUNCATION and `lo`
    # must be the residual against truncation -- pairing it with a
    # round-to-nearest `hi` measures 2.0e-02, worse than not correcting at all.
    atom_lm, tLm = nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(), mL, sLa_layout, MMA_TILER, tiled_mma,
    )
    atom_ln, tLn = nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(), mL, sLb_layout, MMA_TILER, tiled_mma,
    )
    a_cta_v = cute.composition(
        cute.make_identity_layout(mA.shape), (BLOCK_M, BLOCK_N)
    )
    atom_a_ld, tA_ld = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mA, sA_layout, a_cta_v
    )
    atom_a_st, tA_st = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mA, sA_layout, a_cta_v
    )

    acc_fake = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C(MMA_TILER[:2]))
    num_tmem_cols: cutlass.Constexpr = utils.get_num_tmem_alloc_cols(
        [acc_fake] * NUM_ACC
    )

    _syrk_kernel_tcgen05(
        atom_lm, tLm, atom_ln, tLn, atom_a_ld, tA_ld, atom_a_st, tA_st,
        sLa_layout, sLb_layout, sA_layout, tiled_mma,
        bk, start, num_n_tiles, num_tmem_cols,
    ).launch(
        grid=(cute.ceil_div(num_n_tiles, 2), num_m_tiles, batch),
        block=(SYRK_THREADS, 1, 1),
        smem=SYRK_SMEM_BYTES,
    )


# ---------------------------------------------------------------------------
# Host driver
# ---------------------------------------------------------------------------
_compiled = {}


def _get_compiled(tA: cute.Tensor, tL: cute.Tensor):
    """Compile once for the whole family of shapes. Every extent and stride is
    marked dynamic and bk/start/grid are runtime scalars, so a single pair of
    compiled kernels serves every (batch, n) and every bk step."""
    key = "syrk_blocked_f32"
    if key not in _compiled:
        _compiled[key] = (
            cute.compile(
                _panel_launcher, tA, tL,
                cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0),
            ),
            cute.compile(
                _syrk_launcher, tL, tA,
                cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0),
                cutlass.Int32(0), cutlass.Int32(0),
            ),
        )
    return _compiled[key]


def custom_kernel(data: input_t) -> output_t:
    A = data.to(torch.float32).contiguous().clone()
    batch, n, _ = A.shape
    L = torch.zeros_like(A)

    # (batch, n, n) -> (n, n, batch): rows and columns stay the leading two
    # modes so every tile is 2D and contiguous in the column direction, and
    # the batch becomes the TMA L-mode.
    tA = from_dlpack(A.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    tL = from_dlpack(L.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    panel, syrk = _get_compiled(tA, tL)

    # Every launch goes to CUDA's default execution queue, which is also where
    # the surrounding torch work runs, so the panel/SYRK alternation below is
    # ordered by the queue itself and needs no explicit handle.
    for bk in range(0, n, NB):
        panel(tA, tL, bk, n, batch)

        # Only a full NB-wide panel leaves a trailing submatrix: a short tail
        # panel (n - bk < NB) is the last block-column, so trailing == 0.
        trailing = n - bk - NB
        if trailing > 0:
            start = bk + NB
            num_m_tiles = (trailing + BLOCK_M - 1) // BLOCK_M
            num_n_tiles = (trailing + BLOCK_N - 1) // BLOCK_N
            syrk(tL, tA, bk, start, num_m_tiles, num_n_tiles, batch)

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
