import time

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, tensor_memory_descriptor, allocate_tensor_memory, get_tmem_reg_layout,
    tma, mbarrier, tcgen05_mma, tcgen05_commit, fence_async_shared,
)

from task import input_t, output_t

# Warp-specialized version of the persistent, batch-flattened tcgen05 SYRK
# kernel (cholesky_gluon_tcgen05_persistent.py), following the real
# load/mma/epilogue partition pattern from Gluon's own
# 08-warp-specialization.py tutorial (matmul_warp_specialized_kernel),
# adapted for the tf32x3 hi/lo split and its three independent accumulators.
#
# Why this is a genuinely different change from plain persistence: the
# earlier persistent-but-not-specialized kernel was measured to be a net
# REGRESSION (worse at every size, growing to +123% at n=32768) because it
# traded away the non-persistent version's "free" hardware-scheduled overlap
# (many small blocks, several resident per SM) for nothing -- one block per
# SM, doing load-wait-compute-wait-store in lockstep, with no compensating
# overlap of its own. Warp specialization is what's supposed to earn that
# trade back: dedicated load/mma/epilogue partitions run concurrently within
# each persistent block, so while the mma partition works on tile i, the
# load partition is already fetching tile i+k, and the epilogue partition is
# draining tile i-1 -- overlap from *within* one block instead of *between*
# many.
#
# Structure (see 08-warp-specialization.py's matmul_load_partition /
# matmul_mma_partition / matmul_epilogue_partition for the template this
# follows):
#   - Tiles are enumerated densely as (b, pid_m, pid_n) over the WHOLE
#     batch*num_m_tiles*num_n_tiles space -- no lo/hi mirroring trick here
#     (that was a grid-size-halving trick for one-block-per-tile launches;
#     it doesn't help a software tile loop). Each of the three partitions
#     independently computes the same (b, pid_m, pid_n) sequence from the
#     same idx range and skips upper-triangular tiles (pid_n > pid_m) with
#     the same branch, so they stay in lockstep without explicit
#     coordination about what's skipped.
#   - l_m/l_n/a use a 2-stage ring buffer. l_m/l_n are freed by the mma
#     partition once split into tf32 hi/lo; `a` isn't needed until the
#     epilogue partition subtracts from it, much later in the pipeline -- so
#     each stage's "empty" barrier has count=2 (both the mma partition,
#     after consuming l_m/l_n, and the epilogue partition, after consuming
#     `a`, must arrive before the load partition can reuse that stage).
#   - The accumulator (3 independent TMEM buffers per tile: hi*hi, hi*lo,
#     lo*hi) is double-buffered across 2 stages, matching the tutorial's
#     accumulator pattern. The tf32 hi/lo scratch buffers ride the same
#     stage index as the accumulator: by the time the epilogue partition
#     signals a given accumulator stage "empty" (having read the result
#     out), the mma that produced it -- and therefore its reads of that
#     stage's hi/lo scratch -- is necessarily long complete, so reusing the
#     scratch buffers is safe without any separate barrier for them.


@aggregate
class Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, index, phase, num_barriers):
        self.index = index
        self.phase = phase
        self.num_barriers = gl.constexpr(num_barriers)

    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def next(self, pred=True):
        incr = self.index + gl.where(pred, 1, 0)
        rollover = incr == self.num_barriers
        index = gl.where(rollover, 0, incr)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return Counter(index, phase, self.num_barriers)


@aggregate
class PartitionArgs:
    # Deliberately excludes A_ptr/L_ptr/bk/start/stride_*/n/batch: Triton's
    # `equal_to_1` argument specialization folds any scalar kernel argument
    # whose runtime value is exactly 1 (e.g. stride_c=1 for a contiguous last
    # dim, or batch=1) into a compile-time Python int rather than a
    # gl.tensor. This aggregate's __init__ strictly type-checks fields
    # against their annotation, so a strict `gl.tensor` annotation breaks the
    # batch=1/stride_c=1 case, while forcing everything through
    # gl.to_tensor() breaks tma.make_tensor_descriptor's own static
    # `last_stride != 1` check downstream. Simplest fix: keep these scalars
    # as plain function parameters (passed to each partition directly,
    # alongside `p`), where ordinary duck-typed call semantics apply and no
    # such check exists.
    l_m_bufs: gl.shared_memory_descriptor
    l_n_bufs: gl.shared_memory_descriptor
    a_bufs: gl.shared_memory_descriptor
    l_m_hi_bufs: gl.shared_memory_descriptor
    l_m_lo_bufs: gl.shared_memory_descriptor
    l_n_hi_bufs: gl.shared_memory_descriptor
    l_n_lo_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_hh_bufs: tensor_memory_descriptor
    acc_hl_bufs: tensor_memory_descriptor
    acc_lh_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    NB: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_M_TILES: gl.constexpr
    NUM_N_TILES: gl.constexpr
    num_warps: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, l_m_bufs, l_n_bufs, a_bufs, l_m_hi_bufs, l_m_lo_bufs, l_n_hi_bufs, l_n_lo_bufs,
                 load_empty_bars, load_ready_bars,
                 acc_hh_bufs, acc_hl_bufs, acc_lh_bufs, acc_empty_bars, acc_ready_bars,
                 NB, BLOCK_M, BLOCK_N, NUM_M_TILES, NUM_N_TILES, num_warps):
        self.l_m_bufs = l_m_bufs
        self.l_n_bufs = l_n_bufs
        self.a_bufs = a_bufs
        self.l_m_hi_bufs = l_m_hi_bufs
        self.l_m_lo_bufs = l_m_lo_bufs
        self.l_n_hi_bufs = l_n_hi_bufs
        self.l_n_lo_bufs = l_n_lo_bufs
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.acc_hh_bufs = acc_hh_bufs
        self.acc_hl_bufs = acc_hl_bufs
        self.acc_lh_bufs = acc_lh_bufs
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.NB = gl.constexpr(NB)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.NUM_M_TILES = gl.constexpr(NUM_M_TILES)
        self.NUM_N_TILES = gl.constexpr(NUM_N_TILES)
        self.num_warps = gl.constexpr(num_warps)


@gluon.jit
def _round_tf32(x):
    bits = x.to(gl.uint32, bitcast=True)
    rounded = (bits + 0x1000) & 0xFFFFE000
    return rounded.to(gl.float32, bitcast=True)


@gluon.jit
def _decode_mn_tile(idx, NUM_N_TILES: gl.constexpr):
    # `b` is deliberately NOT decoded here -- it's a real grid dimension
    # (program_id(1)), fixed for a persistent CTA's whole lifetime. Earlier
    # revisions folded batch into this same flattened idx (decoding b via
    # idx // TILES_PER_BATCH), which let a single persistent CTA's tile loop
    # cross from one batch element to the next mid-loop. That crossing was
    # empirically confirmed (via a debug run forcing 1 tile/CTA, which fixed
    # it) to corrupt results at ~1-5% max error whenever a CTA's loop spanned
    # more than one `b` -- root cause not fully isolated, but the aggregate
    # arg-passing/scratch-descriptor machinery evidently doesn't tolerate a
    # per-thread runtime pointer offset changing mid-loop the way it needs
    # to. Keeping `b` a fixed per-CTA grid dimension sidesteps the whole
    # class of bug by construction.
    pid_m = idx // NUM_N_TILES
    pid_n = idx % NUM_N_TILES
    return pid_m, pid_n


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
def _syrk_load_partition(A_ptr, L_ptr, bk, start, stride_b, stride_r, stride_c, n, b, p, total_tiles):
    l_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([p.BLOCK_M, p.NB], gl.float32)
    a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([p.BLOCK_M, p.BLOCK_N], gl.float32)

    pid = gl.program_id(0)
    num_programs = gl.num_programs(0)
    state = Counter.create(1, p.load_empty_bars.shape[0])

    for idx in range(pid, total_tiles, num_programs):
        pid_m, pid_n = _decode_mn_tile(idx, p.NUM_N_TILES)
        if pid_n <= pid_m:
            off_m = start + pid_m * p.BLOCK_M
            off_n = start + pid_n * p.BLOCK_N

            l_desc = tma.make_tensor_descriptor(
                L_ptr + b * stride_b, [n, n], [stride_r, stride_c], [p.BLOCK_M, p.NB], l_layout,
            )
            a_desc = tma.make_tensor_descriptor(
                A_ptr + b * stride_b, [n, n], [stride_r, stride_c], [p.BLOCK_M, p.BLOCK_N], a_layout,
            )

            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
            bar = p.load_ready_bars.index(state.index)
            mbarrier.expect(bar, 2 * l_desc.block_type.nbytes + a_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(l_desc, [off_m, bk], bar, p.l_m_bufs.index(state.index))
            tma.async_copy_global_to_shared(l_desc, [off_n, bk], bar, p.l_n_bufs.index(state.index))
            tma.async_copy_global_to_shared(a_desc, [off_m, off_n], bar, p.a_bufs.index(state.index))
            state = state.next()


@gluon.jit
def _syrk_mma_partition(p, total_tiles):
    load_state = Counter.create(0, p.load_ready_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])

    REG_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 16], [16, 2], [p.num_warps, 1], [1, 0])

    for idx in range(gl.program_id(0), total_tiles, gl.num_programs(0)):
        pid_m, pid_n = _decode_mn_tile(idx, p.NUM_N_TILES)
        if pid_n <= pid_m:
            mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)

            l_m_smem = p.l_m_bufs.index(load_state.index)
            l_n_smem = p.l_n_bufs.index(load_state.index)
            l_m_reg = l_m_smem.load(REG_LAYOUT)
            l_m_hi = _round_tf32(l_m_reg)
            l_m_lo = _round_tf32(l_m_reg - l_m_hi)
            l_n_reg = l_n_smem.load(REG_LAYOUT)
            l_n_hi = _round_tf32(l_n_reg)
            l_n_lo = _round_tf32(l_n_reg - l_n_hi)

            mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
            l_m_hi_smem = p.l_m_hi_bufs.index(acc_state.index)
            l_m_lo_smem = p.l_m_lo_bufs.index(acc_state.index)
            l_n_hi_smem = p.l_n_hi_bufs.index(acc_state.index)
            l_n_lo_smem = p.l_n_lo_bufs.index(acc_state.index)
            l_m_hi_smem.store(l_m_hi)
            l_m_lo_smem.store(l_m_lo)
            l_n_hi_smem.store(l_n_hi)
            l_n_lo_smem.store(l_n_lo)
            fence_async_shared()

            # Loads consumed; signal load_empty (count=2 -- epilogue also
            # arrives here once it's done with `a`, see _syrk_epilogue_partition).
            mbarrier.arrive(p.load_empty_bars.index(load_state.index), count=1)
            load_state = load_state.next()

            acc_hh = p.acc_hh_bufs.index(acc_state.index)
            acc_hl = p.acc_hl_bufs.index(acc_state.index)
            acc_lh = p.acc_lh_bufs.index(acc_state.index)
            tcgen05_mma(l_m_hi_smem, l_n_hi_smem.permute((1, 0)), acc_hh, use_acc=False)
            tcgen05_mma(l_m_hi_smem, l_n_lo_smem.permute((1, 0)), acc_hl, use_acc=False)
            tcgen05_mma(l_m_lo_smem, l_n_hi_smem.permute((1, 0)), acc_lh, use_acc=False)
            tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
            acc_state = acc_state.next()


@gluon.jit
def _syrk_epilogue_partition(A_ptr, stride_b, stride_r, stride_c, n, start, b, p, total_tiles):
    acc_state = Counter.create(0, p.acc_ready_bars.shape[0])
    # `a_bufs` is sized/indexed by the LOAD partition's stage count
    # (NUM_LOAD_STAGES), not the accumulator's (NUM_ACC_STAGES) -- these two
    # only happen to be numerically interchangeable when the stage counts
    # are equal. Track this partition's own load-stage index, advancing in
    # the same skip-gated lockstep as mma's `load_state` and this
    # partition's own `acc_state`, so `a_bufs`/`load_empty_bars` accesses
    # land on the correct physical stage regardless of NUM_LOAD_STAGES vs
    # NUM_ACC_STAGES.
    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    tmem_layout: gl.constexpr = TensorMemoryLayout([p.BLOCK_M, p.BLOCK_N], col_stride=1)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (p.BLOCK_M, p.BLOCK_N), tmem_layout, p.num_warps)
    a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([p.BLOCK_M, p.BLOCK_N], gl.float32)

    for idx in range(gl.program_id(0), total_tiles, gl.num_programs(0)):
        pid_m, pid_n = _decode_mn_tile(idx, p.NUM_N_TILES)
        if pid_n <= pid_m:
            off_m = start + pid_m * p.BLOCK_M
            off_n = start + pid_n * p.BLOCK_N

            mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
            update = (p.acc_hh_bufs.index(acc_state.index).load(acc_reg_layout)
                      + p.acc_hl_bufs.index(acc_state.index).load(acc_reg_layout)
                      + p.acc_lh_bufs.index(acc_state.index).load(acc_reg_layout))

            a_reg = p.a_bufs.index(load_state.index).load(acc_reg_layout)
            result = a_reg - update

            # `a` consumed; signal load_empty (count=2, paired with the mma
            # partition's arrival for the same stage's l_m/l_n).
            mbarrier.arrive(p.load_empty_bars.index(load_state.index), count=1)
            load_state = load_state.next()

            out_smem = gl.allocate_shared_memory(gl.float32, [p.BLOCK_M, p.BLOCK_N], a_layout)
            out_smem.store(result)
            fence_async_shared()
            a_desc = tma.make_tensor_descriptor(
                A_ptr + b * stride_b, [n, n], [stride_r, stride_c], [p.BLOCK_M, p.BLOCK_N], a_layout,
            )
            tma.async_copy_shared_to_global(a_desc, [off_m, off_n], out_smem)
            tma.store_wait(pendings=0)

            mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
            acc_state = acc_state.next()


@gluon.jit
def _syrk_kernel_tcgen05_warpspec(A_ptr, L_ptr, bk, start,
                                  stride_b, stride_r, stride_c, n,
                                  NB: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                  NUM_M_TILES: gl.constexpr, NUM_N_TILES: gl.constexpr,
                                  NUM_LOAD_STAGES: gl.constexpr, NUM_ACC_STAGES: gl.constexpr,
                                  num_warps: gl.constexpr):
    # `b` is grid axis 1, fixed for this persistent CTA's whole lifetime --
    # see _decode_mn_tile's docstring for why batch must be a real grid
    # dimension rather than folded into the flattened persistent tile index.
    b = gl.program_id(1)
    l_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, NB], gl.float32)
    a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float32)

    l_m_bufs = gl.allocate_shared_memory(gl.float32, [NUM_LOAD_STAGES, BLOCK_M, NB], l_layout)
    l_n_bufs = gl.allocate_shared_memory(gl.float32, [NUM_LOAD_STAGES, BLOCK_M, NB], l_layout)
    a_bufs = gl.allocate_shared_memory(gl.float32, [NUM_LOAD_STAGES, BLOCK_M, BLOCK_N], a_layout)
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_LOAD_STAGES, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_LOAD_STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(NUM_LOAD_STAGES):
        # count=2: freed once both the mma partition (l_m/l_n) and the
        # epilogue partition (a) have finished with this stage.
        mbarrier.init(load_empty_bars.index(i), count=2)
        mbarrier.init(load_ready_bars.index(i), count=1)

    # l_*_hi_bufs/l_*_lo_bufs hold the tf32 hi/lo split per acc-stage -- see
    # module docstring for why these safely ride the accumulator's stage
    # index instead of needing their own barrier.
    l_m_hi_bufs = gl.allocate_shared_memory(gl.float32, [NUM_ACC_STAGES, BLOCK_M, NB], l_layout)
    l_m_lo_bufs = gl.allocate_shared_memory(gl.float32, [NUM_ACC_STAGES, BLOCK_M, NB], l_layout)
    l_n_hi_bufs = gl.allocate_shared_memory(gl.float32, [NUM_ACC_STAGES, BLOCK_M, NB], l_layout)
    l_n_lo_bufs = gl.allocate_shared_memory(gl.float32, [NUM_ACC_STAGES, BLOCK_M, NB], l_layout)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_hh_bufs = allocate_tensor_memory(gl.float32, [NUM_ACC_STAGES, BLOCK_M, BLOCK_N], tmem_layout)
    acc_hl_bufs = allocate_tensor_memory(gl.float32, [NUM_ACC_STAGES, BLOCK_M, BLOCK_N], tmem_layout)
    acc_lh_bufs = allocate_tensor_memory(gl.float32, [NUM_ACC_STAGES, BLOCK_M, BLOCK_N], tmem_layout)
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_ACC_STAGES, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_ACC_STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(NUM_ACC_STAGES):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    p = PartitionArgs(l_m_bufs, l_n_bufs, a_bufs, l_m_hi_bufs, l_m_lo_bufs, l_n_hi_bufs, l_n_lo_bufs,
                      load_empty_bars, load_ready_bars,
                      acc_hh_bufs, acc_hl_bufs, acc_lh_bufs, acc_empty_bars, acc_ready_bars,
                      NB, BLOCK_M, BLOCK_N, NUM_M_TILES, NUM_N_TILES, num_warps)

    total_tiles = NUM_M_TILES * NUM_N_TILES

    # gl.warp_specialize's FIRST entry is the "default" partition -- it
    # always runs with the kernel's own num_warps. The rest are "workers"
    # with an explicit warp count from worker_num_warps. Both the mma
    # partition (tf32 hi/lo split of l_m/l_n, a genuine per-thread register
    # tensor over [BLOCK_M, NB] via REG_LAYOUT) and the epilogue partition
    # (TMEM accumulator register read over [BLOCK_M, BLOCK_N] via
    # acc_reg_layout) do real register-tile work sized for `num_warps`
    # warps, so both need the full warp count -- unlike the tutorial's
    # matmul, where mma is issue-only (no register tensors) and can run on a
    # single warp. Only the load partition (pure TMA issue, all scalar) is
    # cheap enough for 1 warp.
    gl.warp_specialize([
        (_syrk_mma_partition, (p, total_tiles)),
        (_syrk_epilogue_partition, (A_ptr, stride_b, stride_r, stride_c, n, start, b, p, total_tiles)),
        (_syrk_load_partition, (A_ptr, L_ptr, bk, start, stride_b, stride_r, stride_c, n, b, p, total_tiles)),
    ], [num_warps, 1], [128, 24])


def _alloc_fn(size, alignment, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


_allocator_set = False


def custom_kernel(data: input_t) -> output_t:
    global _allocator_set
    if not _allocator_set:
        triton.set_allocator(_alloc_fn)
        _allocator_set = True

    A = data.to(torch.float32).contiguous().clone()
    batch, n, _ = A.shape
    L = torch.zeros_like(A)

    stride_b, stride_r, stride_c = L.stride()
    NB = 32
    BLOCK_I = 32
    BLOCK_M = BLOCK_N = 64
    NUM_LOAD_STAGES = 2
    NUM_ACC_STAGES = 2
    NUM_WARPS = 4

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
            tiles_per_batch = num_m_tiles * num_n_tiles
            # `b` is a real grid dimension (axis 1) -- each persistent CTA is
            # bound to exactly one batch element for its whole lifetime and
            # only loops over that batch's (pid_m, pid_n) tile space. See
            # _decode_mn_tile's docstring: folding batch into the flattened
            # persistent tile index let a single CTA's loop cross from one
            # batch element to the next mid-loop, which corrupted results.
            num_persistent = min(num_sms, tiles_per_batch)
            start = bk + nb_valid
            _syrk_kernel_tcgen05_warpspec[(num_persistent, batch)](
                A, L, bk, start, stride_b, stride_r, stride_c, n,
                NB=NB, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                NUM_M_TILES=num_m_tiles, NUM_N_TILES=num_n_tiles,
                NUM_LOAD_STAGES=NUM_LOAD_STAGES, NUM_ACC_STAGES=NUM_ACC_STAGES,
                num_warps=NUM_WARPS, maxnreg=128,
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
