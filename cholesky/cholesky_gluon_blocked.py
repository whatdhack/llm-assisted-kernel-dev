import time

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from task import input_t, output_t

# Gluon has no drop-in replacement for `tl.dot(..., input_precision="tf32x3")`:
# real tensor-core MMA (wgmma/tcgen05) needs shared-memory-resident operands,
# mbarriers, and warp-group (128-thread) granularity, which doesn't fit these
# small tiles without restructuring the whole panel/syrk split. This port
# instead uses `gl.dot_fma`, a CUDA-core FMA dot that works directly on
# register-resident BlockedLayout tiles -- same algorithm shape as the Triton
# kernel, but the SYRK step no longer runs at tensor-core throughput.
#
# The panel/TRSM kernel below keeps NB = BLOCK_I = 32 and num_warps=1: NB is
# tied to the warp size by TILE_LAYOUT's threads_per_warp=[32, 1] (one lane
# per row, all NB columns as contiguous per-lane registers), and the
# redundant per-block panel refactorization trick it relies on is only cheap
# because the panel is tiny -- see _panel_trsm_kernel's docstring-style
# comments below. The SYRK step, in contrast, uses a bigger BLOCK_M/BLOCK_N
# with more warps (see the comment above _syrk_tile) since it's plain
# CUDA-core FMA throughput there, not a serial recurrence, and num_warps=1
# leaves most of an SM idle during it.


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

    # Triton's untyped `idx = tl.arange(0, NB)` plays both a "row" and a
    # "column" role interchangeably via broadcasting. Gluon tensors carry a
    # concrete layout, so the same [0, NB) range needs two materializations,
    # one per axis it gets compared/combined against.
    idx_row = gl.arange(0, NB, layout=ROW_LAYOUT)
    idx_col = gl.arange(0, NB, layout=COL_LAYOUT)
    idx_row_valid = idx_row < nb_valid
    row_idx = idx_row[:, None]
    col_idx = idx_col[None, :]

    # --- Step 1: redundantly factor the nb_valid x nb_valid panel locally. ---
    Lp = gl.zeros((NB, NB), dtype=gl.float32, layout=TILE_LAYOUT)
    for jp in range(nb_valid):
        row_jp = gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)  # (NB,) COL_LAYOUT: Lp[jp, :]
        contrib = gl.sum(Lp * row_jp[None, :], axis=1)             # (NB,) ROW_LAYOUT

        a_col = gl.load(
            A_ptr + b * stride_b + (bk + idx_row) * stride_r + (bk + jp) * stride_c,
            mask=idx_row_valid, other=0.0,
        )
        diff = a_col - contrib
        ljj = gl.sqrt(gl.sum(gl.where(idx_row == jp, diff, 0.0), axis=0))
        new_col = gl.where(idx_row == jp, ljj, gl.where(idx_row > jp, diff / ljj, 0.0))
        Lp = gl.where(col_idx == jp, new_col[:, None], Lp)

    # --- Step 2: forward-substitute this program's BLOCK_I rows. ---
    i_idx = gl.arange(0, BLOCK_I, layout=ROW_LAYOUT)
    i = bk + pid_i * BLOCK_I + i_idx
    row_mask = i < n

    L_rows = gl.zeros((BLOCK_I, NB), dtype=gl.float32, layout=TILE_LAYOUT)
    for jp in range(nb_valid):
        row_jp = gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)  # (NB,) COL_LAYOUT, same formula as Step 1
        contrib = gl.sum(L_rows * row_jp[None, :], axis=1)         # (BLOCK_I,) ROW_LAYOUT

        a_val = gl.load(
            A_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
            mask=row_mask, other=0.0,
        )
        diff = a_val - contrib
        # row_jp is COL_LAYOUT (it came from an axis=0 reduce), so the
        # matching index range here is idx_col, not idx_row.
        ljj = gl.sum(gl.where(idx_col == jp, row_jp, 0.0), axis=0)

        is_diag = i == (bk + jp)
        val = gl.where(is_diag, ljj, diff / ljj)
        L_rows = gl.where(col_idx == jp, val[:, None], L_rows)

        gl.store(
            L_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
            val, mask=row_mask,
        )


# The SYRK step below uses BLOCK_M = BLOCK_N = 128 (vs. NB = BLOCK_I = 32 for
# the panel kernel above) with num_warps > 1, unlike the uniform 32-everywhere
# design the panel kernel keeps. Three distinct layouts are needed instead of
# one shared TILE_LAYOUT, because the three tensors involved no longer share a
# single physical shape once BLOCK_M/N != NB:
#
#   MMA_LAYOUT   (BLOCK_M, BLOCK_N) -- the accumulator's own BlockedLayout.
#                One lane per M-row, spread across `num_warps` warps along M;
#                all BLOCK_N columns live in each lane's own registers. This
#                doubles as dot_fma's required mma_layout (the `parent` both
#                DotOperandLayouts wrap).
#   A_LOAD_LAYOUT (BLOCK_M, NB)     -- l_m's load layout. Deliberately mirrors
#                MMA_LAYOUT's M-direction mapping exactly (same
#                threads_per_warp/warps_per_cta/order, only the per-thread
#                K-width differs), so converting it into
#                DotOperandLayout(0, MMA_LAYOUT, ...) is just a relabeling,
#                not real data movement.
#   B_LOAD_LAYOUT (NB, BLOCK_N)     -- l_n_t's load layout. K spread across
#                lanes, N spread across warps -- a different thread/warp
#                mapping than MMA_LAYOUT's own N-direction (which has no
#                warp/lane spread along N at all), so converting this one
#                into DotOperandLayout(1, MMA_LAYOUT, ...) is a real,
#                non-free layout conversion. Correctness doesn't depend on
#                the conversion being free (gl.convert_layout inserts
#                whatever data movement is needed either way), only
#                performance does.
#
# rm/rn each need two materializations for the same reason idx_row/idx_col
# did in the panel kernel: the "M row" and "N column" index ranges are used
# against tensors in two different layouts (the load layout and MMA_LAYOUT).


@gluon.jit
def _syrk_tile(A_ptr, L_ptr, l_m, rm_mma, mask_m_mma, b, pid_n, start,
               stride_b, stride_r, stride_c, bk, n,
               NB: gl.constexpr, nb_valid: gl.constexpr, BLOCK_N: gl.constexpr,
               MMA_LAYOUT: gl.constexpr, N_COL: gl.constexpr,
               B_ROW: gl.constexpr, B_COL: gl.constexpr,
               A_OPERAND: gl.constexpr, B_OPERAND: gl.constexpr):
    rn_mma = start + pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=N_COL)
    rn_b = start + pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=B_COL)
    mask_n_mma = rn_mma < n
    mask_n_b = rn_b < n
    rk = gl.arange(0, NB, layout=B_ROW)
    k_mask = rk < nb_valid

    # Load L[rn, bk:bk+nb_valid] already transposed to (K, N) -- this is
    # dot_fma's required "b" operand shape, so it sidesteps the separate
    # tl.trans() the Triton kernel needed.
    l_n_t = gl.load(
        L_ptr + b * stride_b + rn_b[None, :] * stride_r + (bk + rk)[:, None] * stride_c,
        mask=k_mask[:, None] & mask_n_b[None, :], other=0.0,
    )

    a_ptrs = A_ptr + b * stride_b + rm_mma[:, None] * stride_r + rn_mma[None, :] * stride_c
    a_mask = mask_m_mma[:, None] & mask_n_mma[None, :]
    a_old = gl.load(a_ptrs, mask=a_mask, other=0.0)

    a_op = gl.convert_layout(l_m, A_OPERAND)
    b_op = gl.convert_layout(-l_n_t, B_OPERAND)
    # dot_fma computes acc + a @ b in a single CUDA-core FMA sweep; folding
    # the negation into b gives a_old - l_m @ l_n_t directly, instead of a
    # separate zero-initialized dot plus a subtract.
    a_new = gl.dot_fma(a_op, b_op, a_old)

    gl.store(a_ptrs, a_new, mask=a_mask)


@gluon.jit
def _syrk_kernel(
    A_ptr, L_ptr,
    bk,
    stride_b, stride_r, stride_c,
    n,
    NB: gl.constexpr,
    nb_valid: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_N_TILES: gl.constexpr,
    num_warps: gl.constexpr,
):
    MMA_LAYOUT: gl.constexpr = gl.BlockedLayout([1, BLOCK_N], [32, 1], [num_warps, 1], [1, 0])
    M_ROW: gl.constexpr = gl.SliceLayout(dim=1, parent=MMA_LAYOUT)
    N_COL: gl.constexpr = gl.SliceLayout(dim=0, parent=MMA_LAYOUT)

    A_LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout([1, NB], [32, 1], [num_warps, 1], [1, 0])
    A_ROW: gl.constexpr = gl.SliceLayout(dim=1, parent=A_LOAD_LAYOUT)
    A_COL: gl.constexpr = gl.SliceLayout(dim=0, parent=A_LOAD_LAYOUT)

    B_LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout([1, BLOCK_N // num_warps], [32, 1], [1, num_warps], [1, 0])
    B_ROW: gl.constexpr = gl.SliceLayout(dim=1, parent=B_LOAD_LAYOUT)
    B_COL: gl.constexpr = gl.SliceLayout(dim=0, parent=B_LOAD_LAYOUT)

    A_OPERAND: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA_LAYOUT, k_width=0)
    B_OPERAND: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA_LAYOUT, k_width=0)

    b = gl.program_id(0)
    pid_m = gl.program_id(1)
    pid_n_lo = gl.program_id(2)
    pid_n_hi = NUM_N_TILES - 1 - pid_n_lo

    start = bk + nb_valid

    rm_a = start + pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=A_ROW)
    mask_m_a = rm_a < n
    rk_a = gl.arange(0, NB, layout=A_COL)
    k_mask_a = rk_a < nb_valid
    l_m = gl.load(
        L_ptr + b * stride_b + rm_a[:, None] * stride_r + (bk + rk_a)[None, :] * stride_c,
        mask=mask_m_a[:, None] & k_mask_a[None, :], other=0.0,
    )

    rm_mma = start + pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=M_ROW)
    mask_m_mma = rm_mma < n

    if pid_n_lo <= pid_m:
        _syrk_tile(A_ptr, L_ptr, l_m, rm_mma, mask_m_mma, b, pid_n_lo, start,
                   stride_b, stride_r, stride_c, bk, n,
                   NB=NB, nb_valid=nb_valid, BLOCK_N=BLOCK_N,
                   MMA_LAYOUT=MMA_LAYOUT, N_COL=N_COL, B_ROW=B_ROW, B_COL=B_COL,
                   A_OPERAND=A_OPERAND, B_OPERAND=B_OPERAND)
    if pid_n_hi <= pid_m and pid_n_hi != pid_n_lo:
        _syrk_tile(A_ptr, L_ptr, l_m, rm_mma, mask_m_mma, b, pid_n_hi, start,
                   stride_b, stride_r, stride_c, bk, n,
                   NB=NB, nb_valid=nb_valid, BLOCK_N=BLOCK_N,
                   MMA_LAYOUT=MMA_LAYOUT, N_COL=N_COL, B_ROW=B_ROW, B_COL=B_COL,
                   A_OPERAND=A_OPERAND, B_OPERAND=B_OPERAND)


def custom_kernel(data: input_t) -> output_t:
    A = data.to(torch.float32).contiguous().clone()
    batch, n, _ = A.shape
    L = torch.zeros_like(A)

    stride_b, stride_r, stride_c = L.stride()
    # NB is tied to the warp size (32) by the panel kernel's TILE_LAYOUT
    # (threads_per_warp=[32, 1]); BLOCK_I must match NB for the same reason
    # as before. BLOCK_M/BLOCK_N (the SYRK trailing-update tile) are a
    # separate, larger knob -- see the comment above _syrk_tile.
    NB = 32
    BLOCK_I = 32
    BLOCK_M = BLOCK_N = 64 #32
    SYRK_NUM_WARPS = 2 #4

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
            num_m_tiles = triton.cdiv(trailing, BLOCK_M)
            num_n_tiles = triton.cdiv(trailing, BLOCK_N)
            grid_syrk = (batch, num_m_tiles, triton.cdiv(num_n_tiles, 2))
            _syrk_kernel[grid_syrk](
                A, L, bk, stride_b, stride_r, stride_c, n,
                NB=NB, nb_valid=nb_valid, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                NUM_N_TILES=num_n_tiles, num_warps=SYRK_NUM_WARPS,
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
