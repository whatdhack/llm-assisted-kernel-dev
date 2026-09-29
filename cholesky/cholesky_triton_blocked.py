"""
Blocked right-looking Cholesky: panel-factor + TRSM fused into one kernel,
SYRK (the trailing-submatrix update) as a separate tiled-matmul kernel.

Unlike cholesky_triton.py (one launch per *column*, O(n) launches total),
this does one panel+TRSM launch and one SYRK launch per *block-column* of
width up to NB, i.e. O(n/NB) launches. The panel+TRSM step still does a
sequential recurrence across the block's columns, but it's unrolled
*inside* a single kernel (a compile-time Python `for` loop) instead of
being separate host-side launches. The SYRK step is a genuine dense
matmul (tl.dot), which is what actually closes the gap with cuSOLVER —
it's where the O(n^3) bulk of the FLOPs end up, running at GEMM speed
instead of as many small per-row dot products.

NB is the fixed, power-of-2 buffer width used for tl.arange (Triton
requires arange sizes to be powers of 2). nb_valid is the actual number
of columns in *this* block-column's panel — equal to NB except for a
possible smaller tail block when n is not a multiple of NB. Anything
with local index >= nb_valid is masked off as invalid.
================================================================================
ALGORITHM: Blocked Right-Looking Cholesky Factorization (A = L * L^T)
================================================================================
INPUT:  An n x n real symmetric positive definite matrix 'A'
        A block size parameter 'b' 
OUTPUT: The lower triangular matrix 'L' overwriting the lower triangle of 'A'
================================================================================

--------------------------------------------------------------------------------
1. MATRIX PARTITIONING SCHEME & DERIVATION
--------------------------------------------------------------------------------
At any given step of the algorithm, the active trailing portion of the matrix 
is conceptually partitioned into a 2x2 grid of blocks based on the current loop 
index 'i' and a chosen block width 'b':

    A = [ A11   A21^T ]
        [ A21   A22   ]

    A11: The active diagonal block of size b x b.
    A21: The vertical panel block sitting directly below A11, 
         of size (n - i - b + 1) x b.
    A22: The remaining un-factored trailing submatrix sitting to the right 
         and below A21, of size (n - i - b + 1) x (n - i - b + 1).

By setting A = L * L^T, we equate the partitioned blocks to derive the formulas:

    [ A11   A21^T ]   [ L11    0  ]   [ L11^T   L21^T ]
    [             ] = [           ] * [               ]
    [ A21   A22   ]   [ L21   L22 ]   [   0     L22^T ]

                      [ L11*L11^T       L11*L21^T       ]
                    = [                                 ]
                      [ L21*L11^T   L21*L21^T + L22*L22^T ]

--------------------------------------------------------------------------------
2. MATHEMATICAL STEPS
--------------------------------------------------------------------------------
By matching the blocks on both sides of the derived partitioning equation, the 
factorization is broken down into three distinct computational phases per iteration:

    Step 1: Factor the Diagonal Block (Unblocked Cholesky)
            A11 = L11 * L11^T  =>  L11 = Cholesky(A11)
            A standard, element-wise scalar Cholesky factorization is performed 
            locally on the tiny b x b matrix A11 to compute L11 in place.

    Step 2: Solve the Panel (Triangular Solve)
            L21 * L11^T = A21  =>  L21 = A21 * (L11^T)^-1
            The vertical panel A21 is updated by solving for L21. In high-
            performance software, this maps directly to the BLAS Level 3 TRSM 
            (Triangular Matrix Solve) routine.

    Step 3: Update the Trailing Submatrix ("Right-Looking" Step)
            L22 * L22^T = A22 - L21 * L21^T  =>  A22 <- A22 - L21 * L21^T
            This is the defining "right-looking" action. The un-factored matrix 
            A22 lying to the right and below is immediately modified by 
            subtracting the outer product of the newly calculated panel. This 
            maps directly to the BLAS Level 3 SYRK (Symmetric Rank-k Update).

--------------------------------------------------------------------------------
3. ALGORITHMIC DESCRIPTION
--------------------------------------------------------------------------------
The Blocked Right-Looking Cholesky Factorization decomposes an n x n real 
symmetric positive definite matrix A into a lower triangular matrix L such that 
A = L * L^T. It uses a top-down, matrix-matrix multiplication approach designed 
to maximize CPU/GPU cache utilization. 

By shifting the bulk of the floating-point operations (O(n^3) operations) to 
Step 3, the algorithm allows hardware to perform dense matrix multiplications 
directly inside high-speed cache registers. This minimizes slow main memory 
(RAM) access bottlenecks and lets libraries like LAPACK run near peak 
theoretical processor speeds.

================================================================================
EXECUTION LOOPS
================================================================================

For i = 1 to n step b:
    
    block_width = min(b, n - i + 1)
    
    A11 = A[i : i+block_width, i : i+block_width]
    A21 = A[i+block_width : n, i : i+block_width]
    A22 = A[i+block_width : n, i+block_width : n]
    
    # --------------------------------------------------------------------------
    # STEP 1: Unblocked Factorization of Diagonal Block (A11 = L11 * L11^T)
    # --------------------------------------------------------------------------
    For row = 1 to block_width:
        For col = 1 to row:
            sum_val = 0
            For k = 1 to col - 1:
                sum_val = sum_val + A11[row][k] * A11[col][k]
                
            If row == col:
                radicand = A11[row][col] - sum_val
                If radicand <= 0:
                    Exit: Matrix is not positive definite!
                A11[row][col] = sqrt(radicand)
            Else:
                A11[row][col] = (A11[row][col] - sum_val) / A11[col][col]
                
    # Continuation check for final block edge
    If i + block_width > n:
        Continue
        
    # --------------------------------------------------------------------------
    # STEP 2: Panel Triangular Solve (Maps to BLAS 'dtrsm')
    # --------------------------------------------------------------------------
    A21 = A21 * inv(transpose(A11))
    
    # --------------------------------------------------------------------------
    # STEP 3: Trailing Submatrix Update (Maps to BLAS 'dsyrk')
    # --------------------------------------------------------------------------
    A22 = A22 - A21 * transpose(A21)

# ------------------------------------------------------------------------------
# CLEANUP: Zero out upper triangle to return a strict lower triangular factor
# ------------------------------------------------------------------------------
For row = 1 to n:
    For col = row + 1 to n:
        A[row][col] = 0

Return A as L

"""

import time

import torch
import triton
import triton.language as tl

from task import input_t, output_t


@triton.jit
def _panel_trsm_kernel(
    A_ptr, L_ptr,
    bk,
    stride_b, stride_r, stride_c,
    n,
    NB: tl.constexpr,
    nb_valid: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    # Every program redundantly factors the same small panel
    # A[bk:bk+nb_valid, bk:bk+nb_valid] itself (same "recompute instead
    # of cross-block sync" trick as cholesky_triton.py's fused kernel),
    # then uses that fully-known local factor to forward-substitute its
    # own BLOCK_I rows of L[:, bk:bk+nb_valid] -- rows inside the panel
    # and rows below it are handled by the exact same formula (see
    # is_diag below), so this one kernel does what would otherwise be a
    # separate "panel factor" + "TRSM" step.
    b = tl.program_id(0)
    pid_i = tl.program_id(1)

    idx = tl.arange(0, NB)               # local row/col index, 0..NB-1
    idx_valid = idx < nb_valid           # only the first nb_valid are real panel columns
    row_idx = idx[:, None]
    col_idx = idx[None, :]

    # --- Step 1: redundantly factor the nb_valid x nb_valid panel locally. ---
    # Lp is the local panel factor as a real (NB, NB) tensor, built up
    # one column at a time via functional (tl.where) updates -- Triton
    # doesn't support indexing a Python list of tensors with a loop
    # variable inside a jit function, so this avoids that entirely
    # (columns >= jp are still all-zero at the time they're read below,
    # same "unwritten == zero" trick as cholesky_triton.py's k < j mask).
    Lp = tl.zeros((NB, NB), dtype=tl.float32)
    for jp in range(nb_valid):
        row_jp = tl.sum(tl.where(row_idx == jp, Lp, 0.0), axis=0)  # (NB,): Lp[jp, :]
        contrib = tl.sum(Lp * row_jp[None, :], axis=1)             # (NB,)

        a_col = tl.load(
            A_ptr + b * stride_b + (bk + idx) * stride_r + (bk + jp) * stride_c,
            mask=idx_valid, other=0.0,
        )
        diff = a_col - contrib
        ljj = tl.sqrt(tl.sum(tl.where(idx == jp, diff, 0.0), axis=0))
        new_col = tl.where(idx == jp, ljj, tl.where(idx > jp, diff / ljj, 0.0))
        Lp = tl.where(col_idx == jp, new_col[:, None], Lp)

    # --- Step 2: forward-substitute this program's BLOCK_I rows. ---
    i = bk + pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    row_mask = i < n

    # L_rows[r, c] = this program's L[i_r, bk + c], same functional
    # build-up as Lp above.
    col_idx_row = tl.arange(0, NB)[None, :]
    L_rows = tl.zeros((BLOCK_I, NB), dtype=tl.float32)
    for jp in range(nb_valid):
        row_jp = tl.sum(tl.where(row_idx == jp, Lp, 0.0), axis=0)  # (NB,): Lp[jp, :], reused from Step 1
        contrib = tl.sum(L_rows * row_jp[None, :], axis=1)         # (BLOCK_I,)

        a_val = tl.load(
            A_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
            mask=row_mask, other=0.0,
        )
        diff = a_val - contrib
        ljj = tl.sum(tl.where(idx == jp, row_jp, 0.0), axis=0)

        # Rows inside the panel itself (i == bk + jp, the current local
        # pivot row) get the diagonal value directly instead of
        # diff / ljj -- reusing Lp avoids recomputing sqrt here.
        is_diag = i == (bk + jp)
        val = tl.where(is_diag, ljj, diff / ljj)
        L_rows = tl.where(col_idx_row == jp, val[:, None], L_rows)

        tl.store(
            L_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
            val, mask=row_mask,
        )


@triton.jit
def _syrk_tile(A_ptr, L_ptr, l_m, rm, mask_m, b, pid_n, start,
               stride_b, stride_r, stride_c, bk, n,
               NB: tl.constexpr, nb_valid: tl.constexpr, BLOCK_N: tl.constexpr):
    rn = start + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rn < n
    rk = tl.arange(0, NB)
    k_mask = rk < nb_valid

    l_n = tl.load(
        L_ptr + b * stride_b + rn[:, None] * stride_r + (bk + rk)[None, :] * stride_c,
        mask=mask_n[:, None] & k_mask[None, :], other=0.0,
    )
    # input_precision="tf32x3": splits each fp32 operand into a TF32-
    # representable hi part and a lo residual (x = x_hi + x_lo), then
    # computes x*y ~= x_hi*y_hi + x_hi*y_lo + x_lo*y_hi as three TF32
    # tensor-core matmuls accumulated together -- recovers close to full
    # fp32 accuracy while still running on tensor cores, unlike "ieee"
    # (plain CUDA-core FMA, no tensor cores) or plain "tf32" (single-pass,
    # only ~8e-2 error, nowhere near enough precision for Cholesky).
    #
    # Tried loading the K-dimension one column at a time (small persistent
    # accumulator, like cuSOLVER's ~40 registers/thread) instead of
    # materializing full l_m/l_n tiles before this dot: it did cut
    # registers 166->52 and raise occupancy 29.5%->59.2%, but was 69%
    # *slower* anyway -- the full-tile loads above are contiguous along
    # the K dimension (one coalesced vectorized load per row), whereas
    # loading single K-columns one at a time is a strided/uncoalesced
    # gather across rows, done NB times. Register pressure was real but
    # wasn't the binding constraint; memory coalescing was.
    #
    # Also tried TMA tensor-descriptor loads (tl.make_tensor_descriptor +
    # desc.load/.store) instead of plain tl.load: registers barely moved
    # (166->168) and occupancy got *worse* (29.5%->22.2%), plus every
    # kernel launch now paid for a host-side scratch allocation
    # (triton.set_allocator callback -> torch.empty, one per launch) that
    # plain tl.load never needed. 32x32 tiles are too small to amortize
    # TMA's per-descriptor setup cost, and descriptors were built fresh
    # per-call here rather than reused across a pipelined loop.
    #
    # Also tried chunking the K-reduction into a tl.range(..., BLOCK_K,
    # warp_specialize=True) loop (Blackwell-only automatic async
    # load/compute partitioning, meant to overlap a chunk's load with the
    # previous chunk's tl.dot): the compiler's automatic-warp-
    # specialization pass crashes on this kernel's shape ("PassManager::run
    # failed" inside tritongpu-automatic-warp-specialization, which tries
    # to route the accumulator through Blackwell tensor memory) -- matches
    # its docs' own caveat that it only supports "simple matmul loops" and
    # will expand over time. Without the flag, the same chunked loop
    # compiles fine but is ~6% slower than the single-shot version below
    # (reloads rm's L-row data per branch instead of once), for no benefit
    # since no pipelining actually happens. Reverted to the single-shot
    # tl.dot form.
    update = tl.dot(l_m, tl.trans(l_n), input_precision="tf32x3")

    a_ptrs = A_ptr + b * stride_b + rm[:, None] * stride_r + rn[None, :] * stride_c
    a_mask = mask_m[:, None] & mask_n[None, :]
    a_old = tl.load(a_ptrs, mask=a_mask, other=0.0)
    tl.store(a_ptrs, a_old - update, mask=a_mask)


@triton.jit
def _syrk_kernel(
    A_ptr, L_ptr,
    bk,
    stride_b, stride_r, stride_c,
    n,
    NB: tl.constexpr,
    nb_valid: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
):
    # A[bk+nb_valid:, bk+nb_valid:] -= L_below @ L_below^T, where
    # L_below = L[bk+nb_valid:, bk:bk+nb_valid]. This is the dense matmul
    # that carries the bulk of the algorithm's FLOPs -- tl.dot runs it at
    # GEMM throughput instead of the row-at-a-time dot products
    # cholesky_triton.py does.
    #
    # Only col_tile <= row_tile holds real (on/below-diagonal) work, so the
    # column axis is folded in half: program pid_n_lo also handles the
    # mirror column tile pid_n_hi = NUM_N_TILES-1-pid_n_lo, each guarded by
    # its own "<= pid_m" check. This halves launched blocks versus a full
    # square grid (matching the (y, y/2) grid shape cuSOLVER's own
    # potrf_syrk_nc_kernel uses) while keeping the count of fully-wasted
    # (both tiles above the diagonal) launches low -- pairing a low column
    # index with a far, high one means for most rows at least one side of
    # the pair is valid, unlike pairing adjacent columns (2z, 2z+1) which
    # are almost always both-valid or both-invalid together.
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n_lo = tl.program_id(2)
    pid_n_hi = NUM_N_TILES - 1 - pid_n_lo

    start = bk + nb_valid
    rm = start + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rm < n
    rk = tl.arange(0, NB)
    k_mask = rk < nb_valid

    l_m = tl.load(
        L_ptr + b * stride_b + rm[:, None] * stride_r + (bk + rk)[None, :] * stride_c,
        mask=mask_m[:, None] & k_mask[None, :], other=0.0,
    )

    if pid_n_lo <= pid_m:
        _syrk_tile(A_ptr, L_ptr, l_m, rm, mask_m, b, pid_n_lo, start,
                   stride_b, stride_r, stride_c, bk, n,
                   NB=NB, nb_valid=nb_valid, BLOCK_N=BLOCK_N)
    if pid_n_hi <= pid_m and pid_n_hi != pid_n_lo:
        _syrk_tile(A_ptr, L_ptr, l_m, rm, mask_m, b, pid_n_hi, start,
                   stride_b, stride_r, stride_c, bk, n,
                   NB=NB, nb_valid=nb_valid, BLOCK_N=BLOCK_N)


def custom_kernel(data: input_t) -> output_t:
    # .clone() is required, not just .to()/.contiguous(): the SYRK step
    # below writes into A_ptr in place (maintaining the running Schur
    # complement), and .to(torch.float32).contiguous() is a no-op (no
    # copy) whenever data is already float32-contiguous -- without the
    # clone this would silently mutate the caller's input tensor.
    A = data.to(torch.float32).contiguous().clone()
    batch, n, _ = A.shape
    L = torch.zeros_like(A)

    stride_b, stride_r, stride_c = L.stride()
    NB = 32  # fixed power-of-2 buffer width (required by tl.arange)
    BLOCK_I = 32
    BLOCK_M = BLOCK_N = 32

    for bk in range(0, n, NB):
        nb_valid = min(NB, n - bk)

        grid_panel = (batch, triton.cdiv(n - bk, BLOCK_I))
        _panel_trsm_kernel[grid_panel](
            A, L, bk, stride_b, stride_r, stride_c, n,
            NB=NB, nb_valid=nb_valid, BLOCK_I=BLOCK_I,
        )

        trailing = n - (bk + nb_valid)
        if trailing > 0:
            num_m_tiles = triton.cdiv(trailing, BLOCK_M)
            num_n_tiles = triton.cdiv(trailing, BLOCK_N)
            grid_syrk = (batch, num_m_tiles, triton.cdiv(num_n_tiles, 2))
            _syrk_kernel[grid_syrk](
                A, L, bk, stride_b, stride_r, stride_c, n,
                NB=NB, nb_valid=nb_valid, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                NUM_N_TILES=num_n_tiles,
            )

    return L


if __name__ == "__main__":
    #batch, n = 4, 2048
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
