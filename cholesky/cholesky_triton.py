import time

import torch
import triton
import triton.language as tl

from task import input_t, output_t


@triton.jit
def _column_kernel(
    A_ptr, L_ptr,
    j,                              # column being factored this launch (0..n-1)
    stride_b, stride_r, stride_c,   # shared by A and L: both must have matching strides
    n,
    BLOCK_I: tl.constexpr,          # rows of column j handled per program (grid axis 1)
    BLOCK_K: tl.constexpr,          # k-chunk size for the row 0..j-1 reduction loop
):
    # Fuses what used to be two launches (_diag_kernel + _column_kernel)
    # into one: every block already loads the full row L[:, j, :] to
    # dot against its rows, so it gets sum(L[:, j, :] ** 2) — and hence
    # L[:, j, j] — "for free" from that same data instead of reading a
    # value written by a separate diag launch. That matters because
    # Triton gives no ordering/sync *between* blocks in one launch, so
    # block pid_i > 0 can't safely depend on block 0 having written
    # L[:, j, j] first; recomputing it redundantly in every block
    # sidesteps that instead of relying on cross-block sync.

    # Grid is (batch, grid_i): axis 0 is one program per batch matrix
    # (untiled — b is just "which matrix"), axis 1 tiles the n-j-1
    # remaining rows of column j into BLOCK_I-sized chunks. BLOCK_I has
    # nothing to do with the batch dimension.
    b = tl.program_id(0)
    pid_i = tl.program_id(1)

    # The BLOCK_I rows of column j this program is responsible for.
    i = j + 1 + pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    row_mask = i < n

    # acc[r] will hold dot(L[i_r, :j], L[j, :j]) — the subtraction term
    # for L[i_r, j], where i_r is the r-th row in this program's
    # BLOCK_I-sized slice of i. sumsq accumulates sum(L[j, :j] ** 2) — the
    # subtraction term for the diagonal L[j, j]. Both are reductions
    # over the same k range, so they're computed in the same loop
    # instead of two separate kernels (mirrors the merge in
    # cholesky_loops.py, but see the note above on why every block
    # must redo this work rather than share one result).
    acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
    sumsq = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, n, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        # k < j, not k < n: k == j is being written by pid_i == 0 in
        # this very launch, so reading it here would race that store.
        k_mask = k < j

        # Row j's already-known entries for this k-chunk — shared by
        # both the sumsq and acc reductions below.
        l_j_blk = tl.load(
            L_ptr + b * stride_b + j * stride_r + k * stride_c,
            mask=k_mask, other=0.0,
        )
        sumsq += tl.sum(l_j_blk * l_j_blk, axis=0)

        # (BLOCK_I, BLOCK_K) tile: this program's rows x this k-chunk.
        l_i_ptrs = L_ptr + b * stride_b + i[:, None] * stride_r + k[None, :] * stride_c
        l_i_blk = tl.load(
            l_i_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0,
        )
        acc += tl.sum(l_i_blk * l_j_blk[None, :], axis=1)

    # Diagonal: L[j, j] = sqrt(A[j, j] - sumsq). Only pid_i == 0 writes
    # it (every block computed the same value redundantly above; only
    # one needs to store it).
    a_jj = tl.load(A_ptr + b * stride_b + j * stride_r + j * stride_c)
    l_jj = tl.sqrt(a_jj - sumsq)
    if pid_i == 0:
        tl.store(L_ptr + b * stride_b + j * stride_r + j * stride_c, l_jj)

    # Off-diagonal: L[i, j] = (A[i, j] - acc) / L[j, j] for this
    # program's BLOCK_I rows.
    a_ij = tl.load(
        A_ptr + b * stride_b + i * stride_r + j * stride_c, mask=row_mask, other=0.0,
    )
    l_ij = (a_ij - acc) / l_jj
    tl.store(L_ptr + b * stride_b + i * stride_r + j * stride_c, l_ij, mask=row_mask)


def custom_kernel(data: input_t) -> output_t:
    #A = data.to(torch.float32).contiguous()
    A = data
    batch, n, _ = A.shape
    L = torch.zeros_like(A)  # inherits A's strides — see stride_b/r/c note below

    # These strides are computed once from L and reused for both A_ptr
    # and L_ptr inside the kernel, so A and L must share the same
    # memory layout (true here since L = zeros_like(A)).
    stride_b, stride_r, stride_c = L.stride()
    BLOCK_I = 32                                    # row-tile size, see _column_kernel
    BLOCK_K = min(64, triton.next_power_of_2(n))     # k-chunk size for the reduction loop

    # One kernel launch per column j — this is what makes the whole
    # algorithm O(n) launches rather than the O(n/BLOCK) a blocked
    # (panel + TRSM + SYRK) implementation would use.
    for j in range(n):
        num_rows = n - (j + 1)
        grid_i = max(triton.cdiv(num_rows, BLOCK_I), 1)  # >=1 so pid_i==0 always writes L[j, j]
        _column_kernel[(batch, grid_i)](
            A, L, j, stride_b, stride_r, stride_c, n,
            BLOCK_I=BLOCK_I, BLOCK_K=BLOCK_K,
        )

    return L


if __name__ == "__main__":
    batch, n = 4, 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(42)

    a = torch.randn((batch, n, n), device=device, dtype=torch.float32, generator=gen)
    A = (a @ a.transpose(-1, -2)) / n
    A.diagonal(dim1=-2, dim2=-1).add_(1.0)  # guaranteed SPD

    custom_kernel(A)  # warmup: triggers Triton's first-call compilation
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    L = custom_kernel(A)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"custom_kernel time: {time.perf_counter() - t0:.6f} s")

    err = (L @ L.transpose(-1, -2) - A).abs().amax()
    print("max |L L^T - A| =", err.item())
