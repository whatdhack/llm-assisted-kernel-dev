"""
Helion port of cholesky_loops.py.

Same algorithm as cholesky_loops.py / cholesky_triton.py: an unblocked,
column-by-column Cholesky factorization launched from a host-side Python
loop over columns j = 0..n-1 (O(n) kernel launches, not a blocked
panel+TRSM+SYRK scheme). Where cholesky_loops.py expresses the per-column
update with torch ops and cholesky_triton.py hand-writes it in Triton,
here it's expressed with Helion's tile DSL so Helion's autotuner picks the
block sizes / launch config instead of them being hand-tuned constants.

See cholesky_loops.py's module docstring for the math (per-column formula
xjj = sqrt(ajj - sum_k xjk^2), xij = (aij - sum_k xik.xjk) / xjj).
"""

import torch

import helion
import helion.language as hl

from task import input_t, output_t


@helion.kernel(autotune_effort="none")
def _cholesky_column(
    A: torch.Tensor,
    L: torch.Tensor,
    L_T: torch.Tensor,
    A_diag: torch.Tensor,
    L_diag: torch.Tensor,
    j_arg: torch.Tensor,
) -> None:
    # j_arg is a 1-element tensor rather than a plain Python int so Helion
    # treats the column index as a data-dependent runtime value: this
    # kernel is compiled once and reused for all n columns, instead of
    # being re-specialized (and re-autotuned) for every distinct j.
    #
    # Two Helion indexing quirks (as of helion 1.2.0) shape this function:
    #  - A subscript can't use the same dynamic scalar index `j` for two
    #    different dims (an A[b, j, j]-style diagonal gather); worked
    #    around below via A_diag/L_diag, torch.diagonal(A)/(L) views
    #    (shape [batch, n, 1]) so the diagonal load/store only needs `j`
    #    once.
    #  - A dynamic scalar index can't be the *last* dim of a subscript
    #    (miscomputes an output offset); worked around by never putting
    #    `j` last: reads of A use A[b, j, i] instead of A[b, i, j] (valid
    #    since A is symmetric), and writes to L go through L_T = L.T (a
    #    view, same storage) as L_T[b, j, i] instead of L[b, i, j].
    batch, n, _ = A.size()

    for tile_b in hl.tile(batch):
        j = j_arg[0]
        # sum_k L[b, j, k]^2 for k in [0, j) -> subtraction term for L[j, j].
        sumsq = hl.zeros([tile_b], dtype=torch.float32)
        for tile_k in hl.tile(j):
            l_jk = L[tile_b, j, tile_k]
            sumsq = sumsq + torch.sum(l_jk * l_jk, dim=-1)
        l_jj = torch.sqrt(A_diag[tile_b, j, :] - sumsq[:, None])
        L_diag[tile_b, j, :] = l_jj

        # L[i, j] = (A[i, j] - sum_k L[i, k].L[j, k]) / L[j, j] for i > j.
        for tile_i in hl.tile(j + 1, n):
            acc = hl.zeros([tile_b, tile_i], dtype=torch.float32)
            for tile_k in hl.tile(j):
                l_jk = L[tile_b, j, tile_k]
                l_ik = L[tile_b, tile_i, tile_k]
                acc = acc + torch.sum(l_ik * l_jk[:, None, :], dim=-1)
            L_T[tile_b, j, tile_i] = (A[tile_b, j, tile_i] - acc) / l_jj


def custom_kernel(data: input_t) -> output_t:
    A = data.to(torch.float32)
    batch, n, _ = A.shape
    L = torch.zeros_like(A)
    L_T = L.transpose(1, 2)
    A_diag = torch.diagonal(A, dim1=1, dim2=2).unsqueeze(-1)
    L_diag = torch.diagonal(L, dim1=1, dim2=2).unsqueeze(-1)

    for j in range(n):
        j_arg = torch.tensor([j], device=A.device, dtype=torch.int64)
        _cholesky_column(A, L, L_T, A_diag, L_diag, j_arg)

    return L


if __name__ == "__main__":
    batch, n = 4, 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(42)

    a = torch.randn((batch, n, n), device=device, dtype=torch.float32, generator=gen)
    A = (a @ a.transpose(-1, -2)) / n
    A.diagonal(dim1=-2, dim2=-1).add_(1.0)  # guaranteed SPD

    L = custom_kernel(A)
    err = (L @ L.transpose(-1, -2) - A).abs().amax()
    print("max |L L^T - A| =", err.item())
