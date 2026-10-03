"""
Standalone Modal script to test whether tl.range(..., warp_specialize=True)
crashes the same way on B200 as it does on GB10 (this repo's local GPU).

Locally on GB10 (sm_121), compiling _syrk_kernel with warp_specialize=True on
the K-reduction loop crashes with:
    RuntimeError: PassManager::run failed
inside the tritongpu-automatic-warp-specialization pass (which tries to route
the accumulator through Blackwell tensor memory). Without the flag, the same
code compiles and runs correctly. This script reproduces the exact same
kernel structure on a B200 to see whether that's a GB10-specific hardware/
compiler-support gap or a general Triton limitation across all Blackwell
targets.

Usage: modal run test_warp_specialize_b200.py
"""
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "triton")
app = modal.App("triton-warp-specialize-test", image=image)


def run_test():
    import traceback

    import torch
    import triton
    import triton.language as tl

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
        b = tl.program_id(0)
        pid_i = tl.program_id(1)

        idx = tl.arange(0, NB)
        idx_valid = idx < nb_valid
        row_idx = idx[:, None]
        col_idx = idx[None, :]

        Lp = tl.zeros((NB, NB), dtype=tl.float32)
        for jp in range(nb_valid):
            row_jp = tl.sum(tl.where(row_idx == jp, Lp, 0.0), axis=0)
            contrib = tl.sum(Lp * row_jp[None, :], axis=1)

            a_col = tl.load(
                A_ptr + b * stride_b + (bk + idx) * stride_r + (bk + jp) * stride_c,
                mask=idx_valid, other=0.0,
            )
            diff = a_col - contrib
            ljj = tl.sqrt(tl.sum(tl.where(idx == jp, diff, 0.0), axis=0))
            new_col = tl.where(idx == jp, ljj, tl.where(idx > jp, diff / ljj, 0.0))
            Lp = tl.where(col_idx == jp, new_col[:, None], Lp)

        i = bk + pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        row_mask = i < n

        col_idx_row = tl.arange(0, NB)[None, :]
        L_rows = tl.zeros((BLOCK_I, NB), dtype=tl.float32)
        for jp in range(nb_valid):
            row_jp = tl.sum(tl.where(row_idx == jp, Lp, 0.0), axis=0)
            contrib = tl.sum(L_rows * row_jp[None, :], axis=1)

            a_val = tl.load(
                A_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
                mask=row_mask, other=0.0,
            )
            diff = a_val - contrib
            ljj = tl.sum(tl.where(idx == jp, row_jp, 0.0), axis=0)

            is_diag = i == (bk + jp)
            val = tl.where(is_diag, ljj, diff / ljj)
            L_rows = tl.where(col_idx_row == jp, val[:, None], L_rows)

            tl.store(
                L_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
                val, mask=row_mask,
            )

    @triton.jit
    def _syrk_tile(A_ptr, L_ptr, rm, mask_m, b, pid_n, start,
                   stride_b, stride_r, stride_c, bk, n,
                   NB: tl.constexpr, nb_valid: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        rn = start + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = rn < n

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in tl.range(0, nb_valid, BLOCK_K, warp_specialize=True):
            rk = k0 + tl.arange(0, BLOCK_K)
            k_mask = rk < nb_valid
            a_chunk = tl.load(
                L_ptr + b * stride_b + rm[:, None] * stride_r + (bk + rk)[None, :] * stride_c,
                mask=mask_m[:, None] & k_mask[None, :], other=0.0,
            )
            b_chunk = tl.load(
                L_ptr + b * stride_b + rn[:, None] * stride_r + (bk + rk)[None, :] * stride_c,
                mask=mask_n[:, None] & k_mask[None, :], other=0.0,
            )
            acc = tl.dot(a_chunk, tl.trans(b_chunk), acc, input_precision="ieee")

        a_ptrs = A_ptr + b * stride_b + rm[:, None] * stride_r + rn[None, :] * stride_c
        a_mask = mask_m[:, None] & mask_n[None, :]
        a_old = tl.load(a_ptrs, mask=a_mask, other=0.0)
        tl.store(a_ptrs, a_old - acc, mask=a_mask)

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
        BLOCK_K: tl.constexpr,
        NUM_N_TILES: tl.constexpr,
    ):
        b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n_lo = tl.program_id(2)
        pid_n_hi = NUM_N_TILES - 1 - pid_n_lo

        start = bk + nb_valid
        rm = start + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = rm < n

        if pid_n_lo <= pid_m:
            _syrk_tile(A_ptr, L_ptr, rm, mask_m, b, pid_n_lo, start,
                       stride_b, stride_r, stride_c, bk, n,
                       NB=NB, nb_valid=nb_valid, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        if pid_n_hi <= pid_m and pid_n_hi != pid_n_lo:
            _syrk_tile(A_ptr, L_ptr, rm, mask_m, b, pid_n_hi, start,
                       stride_b, stride_r, stride_c, bk, n,
                       NB=NB, nb_valid=nb_valid, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

    def custom_kernel(data):
        A = data.to(torch.float32).contiguous().clone()
        batch, n, _ = A.shape
        L = torch.zeros_like(A)

        stride_b, stride_r, stride_c = L.stride()
        NB = 32
        BLOCK_I = 32
        BLOCK_M = BLOCK_N = 32
        BLOCK_K = 16

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
                    BLOCK_K=BLOCK_K, NUM_N_TILES=num_n_tiles,
                )

        return L

    device = torch.device("cuda")
    info = {"gpu_name": torch.cuda.get_device_name(0), "triton_version": triton.__version__}

    gen = torch.Generator(device=device).manual_seed(42)
    batch, n = 4, 2048
    a = torch.randn((batch, n, n), device=device, dtype=torch.float32, generator=gen)
    A = (a @ a.transpose(-1, -2)) / n
    A.diagonal(dim1=-2, dim2=-1).add_(1.0)

    try:
        custom_kernel(A)  # warmup / triggers compile
        torch.cuda.synchronize()
        L = custom_kernel(A)
        torch.cuda.synchronize()
        err = (L @ L.transpose(-1, -2) - A).abs().amax()
        return {**info, "status": "ok", "max_err": err.item()}
    except Exception as e:
        return {**info, "status": "error", "error": str(e), "traceback": traceback.format_exc()}


@app.cls(gpu="B200", scaledown_window=60)
class WarpSpecializeTest:
    @modal.method()
    def run(self):
        return run_test()


@app.local_entrypoint()
def main():
    runner = WarpSpecializeTest()
    result = runner.run.remote()
    print("=" * 70)
    for k, v in result.items():
        if k == "traceback":
            continue
        print(f"{k}: {v}")
    if result.get("status") == "error":
        print("-" * 70)
        print(result["traceback"])
    print("=" * 70)
