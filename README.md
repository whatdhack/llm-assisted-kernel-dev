# LLM-Assisted Kernel Development in CuTe DSL, Helion and Triton

Companion code for the article of the same name. Four problems, each worked
through several implementations in different DSLs, on NVIDIA Blackwell (B200,
sm_100a) unless noted.

The files here are copies taken from a private working repository. They are the
kernels the article discusses, not a runnable benchmark harness — see
[Running these](#running-these) below.

## Dual GEMM (`dual_gemm/`)

NVFP4 dual GEMM: two GEMMs sharing an A operand.

| File | What it is |
| :--- | :--- |
| `submission.py` | The competition's CuTe DSL single-warp baseline. |
| `reference.py` | PyTorch reference used for correctness. |
| `dual_gemm_torch.py` | PyTorch derivation exploring split-K and merged-B. |
| `dual_gemm_cute.py` | CuTe DSL as flat functions (`@cute.kernel` + `@cute.jit`): one CTA computes one output tile, one launch per tile grid. |
| `dual_gemm_opt_cute.py` | Optimized persistent kernel, `class Sm100BlockScaledPersistentDenseGemmKernel`, following CUTLASS's persistent tile-scheduling pattern. Derived from CUTLASS's `dense_blockscaled_gemm_persistent_prefetch.py`. |

## Group GEMM (`group_gemm/`)

NVFP4 grouped GEMM. The progression is the interesting part: metadata moves
from runtime GPU reads to compile-time specialization, then the whole thing is
rebuilt on CUTLASS's grouped kernel.

| File | What it is |
| :--- | :--- |
| `group_gemm_cute.py` | First derivation from `submission.py`, plus a Modal benchmark harness, `clock64` instrumentation (`mov.u64 %clock64`) and a `debug_buf: cute.Tensor`. |
| `group_gemm_cute2.py` | Consolidates three separate metadata tensors into one flat `metadata_list` (`dtype=torch.int64`). |
| `group_gemm_cute4.py` | Moves group dispatch from runtime GPU reads to compile-time tuple embedding, dropping the metadata tensor entirely and caching a compiled specialization per problem shape. **Best score: 50/71.** |
| `group_gemm_cute5.py` | Port of CUTLASS `grouped_blockscaled_gemm.py` (`Sm100GroupedBlockScaledGemmKernel`) onto a Modal B200 harness. 3,279 → 3,849 lines. |
| `group_gemm_cute5_v2.py` | Benchmarking expansion, 3,849 → 5,516 lines: dual compile paths (`compile_kernel_original` plus a custom instrumented profiling path). |

## Cholesky (`cholesky/`)

Blocked right-looking Cholesky, `A = L Lᵀ`, in three steps per panel:

1. **POTF2** — factor the diagonal block, `A11 = L11 L11ᵀ`.
2. **TRSM** — solve the vertical panel, `L21 = A21 (L11ᵀ)⁻¹`.
3. **SYRK** — update the trailing submatrix, `A22 ← A22 − L21 L21ᵀ`.

FP32 accuracy on tensor cores comes from **3×TF32**: split each operand into a
tf32 `hi` and the residual `lo`, then
`x·y ≈ x_hi·y_hi + x_hi·y_lo + x_lo·y_hi` — three tf32 passes for ~22 mantissa
bits against fp32's 24. Plain tf32 (11 bits) is not accurate enough here.

| File | What it is |
| :--- | :--- |
| `cholesky_cute_tcgen05_blocked.py` | CuTe DSL blocked `tcgen05` implementation, with the TMEM fix described below. |
| `cholesky_cute_permit_late.py` | The same kernel with the TMEM allocation permit released late — kept as the A/B control. |
| `cholesky_gluon_tcgen05_warpspec.py` | Triton Gluon, warp-specialized `tcgen05` (TMA / MMA / epilogue partitions). |
| `cholesky_gluon_tcgen05_persistent.py` | Triton Gluon, persistent `tcgen05`. |
| `cholesky_gluon_tcgen05_blocked.py` | Triton Gluon, blocked `tcgen05`. |
| `cholesky_gluon_blocked.py` | First Triton Gluon version, FP32 `dot_fma`. |
| `cholesky_triton_blocked.py` | Triton blocked Cholesky with 3×TF32. ~2× cuSOLVER geomean; 3×TF32 was worth ~5%. |
| `cholesky_triton.py` | Baseline Triton Cholesky. |
| `cholesky_helion.py` | Triton Helion version. The unblocked variant hit indexing errors on GB10. |
| `reference.py` | PyTorch/cuSOLVER baseline, `torch.linalg.cholesky_ex(data, check_errors=False).L`. |
| `submission.py`, `starter.py` | Problem entrypoints. |
| `validation.py`, `eval.py` | Correctness validation and benchmark evaluation. |

### The TMEM finding

The CuTe port initially trailed Gluon by 1.28× geomean, and the cause was not
in the kernel body — three rounds of shared-memory and layout work moved
nothing. IKET (CuTe DSL's in-kernel event tracing) showed the SYRK running
**one CTA per SM against a shared-memory block limit of four**, with 384 of
TMEM's 512 columns idle.

The kernel held its `tcgen05` allocation permit for the CTA's whole lifetime:
`relinquish_alloc_permit()` sat beside `tmem.free()` at the end. Until that
permit is released, the next CTA on the SM cannot allocate. Moving it to
immediately after `wait_for_alloc()` — where its promise ("this warp will not
allocate again") is already true — restored 4 CTAs/SM on all 148 SMs.

Measured on a B200 at batch 1 / n = 8192:

```
IKET, first 4 bk steps        before     after
  concurrent CTAs/SM               1         4
  SYRK launch span            165 us     87 us

Kineto, full factorization    before     after     gluon
  GPU busy                    18.241    12.472    12.438 ms
    syrk (255 launches)       13.543     7.700     7.994
    panel (256)                4.661     4.735     4.408

A/B, one container, 15 shapes (geomean, ms)
  permit late 2.847 | permit early 2.365 | gluon 2.390
```

1.20× geomean, scaling with size — 1.43× at n=8192, 1.76× at 16384, 1.94× at
32768 — with bit-identical reconstruction error. Against Gluon the port went
from 1.28× slower to 0.99×, ahead at every shape from n=4096 up.

`cholesky_cute_tcgen05_blocked.py` also carries the IKET instrumentation that
found this: 13 ranges and 2 marks (`panel_L11`, `tmem_alloc`, `split`, `mma`,
`epilogue`, …). CuTe DSL strips every `iket` op at JIT time unless
`CUTE_DSL_COMPILER_OPT` contains `iket`, which `run-iket` sets, so ordinary
runs are unaffected.

## MoE (`moe/`)

Mixture-of-experts with FP8 block-scaled weights, from the FlashInfer Bench
work. DeepSeek-V3 style no-aux routing, then per-expert GEMM1 → SwiGLU → GEMM2.

| File | What it is |
| :--- | :--- |
| `moe_fp8fpX_fused.py` | The optimized kernel. Hybrid split: dequantization (FP8 → BF16) and routing on the host, expert computation (GEMM1 → SwiGLU → GEMM2) in the device kernel. Carries `bf16`/`fp16`/`fp32`/`tf32` variants of both the dequantization and the expert computation, which is how the article's question — whether the FP8 → FP32 conversion can be avoided by staying in lower precision — gets measured. BF16 is the default: same dynamic range as FP32, so none of FP16's ±65504 overflow risk, at half the memory. |
| `moe_fibench_ref.py` | PyTorch reference. Takes FP8 e4m3fn hidden states and both GEMM weight sets with their block scales, applies the DeepSeek-V3 no-aux routing (`sigmoid(logits) + bias`, grouped top-k, `routed_scaling_factor`), and runs the experts in full precision. Shapes are H=7168, I=2048, 256 global experts, 32 local. |

Two notes on naming, since both differ from how the article lists them:
`moe_fp8fpX_fused.py` is a **Helion** kernel (`import helion`, `hl.*`), not
Triton, despite living under `solution/triton/` upstream; and
`moe_fibench_ref.py` is a plain **PyTorch** reference with no Triton in it.

Unlike everything else here, these two were not in the private working repo —
they are fetched from the public
[flashinfer-bench-starter-kit](https://github.com/whatdhack/flashinfer-bench-starter-kit/tree/main/solution/triton)
(`solution/triton/`, upstream at `1663ebbd6c1e`).

## Harness (`utils/`)

The per-problem task definitions and helpers the kernels import. Each problem
has its own, so they are kept in subdirectories rather than flattened:

| Path | What it is |
| :--- | :--- |
| `utils/cholesky/task.py`, `task.yml` | Input/output types, and the benchmark/test grid: 15 benchmark shapes from 4096x32 up to a single 32768x32768, plus the correctness cases, tolerances and `ranking_by: geom`. |
| `utils/cholesky/bench_leaderboard.py` | `generate_input()` and `bench_one()` — the warmup-2, median-of-5 protocol every number here was taken with. |
| `utils/cholesky/utils.py` | Shared helpers (from `problems/pmpp_v2/utils.py` upstream). |
| `utils/dual_gemm/`, `utils/group_gemm/` | The same three files for those problems. |

## Modal runners (`modal/`)

The scripts that ran everything on a Modal B200 — benchmarks, ncu captures,
IKET traces, and the one-off probes behind individual findings.

| Path | What it is |
| :--- | :--- |
| `modal/cholesky/` (53 scripts) | Benchmarks (`bench_*`), ncu captures (`ncu_*`), sweeps (`sweep_*`), probes (`probe_*`), plus the IKET tooling: `iket_cute_b200.py` (capture), `iket_phase_summary.py` (per-launch phase table), `iket_slot_view.py` (re-lays a trace onto per-SM slot rows so residency is visible), and `bench_permit_ab_b200.py` (the A/B above). |
| `modal/dual_gemm/`, `modal/group_gemm/` | The Modal harnesses for those problems. |

Pins matter in these: the images pin `nvidia-cutlass-dsl==4.7.1`,
`triton==3.7.1` and an exact `nsight-compute` package, because two of the CuTe
port's bugs were version-sensitive DSL semantics.

## Running these

Start from a problem's `eval.py` with its `utils/<problem>/` files alongside,
or drive a kernel directly through the matching `modal/` runner (each needs a
Modal account and a B200). Still **not** here:

- CUTLASS's own examples (`grouped_blockscaled_gemm.py`,
  `dense_blockscaled_gemm_persistent_prefetch.py`), which the CuTe versions
  were derived from. Those ship with CUTLASS.
- The captured traces themselves (`.ncu-rep`, `.pftrace`, Kineto JSON) — about
  1.7 GB, and gitignored here.

The MoE files are standalone: they need `helion`, `safetensors` and a workload
file (`--workload`, or `MOE_WORKLOAD_DIR`), and are not wired into the
harnesses above.

Versions the measurements were taken at: `nvidia-cutlass-dsl==4.7.1`,
`triton==3.7.1`, torch 2.13.0+cu130, ncu 2026.3.0, on a Modal B200.

## Provenance

Everything except `moe/` is copied from a private working repository. The two
`moe/` files come from the public
[flashinfer-bench-starter-kit](https://github.com/whatdhack/flashinfer-bench-starter-kit/tree/main/solution/triton)
at upstream `1663ebbd6c1e`.

Three absolute paths tied to the original machine were replaced with
environment lookups, so these differ from their sources by that much:
`moe/moe_fp8fpX_fused.py` (one hardcoded workload path, now
`MOE_WORKLOAD_DIR`), and `modal/cholesky/ncu_source_syrk_b200.py` and
`ncu_import_source.py` (a hardcoded scratch directory, now `SCRATCH_DIR`
defaulting under the system temp dir).
