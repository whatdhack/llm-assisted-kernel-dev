"""
Collects a Kineto (torch.profiler chrome trace) profile of a cholesky
implementation from python_standalone/ on a real B200 via Modal, matching
the existing traces under python_standalone/outputs/traces/ (batch=1,
n=8192, the official GPU MODE leaderboard case with seed 48192).

Usage:
    modal run profile_cholesky_b200.py
    modal run profile_cholesky_b200.py --module cholesky_gluon_tcgen05_blocked --batch 1 --n 8192

The trace is written locally to
python_standalone/outputs/traces/<module>__custom_kernel__<timestamp>.json
"""
import os
import time

import modal

import os as _os
# Kernel sources live one level up, in the cholesky_py problem directory.
LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
TRACE_DIR = os.path.join(LOCAL_DIR, "outputs", "traces")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton")
    .add_local_dir(LOCAL_DIR, "/root/python_standalone")
)

app = modal.App("cholesky-profile-b200", image=image)


@app.function(gpu="B200", timeout=1800)
def profile(module_name: str, batch: int, n: int, seed: int):
    import importlib
    import sys

    import torch
    import triton

    sys.path.insert(0, "/root/python_standalone")
    import bench_leaderboard

    mod = importlib.import_module(module_name)
    custom_kernel = mod.custom_kernel

    data = bench_leaderboard.generate_input(batch, n, 2, seed)

    # Warm up: JIT-compile every kernel variant and settle the allocator so
    # the captured trace contains only steady-state launches.
    for _ in range(2):
        L = custom_kernel(data)
    torch.cuda.synchronize()

    err = (L @ L.transpose(-1, -2) - data).abs().amax().item()
    del L

    trace_path = f"/tmp/{module_name}_kineto.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        custom_kernel(data)
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)

    table = prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=25, max_name_column_width=110
    )

    with open(trace_path, "rb") as f:
        blob = f.read()

    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "max_abs_err": err,
        "table": table,
        "trace": blob,
    }


@app.local_entrypoint()
def main(
    module: str = "cholesky_gluon_tcgen05_blocked",
    batch: int = 1,
    n: int = 8192,
    seed: int = 48192,
):
    result = profile.remote(module, batch, n, seed)

    os.makedirs(TRACE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(TRACE_DIR, f"{module}__custom_kernel__{stamp}.json")
    with open(out_path, "wb") as f:
        f.write(result["trace"])

    print("=" * 70)
    print("gpu_name:      ", result["gpu_name"])
    print("torch_version: ", result["torch_version"])
    print("triton_version:", result["triton_version"])
    print(f"case:           batch={batch}, n={n}, seed={seed}")
    print("max |L L^T - A|:", result["max_abs_err"])
    print("=" * 70)
    print(result["table"])
    print("=" * 70)
    print("trace written to:", out_path)
