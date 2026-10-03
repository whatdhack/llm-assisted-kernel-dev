"""
A/B of the SYRK A-tile shared-memory layout. The epilogue's register layout is
fixed by the TMEM load, so the only lever on its bank-conflict rate is where
sA puts the elements. Each variant also changes how many TMA boxes the A tile
needs (128B is TMA's widest swizzle, so a 64-wide fp32 tile is never one box).

Usage: modal run ab_alayout_b200.py
"""
import os as _os

import modal

LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch", "triton", "numpy")
    .pip_install("nvidia-cutlass-dsl")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib",
        "CUTE_DSL_ARCH": "sm_100a",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(LOCAL_DIR, "/root/python_standalone")
)

app = modal.App("cholesky-ab-alayout-b200")

MODULES = [
    ("base/SW128", "cholesky_cute_tcgen05_blocked"),
    ("K_SW64",     "cholesky_cute_Aksw64"),
    ("K_SW32",     "cholesky_cute_Aksw32"),
    ("K_INTER",    "cholesky_cute_Akinter"),
    ("MN_SW128",   "cholesky_cute_Amnsw128"),
    ("SW128 r10",  "cholesky_cute_Aksw128r"),
    ("gluon",      "cholesky_gluon_tcgen05_blocked"),
]


@app.function(gpu="B200", image=image, timeout=3600)
def bench():
    import importlib
    import json
    import math
    import sys
    import traceback

    import torch

    sys.path.insert(0, "/root/python_standalone")
    import bench_leaderboard

    out = {}
    for label, name in MODULES:
        try:
            mod = importlib.import_module(name)
            times = []
            for spec in bench_leaderboard.BENCHMARKS:
                print("bench", label, spec["batch"], spec["n"], flush=True)
                times.append(bench_leaderboard.bench_one(mod.custom_kernel, spec) * 1e3)
            A = bench_leaderboard.generate_input(1, 2048, 2, 44048)
            L = mod.custom_kernel(A)
            torch.cuda.synchronize()
            out[label] = {
                "times": times,
                "geomean": math.exp(sum(math.log(t) for t in times) / len(times)),
                "recon": (L @ L.transpose(-1, -2) - A).abs().amax().item(),
                "nonfinite": int((~torch.isfinite(L)).sum().item()),
                "n8192": times[12],
            }
            del A, L
            torch.cuda.empty_cache()
        except Exception:
            out[label] = {"error": traceback.format_exc().splitlines()[-4:]}
    print("RESULT_JSON", json.dumps(out), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main():
    import json

    r = json.loads(bench.remote())
    print(f"\n{'variant':<14} {'geomean':>9} {'vs base':>8} {'n=8192':>9} "
          f"{'recon':>10} {'nonfin':>7}")
    print("-" * 62)
    base = r.get("base/SW128", {}).get("geomean")
    for label, _ in MODULES:
        d = r.get(label, {})
        if "error" in d:
            print(f"{label:<14} ERROR {d['error']}")
            continue
        rel = f"{d['geomean']/base:8.3f}" if base else "       -"
        print(f"{label:<14} {d['geomean']:9.3f} {rel} {d['n8192']:9.3f} "
              f"{d['recon']:10.3e} {d['nonfinite']:7d}")
