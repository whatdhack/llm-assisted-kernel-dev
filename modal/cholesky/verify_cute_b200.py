"""
Correctness + timing for the CuTe DSL port against the Gluon original.

Both kernels are the same blocked right-looking Cholesky with a tcgen05
tf32x3 SYRK step, so this checks two things:
  1. does the CuTe version reconstruct A to the same accuracy, and
  2. how does it time against the Gluon version and cuSOLVER.

Usage: modal run verify_cute_b200.py
"""
import os as _os

import modal

LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch", "triton==3.7.1", "numpy")
    # PINNED. Modal caches image layers by the definition hash, so an
    # unpinned "nvidia-cutlass-dsl" would keep whatever version was
    # latest when the layer was FIRST built, silently, forever -- and
    # two of this port's bugs were version-sensitive DSL semantics
    # (internal_type=TFloat32 rounding inside the TMA; cute.copy and
    # cute.gemm electing per-warp). Pin it so the number is a fact in
    # the source and the recorded measurements stay reproducible.
    .pip_install("nvidia-cutlass-dsl==4.7.1")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib",
        "CUTE_DSL_ARCH": "sm_100a",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(LOCAL_DIR, "/root/python_standalone")
)

app = modal.App("cholesky-verify-cute-b200", image=image)

def _print_versions(tag=""):
    """Record what actually ran. The image pins nvidia-cutlass-dsl, but pinning
    is only half of it -- printing the version is what makes a saved artifact
    self-describing months later."""
    import subprocess, sys
    import torch
    import cutlass
    v = [f"torch {torch.__version__}", f"cutlass-dsl {cutlass.__version__}"]
    try:
        import triton
        v.append(f"triton {triton.__version__}")
    except ImportError:
        pass
    v.append(f"gpu {torch.cuda.get_device_name(0)}")
    print(("VERSIONS " + tag + ": " if tag else "VERSIONS: ") + ", ".join(v),
          flush=True)



CASES = [
    (1, 64), (4, 64), (2, 100), (1, 128), (3, 200), (2, 256),
    (1, 512), (2, 512), (1, 1024), (1, 2048),
]


@app.function(gpu="B200", timeout=3000)
def verify(run_bench: bool = True):
    import importlib
    import json
    import sys
    import time
    import traceback

    import torch

    sys.path.insert(0, "/root/python_standalone")
    import bench_leaderboard

    _print_versions()
    out = {"env": {}, "cases": [], "bench": {}, "error": None}
    out["env"]["gpu"] = torch.cuda.get_device_name(0)
    out["env"]["torch"] = str(torch.__version__)
    try:
        import cutlass
        out["env"]["cutlass_dsl"] = str(getattr(cutlass, "__version__", "?"))
    except Exception as exc:
        out["error"] = f"import cutlass: {exc}"
        return out

    try:
        cute_mod = importlib.import_module("cholesky_cute_tcgen05_blocked")
        gluon_mod = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    except Exception:
        out["error"] = traceback.format_exc()
        return out

    for batch, n in CASES:
        rec = {"batch": batch, "n": n}
        print("case", batch, n, flush=True)
        try:
            gen = torch.Generator(device="cuda").manual_seed(1000 + n)
            a = torch.randn((batch, n, n), device="cuda", dtype=torch.float32,
                            generator=gen)
            A = (a @ a.transpose(-1, -2)) / n
            A.diagonal(dim1=-2, dim2=-1).add_(1.0)

            Lc = cute_mod.custom_kernel(A)
            torch.cuda.synchronize()
            Lg = gluon_mod.custom_kernel(A)
            torch.cuda.synchronize()
            Lt = torch.linalg.cholesky(A)

            rec["cute_recon"] = (Lc @ Lc.transpose(-1, -2) - A).abs().amax().item()
            rec["gluon_recon"] = (Lg @ Lg.transpose(-1, -2) - A).abs().amax().item()
            rec["torch_recon"] = (Lt @ Lt.transpose(-1, -2) - A).abs().amax().item()
            rec["cute_vs_gluon"] = (Lc - Lg).abs().amax().item()
            rec["cute_nonfinite"] = int((~torch.isfinite(Lc)).sum().item())
            # the factor must stay lower-triangular
            rec["cute_upper"] = Lc.triu(1).abs().amax().item()
            del A, Lc, Lg, Lt, a
            torch.cuda.empty_cache()
        except Exception:
            rec["error"] = traceback.format_exc()
        out["cases"].append(rec)

    if not run_bench:
        print("RESULT_JSON", json.dumps(out), flush=True)
        return json.dumps(out)

    for label, mod in (("cute", cute_mod), ("gluon", gluon_mod),
                       ("cusolver", importlib.import_module("starter"))):
        try:
            times = []
            for spec in bench_leaderboard.BENCHMARKS:
                print("bench", label, spec["batch"], spec["n"], flush=True)
                times.append(bench_leaderboard.bench_one(mod.custom_kernel, spec) * 1e3)
            out["bench"][label] = times
        except Exception:
            out["bench"][label] = traceback.format_exc().splitlines()[-3:]
    print("RESULT_JSON", json.dumps(out), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main():
    import math

    import json
    r = json.loads(verify.remote(True))
    print("=" * 96)
    print(f"{r['env']}")
    print("=" * 96)
    if r["error"]:
        print(r["error"])
        return

    print(f"{'batch':>6} {'n':>6} | {'cute recon':>11} {'gluon recon':>11} "
          f"{'torch recon':>11} | {'|cute-gluon|':>12} {'nonfin':>7} {'upper':>9}")
    print("-" * 96)
    for c in r["cases"]:
        if "error" in c:
            print(f"{c['batch']:6} {c['n']:6} | ERROR")
            print(c["error"])
            continue
        print(f"{c['batch']:6} {c['n']:6} | {c['cute_recon']:11.3e} "
              f"{c['gluon_recon']:11.3e} {c['torch_recon']:11.3e} | "
              f"{c['cute_vs_gluon']:12.3e} {c['cute_nonfinite']:7d} "
              f"{c['cute_upper']:9.1e}")

    print()
    b = r["bench"]
    if all(isinstance(b.get(k), list) and isinstance(b[k][0], float)
           for k in ("cute", "gluon", "cusolver")):
        print(f"{'batch':>6} {'n':>6} | {'cuSOLVER':>9} {'gluon':>9} {'cute':>9} |"
              f" {'cute/gluon':>10}")
        print("-" * 62)
        for spec, s, g, c in zip([(x["batch"], x["n"]) for x in _SPECS],
                                 b["cusolver"], b["gluon"], b["cute"]):
            print(f"{spec[0]:6} {spec[1]:6} | {s:9.3f} {g:9.3f} {c:9.3f} |"
                  f" {c/g:9.2f}x")
        geo = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
        print("-" * 62)
        print(f"{'GEOMEAN':>13} | {geo(b['cusolver']):9.3f} {geo(b['gluon']):9.3f} "
              f"{geo(b['cute']):9.3f} | {geo(b['cute'])/geo(b['gluon']):9.2f}x")
    else:
        for k, v in b.items():
            print(k, v)


_SPECS = [
    {"batch": 4096, "n": 32}, {"batch": 1024, "n": 64}, {"batch": 256, "n": 128},
    {"batch": 64, "n": 256}, {"batch": 16, "n": 512}, {"batch": 640, "n": 512},
    {"batch": 4, "n": 1024}, {"batch": 60, "n": 1024}, {"batch": 2, "n": 2048},
    {"batch": 8, "n": 2048}, {"batch": 1, "n": 4096}, {"batch": 2, "n": 4096},
    {"batch": 1, "n": 8192}, {"batch": 1, "n": 16384}, {"batch": 1, "n": 32768},
]
