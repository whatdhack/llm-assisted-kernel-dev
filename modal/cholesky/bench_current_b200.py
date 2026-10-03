"""
Official 15-case leaderboard suite for the CURRENT kernel, using
bench_leaderboard's own protocol (warmup=2, median of 5, fresh input per
case) so the geomean is directly comparable to the recorded table.

NOT the interleaved harness: that one reuses a warm input with several
modules resident and runs the small shapes faster, so its geomean cannot be
compared across runs -- only its within-run ratios can. This script exists
to produce a number that CAN go in leaderboard_benchmark_results.txt.

starter is run alongside as a control: it is unchanged code, so if its
geomean matches earlier runs the harness is consistent.

Also saves a Kineto trace of the current kernel at batch=1/n=8192.

Usage: modal run bench_current_b200.py
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

app = modal.App("cholesky-bench-current-b200", image=image)


@app.function(gpu="B200", timeout=7200)
def bench():
    import importlib
    import json
    import math
    import sys

    import torch
    import triton

    sys.path.insert(0, "/root/python_standalone")
    import bench_leaderboard

    MODULES = [("starter", "starter"),
               ("current", "cholesky_gluon_tcgen05_blocked")]

    results = {}
    for label, modname in MODULES:
        mod = importlib.import_module(modname)
        times, errs, nans = [], [], 0
        for spec in bench_leaderboard.BENCHMARKS:
            times.append(bench_leaderboard.bench_one(mod.custom_kernel, spec) * 1e3)
            if spec["n"] <= 2048:
                A = bench_leaderboard.generate_input(spec["batch"], spec["n"],
                                                     spec["cond"], spec["seed"])
                L = mod.custom_kernel(A)
                torch.cuda.synchronize()
                errs.append((L @ L.transpose(-1, -2) - A).abs().amax().item())
                nans += int((~torch.isfinite(L)).sum().item())
                del A, L
                torch.cuda.empty_cache()
        results[label] = {
            "times": times,
            "geomean": math.exp(sum(math.log(t) for t in times) / len(times)),
            "max_err": max(errs) if errs else None,
            "nans": nans,
        }

    mod = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    A = bench_leaderboard.generate_input(1, 8192, 2, 48192)
    mod.custom_kernel(A)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        mod.custom_kernel(A)
        torch.cuda.synchronize()
    prof.export_chrome_trace("/tmp/kineto.json")
    ev = json.load(open("/tmp/kineto.json"))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel"]
    agg = {}
    for e in ks:
        k = "panel" if "panel" in e["name"] else ("syrk" if "syrk" in e["name"] else "other")
        a = agg.setdefault(k, [0, 0.0])
        a[0] += 1
        a[1] += e["dur"]
    sk = next(e for e in ks if "syrk" in e["name"])
    with open("/tmp/kineto.json", "rb") as f:
        trace = f.read()

    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "triton": triton.__version__,
        "specs": [(s["batch"], s["n"]) for s in bench_leaderboard.BENCHMARKS],
        "results": results,
        "split": {k: {"n": v[0], "ms": v[1] / 1000} for k, v in agg.items()},
        "syrk_smem": sk["args"].get("shared memory"),
        "syrk_regs": sk["args"].get("registers per thread"),
        "trace": trace,
    }


# recorded columns from leaderboard_benchmark_results.txt, same 15 cases
HIST_REDUCE = [0.268, 0.255, 0.285, 0.467, 0.911, 6.561, 1.788, 3.707,
               3.674, 4.476, 7.781, 8.695, 21.44, 91.29, 573.9]      # tcgen05(batched), 08-03
HIST_SMEM = [0.126, 0.133, 0.183, 0.342, 0.666, 3.596, 1.291, 2.283,
             2.551, 3.223, 5.008, 5.294, 14.645, 78.025, 542.858]    # smem era, 08-09


@app.local_entrypoint()
def main():
    import math
    r = bench.remote()
    st = r["results"]["starter"]["times"]
    cu = r["results"]["current"]["times"]
    print("=" * 104)
    print(f"{r['gpu']}  torch {r['torch']}  triton {r['triton']}"
          "   -- bench_leaderboard protocol (median of 5, warmup 2)")
    print("=" * 104)
    print(f"{'batch':>6} {'n':>6} | {'starter':>9} {'08-03':>9} {'08-09':>9} {'current':>9} |"
          f" {'vs 08-09':>9} {'vs cuSOLVER':>12}")
    print("-" * 104)
    for (b, n), s, h0, h1, c in zip(r["specs"], st, HIST_REDUCE, HIST_SMEM, cu):
        mark = "*" if c <= s else " "
        print(f"{b:6} {n:6} | {s:9.3f} {h0:9.3f} {h1:9.3f} {c:8.3f}{mark} |"
              f" {h1/c:8.2f}x {c/s:11.2f}x")
    g = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    gs, gc = r["results"]["starter"]["geomean"], r["results"]["current"]["geomean"]
    print("-" * 104)
    print(f"{'GEOMEAN':>13} | {gs:9.3f} {g(HIST_REDUCE):9.3f} {g(HIST_SMEM):9.3f} {gc:9.3f} |"
          f" {g(HIST_SMEM)/gc:8.2f}x {gc/gs:11.2f}x")
    print()
    wins = sum(1 for s, c in zip(st, cu) if c <= s)
    print(f"  cases won outright vs cuSOLVER: {wins}/15")
    print(f"  max |L L^T - A| (n<=2048): {r['results']['current']['max_err']:.3e}"
          f"   nonfinite: {r['results']['current']['nans']}")
    print(f"  starter geomean this run {gs:.3f} (earlier runs 2.383 / 2.396 / 2.483"
          " -- harness consistency check)")
    print()
    print(f"--- kernel split, batch=1 n=8192 --- syrk smem {r['syrk_smem']}B regs {r['syrk_regs']}")
    for k, v in sorted(r["split"].items(), key=lambda x: -x[1]["ms"]):
        print(f"  {k:8} {v['ms']:8.3f} ms  x{v['n']}")

    os.makedirs(TRACE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    p = os.path.join(TRACE_DIR, f"cholesky_gluon_tcgen05_blocked__custom_kernel__{stamp}.json")
    with open(p, "wb") as f:
        f.write(r["trace"])
    print("\nkineto ->", p)
