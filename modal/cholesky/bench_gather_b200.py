"""
Runs the official 15-case leaderboard suite on a Modal B200 for the
gl.gather-patched cholesky_gluon_tcgen05_blocked.py, alongside the
pre-patch (masked-reduce) version and cuSOLVER, all in ONE container so
the comparison is on identical hardware -- matching the methodology of
leaderboard_benchmark_results.txt.

Re-running the old version rather than only citing the 2026-08-03 numbers
from that file lets us confirm the baseline reproduces before claiming a
speedup against it.

Also captures a Kineto trace at batch=1/n=8192 in the same container.

Usage: modal run bench_gather_b200.py
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

app = modal.App("cholesky-bench-gather-b200", image=image)


def _make_orig():
    """Reverse the gather edit to reconstruct the masked-reduce version."""
    import re
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert "gl.gather" in src, "expected the gather-patched working tree"
    o = src
    o = re.sub(
        r"\n *# row_jp = Lp\[jp, :\]\. gl\.gather.*?\n *jp_idx = gl\.zeros\(\[1, NB\], gl\.int32, layout=TILE_LAYOUT\) \+ jp\n *row_jp = gl\.sum\(gl\.gather\(Lp, jp_idx, 0\), axis=0\) *# \[NB\] COL_LAYOUT\n",
        "\n        row_jp = gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)\n",
        o, flags=re.S)
    o = re.sub(
        r"\n *# Same gather-instead-of-butterfly.*?\n *jp_idx = gl\.zeros\(\[1, NB\], gl\.int32, layout=TILE_LAYOUT\) \+ jp\n *row_jp = gl\.sum\(gl\.gather\(Lp, jp_idx, 0\), axis=0\)\n",
        "\n        row_jp = gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)\n",
        o, flags=re.S)
    assert "gl.gather" not in o, "reverse-patch failed"
    open("/root/python_standalone/_orig_reduce.py", "w").write(o)


@app.function(gpu="B200", timeout=5400)
def bench():
    import importlib
    import json
    import math
    import sys

    import torch
    import triton

    sys.path.insert(0, "/root/python_standalone")
    _make_orig()
    import bench_leaderboard

    MODULES = [
        ("starter", "starter"),
        ("tcgen05(reduce)", "_orig_reduce"),
        ("tcgen05(gather)", "cholesky_gluon_tcgen05_blocked"),
    ]

    results = {}
    for label, modname in MODULES:
        mod = importlib.import_module(modname)
        times, errs = [], []
        for spec in bench_leaderboard.BENCHMARKS:
            t = bench_leaderboard.bench_one(mod.custom_kernel, spec)
            times.append(t * 1e3)
            # accuracy spot-check on the smaller cases only (n^3 matmul is slow)
            if spec["n"] <= 2048:
                A = bench_leaderboard.generate_input(spec["batch"], spec["n"],
                                                     spec["cond"], spec["seed"])
                L = mod.custom_kernel(A)
                torch.cuda.synchronize()
                errs.append((L @ L.transpose(-1, -2) - A).abs().amax().item())
                del A, L
                torch.cuda.empty_cache()
        geo = math.exp(sum(math.log(t) for t in times) / len(times))
        results[label] = {"times": times, "geomean": geo,
                          "max_err": max(errs) if errs else None}

    # Kineto trace of the patched kernel at the headline shape
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
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12,
                                      max_name_column_width=90)
    ev = json.load(open("/tmp/kineto.json"))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel"]
    agg = {}
    for e in ks:
        key = "panel" if "panel" in e["name"] else ("syrk" if "syrk" in e["name"] else "other")
        a = agg.setdefault(key, [0, 0.0])
        a[0] += 1
        a[1] += e["dur"]
    with open("/tmp/kineto.json", "rb") as f:
        trace = f.read()

    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "triton": triton.__version__,
        "specs": [(s["batch"], s["n"]) for s in bench_leaderboard.BENCHMARKS],
        "results": results,
        "kernel_split": {k: {"n": v[0], "ms": v[1] / 1000} for k, v in agg.items()},
        "table": table,
        "trace": trace,
    }


# 2026-08-03 numbers from leaderboard_benchmark_results.txt, tcgen05(batched)
# column -- the same kernel before the gather change.
HISTORIC = [0.268, 0.255, 0.285, 0.467, 0.911, 6.561, 1.788, 3.707,
            3.674, 4.476, 7.781, 8.695, 21.44, 91.29, 573.9]
HISTORIC_STARTER = [0.140, 0.139, 0.203, 0.370, 0.766, 3.936, 1.559, 3.186,
                    3.849, 5.575, 1.544, 12.50, 6.413, 34.22, 220.7]


@app.local_entrypoint()
def main():
    r = bench.remote()
    print("=" * 100)
    print(f"{r['gpu']}  torch {r['torch']}  triton {r['triton']}")
    print("=" * 100)
    st = r["results"]["starter"]["times"]
    old = r["results"]["tcgen05(reduce)"]["times"]
    new = r["results"]["tcgen05(gather)"]["times"]
    print(f"{'batch':>6} {'n':>6} | {'starter':>9} | {'reduce(now)':>11} {'reduce(8/03)':>12} |"
          f" {'gather':>9} | {'speedup':>8} {'vs cuSOLVER':>12}")
    print("-" * 100)
    for (b, n), s, o, nw, h in zip(r["specs"], st, old, new, HISTORIC):
        print(f"{b:6} {n:6} | {s:9.3f} | {o:11.3f} {h:12.3f} | {nw:9.3f} |"
              f" {o/nw:7.2f}x {nw/s:11.2f}x")
    print("-" * 100)
    import math
    gs = r["results"]["starter"]["geomean"]
    go = r["results"]["tcgen05(reduce)"]["geomean"]
    gn = r["results"]["tcgen05(gather)"]["geomean"]
    gh = math.exp(sum(math.log(t) for t in HISTORIC) / len(HISTORIC))
    print(f"{'GEOMEAN':>13} | {gs:9.3f} | {go:11.3f} {gh:12.3f} | {gn:9.3f} |"
          f" {go/gn:7.2f}x {gn/gs:11.2f}x")
    print()
    for label in ("tcgen05(reduce)", "tcgen05(gather)"):
        print(f"  {label}: max |L L^T - A| over n<=2048 cases = {r['results'][label]['max_err']:.3e}")
    print()
    print("--- kernel split, batch=1 n=8192 (gather version) ---")
    for k, v in sorted(r["kernel_split"].items(), key=lambda x: -x[1]["ms"]):
        print(f"  {k:8} {v['ms']:8.3f} ms  x{v['n']}")

    os.makedirs(TRACE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    p = os.path.join(TRACE_DIR, f"cholesky_gluon_tcgen05_blocked__custom_kernel__{stamp}.json")
    with open(p, "wb") as f:
        f.write(r["trace"])
    print("\nkineto trace ->", p)
