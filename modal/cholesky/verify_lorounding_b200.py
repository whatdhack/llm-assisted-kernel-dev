"""
A/B removing _round_tf32 from the tf32x3 "lo" path.

  now : l_m_lo = _round_tf32(l_m_reg - l_m_hi)
  try : l_m_lo =              l_m_reg - l_m_hi

`hi` MUST stay rounded -- `lo` is defined as x - hi, so hi has to be exactly
representable in tf32 or the split is inconsistent. But `lo` is handed
straight to a tf32 MMA, which rounds it anyway; the explicit rounding only
controls the rounding MODE (round-to-nearest via the +0x1000, vs whatever
tcgen05 does on input). So this trades an IADD3+LOP3 per element on both lo
paths (~2% of syrk) against some accuracy.

This is an ACCURACY experiment first and a speed experiment second, so the
error column matters more than the timing column.

Working tree keeps the rounded version; the variant is forward-patched here.

Usage: modal run verify_lorounding_b200.py
"""
import modal

import os as _os
# Kernel sources live one level up, in the cholesky_py problem directory.
LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton")
    .add_local_dir(LOCAL_DIR, "/root/python_standalone")
)

app = modal.App("cholesky-verify-lorounding-b200", image=image)

REPS = 9


def _write_variant():
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert src.count("_round_tf32(l_m_reg - l_m_hi)") == 1
    assert src.count("_round_tf32(l_n_reg - l_n_hi)") == 1
    o = src.replace("_round_tf32(l_m_reg - l_m_hi)", "(l_m_reg - l_m_hi)")
    o = o.replace("_round_tf32(l_n_reg - l_n_hi)", "(l_n_reg - l_n_hi)")
    assert o != src
    open("/root/python_standalone/_v_noloround.py", "w").write(o)


@app.function(gpu="B200", timeout=7200)
def bench():
    import importlib
    import json
    import math
    import statistics
    import sys
    import time

    import torch

    sys.path.insert(0, "/root/python_standalone")
    _write_variant()

    old = importlib.import_module("cholesky_gluon_tcgen05_blocked")   # rounded lo
    new = importlib.import_module("_v_noloround")                      # unrounded lo
    import bench_leaderboard

    rows = []
    for spec in bench_leaderboard.BENCHMARKS:
        A = bench_leaderboard.generate_input(spec["batch"], spec["n"], spec["cond"], spec["seed"])
        for m in (old, new):
            for _ in range(2):
                m.custom_kernel(A)
        torch.cuda.synchronize()
        to, tn = [], []
        for _ in range(REPS):
            for m, acc in ((old, to), (new, tn)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                m.custom_kernel(A)
                torch.cuda.synchronize()
                acc.append((time.perf_counter() - t0) * 1e3)
        Lo, Ln = old.custom_kernel(A), new.custom_kernel(A)
        torch.cuda.synchronize()
        rows.append({
            "batch": spec["batch"], "n": spec["n"],
            "old": statistics.median(to), "new": statistics.median(tn),
            "old_err": (Lo @ Lo.transpose(-1, -2) - A).abs().amax().item(),
            "new_err": (Ln @ Ln.transpose(-1, -2) - A).abs().amax().item(),
            "diff": (Lo - Ln).abs().amax().item(),
            "nan": int((~torch.isfinite(Ln)).sum().item()),
        })
        del Lo, Ln, A
        torch.cuda.empty_cache()

    A = bench_leaderboard.generate_input(1, 8192, 2, 48192)
    k = {}
    for tag, m in (("old", old), ("new", new)):
        m.custom_kernel(A)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            m.custom_kernel(A)
            torch.cuda.synchronize()
        prof.export_chrome_trace(f"/tmp/{tag}.json")
        ev = json.load(open(f"/tmp/{tag}.json"))["traceEvents"]
        ks = [e for e in ev if e.get("cat") == "kernel"]
        agg = {}
        for e in ks:
            key = "panel" if "panel" in e["name"] else ("syrk" if "syrk" in e["name"] else "other")
            a = agg.setdefault(key, [0, 0.0])
            a[1] += e["dur"]
        sk = next(e for e in ks if "syrk" in e["name"])
        k[tag] = {kk: v[1] / 1000 for kk, v in agg.items()}
        k[tag]["regs"] = sk["args"].get("registers per thread")
    return {"gpu": torch.cuda.get_device_name(0), "rows": rows, "k": k}


@app.local_entrypoint()
def main():
    import math
    r = bench.remote()
    print("=" * 104)
    print(r["gpu"], "  old = _round_tf32 on lo   new = raw (x - hi), MMA rounds it")
    print("=" * 104)
    print(f"{'batch':>6} {'n':>6} | {'old err':>11} {'new err':>11} {'err ratio':>10} |"
          f" {'old ms':>9} {'new ms':>9} {'speedup':>8}")
    lo, ln = [], []
    worst = 1.0
    for x in r["rows"]:
        lo.append(x["old"]); ln.append(x["new"])
        ratio = x["new_err"] / x["old_err"] if x["old_err"] else float("nan")
        worst = max(worst, ratio)
        print(f"{x['batch']:6} {x['n']:6} | {x['old_err']:11.3e} {x['new_err']:11.3e}"
              f" {ratio:9.2f}x | {x['old']:9.3f} {x['new']:9.3f} {x['old']/x['new']:7.3f}x")
    g = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 104)
    print(f"{'GEOMEAN':>13} | {'':>23} {worst:9.2f}x | {g(lo):9.3f} {g(ln):9.3f} {g(lo)/g(ln):7.3f}x")
    print(f"\n  worst accuracy degradation: {worst:.2f}x")
    for tag in ("old", "new"):
        t = r["k"][tag]
        print(f"  {tag}: panel {t['panel']:7.3f} ms  syrk {t['syrk']:7.3f} ms  regs {t['regs']}")
    o, n = r["k"]["old"]["syrk"], r["k"]["new"]["syrk"]
    print(f"  syrk: {o:.3f} -> {n:.3f} ms ({100*(n-o)/o:+.2f}%)")
    print("  NaNs:", sum(x["nan"] for x in r["rows"]))
