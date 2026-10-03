"""
A/B the gl.gather row-extraction change in cholesky_gluon_tcgen05_blocked.py
on a real B200.

The old idiom `gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)` extracted one
row of Lp via a full warp butterfly all-reduce (NB*log2(NB) = 160 shuffles,
31/32 of the inputs being zeros the `where` had just written). gl.gather does
the same extraction with one shfl.idx per element (NB = 32 shuffles).

Locally on GB10 the panel kernel alone went 325 -> 69 shuffles with bitwise
identical output. This measures the end-to-end effect on B200, where the
panel kernel is ~54% of total runtime.

Usage: modal run verify_gather_b200.py
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

app = modal.App("cholesky-verify-gather-b200", image=image)

SHAPES = [
    (1, 8192), (1, 4096), (2, 2048), (16, 512), (64, 256), (1024, 64), (4096, 32),
    (1, 8196), (4, 100), (1, 36),
]


def _make_orig():
    """Reverse the gather edit to reconstruct the masked-reduce version."""
    import re
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert "gl.gather" in src, "expected the gather-patched working tree"
    o = src
    # loop 1
    o = re.sub(
        r"\n *# row_jp = Lp\[jp, :\]\. gl\.gather.*?\n *jp_idx = gl\.zeros\(\[1, NB\], gl\.int32, layout=TILE_LAYOUT\) \+ jp\n *row_jp = gl\.sum\(gl\.gather\(Lp, jp_idx, 0\), axis=0\) *# \[NB\] COL_LAYOUT\n",
        "\n        row_jp = gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)\n",
        o, flags=re.S)
    # loop 2
    o = re.sub(
        r"\n *# Same gather-instead-of-butterfly.*?\n *jp_idx = gl\.zeros\(\[1, NB\], gl\.int32, layout=TILE_LAYOUT\) \+ jp\n *row_jp = gl\.sum\(gl\.gather\(Lp, jp_idx, 0\), axis=0\)\n",
        "\n        row_jp = gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)\n",
        o, flags=re.S)
    assert "gl.gather" not in o, "reverse-patch failed"
    assert o != src
    open("/root/python_standalone/_orig_reduce.py", "w").write(o)


@app.function(gpu="B200", timeout=3600)
def verify():
    import importlib
    import json
    import statistics
    import sys
    import time

    import torch
    import triton

    sys.path.insert(0, "/root/python_standalone")
    _make_orig()

    new_mod = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    old_mod = importlib.import_module("_orig_reduce")
    import bench_leaderboard

    rows = []
    for batch, n in SHAPES:
        A = bench_leaderboard.generate_input(batch, n, 2, 1234 + n)
        row = {"batch": batch, "n": n}
        for tag, mod in (("old", old_mod), ("new", new_mod)):
            L = mod.custom_kernel(A)
            torch.cuda.synchronize()
            row[f"{tag}_err"] = (L @ L.transpose(-1, -2) - A).abs().amax().item()
            ts = []
            for _ in range(5):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                mod.custom_kernel(A)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            row[f"{tag}_ms"] = statistics.median(ts) * 1e3
            del L
        row["max_diff"] = (old_mod.custom_kernel(A) - new_mod.custom_kernel(A)).abs().amax().item()
        del A
        torch.cuda.empty_cache()
        rows.append(row)

    # per-kernel split at the headline shape
    A = bench_leaderboard.generate_input(1, 8192, 2, 48192)
    ktimes = {}
    for tag, mod in (("old", old_mod), ("new", new_mod)):
        mod.custom_kernel(A)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            mod.custom_kernel(A)
            torch.cuda.synchronize()
        prof.export_chrome_trace(f"/tmp/{tag}.json")
        ev = json.load(open(f"/tmp/{tag}.json"))["traceEvents"]
        ks = [e for e in ev if e.get("cat") == "kernel"]
        agg = {}
        for e in ks:
            key = "panel" if "panel" in e["name"] else ("syrk" if "syrk" in e["name"] else "other")
            a = agg.setdefault(key, [0, 0.0])
            a[0] += 1
            a[1] += e["dur"]
        ktimes[tag] = {k: {"n": v[0], "ms": v[1] / 1000} for k, v in agg.items()}
        ktimes[tag]["regs"] = next(e["args"].get("registers per thread")
                                   for e in ks if "panel" in e["name"])

    return {"gpu": torch.cuda.get_device_name(0), "rows": rows, "ktimes": ktimes}


@app.local_entrypoint()
def main():
    r = verify.remote()
    print("=" * 88)
    print(r["gpu"], "  old = masked-reduce butterfly,  new = gl.gather")
    print("=" * 88)
    print(f"{'batch':>6} {'n':>6} {'old err':>11} {'new err':>11} {'old ms':>9} {'new ms':>9} {'speedup':>9} {'L diff':>9}")
    for x in r["rows"]:
        sp = x["old_ms"] / x["new_ms"]
        print(f"{x['batch']:6} {x['n']:6} {x['old_err']:11.3e} {x['new_err']:11.3e}"
              f" {x['old_ms']:9.3f} {x['new_ms']:9.3f} {sp:8.2f}x {x['max_diff']:9.2e}")
    print()
    print("--- per-kernel, batch=1 n=8192 ---")
    for tag in ("old", "new"):
        t = r["ktimes"][tag]
        print(f"  {tag}: panel {t['panel']['ms']:7.3f} ms x{t['panel']['n']:4}"
              f"   syrk {t['syrk']['ms']:7.3f} ms x{t['syrk']['n']:4}   panel regs {t['regs']}")
    op, np_ = r["ktimes"]["old"]["panel"]["ms"], r["ktimes"]["new"]["panel"]["ms"]
    print(f"  panel: {op:.3f} -> {np_:.3f} ms  ({100*(np_-op)/op:+.1f}%, {op/np_:.2f}x)")
