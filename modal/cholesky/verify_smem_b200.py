"""
A/B the shared-memory staging of Lp in cholesky_gluon_tcgen05_blocked.py
on a real B200, against the gather-only version it replaces.

Locally on GB10 the isolated panel kernel went 91 -> 62 registers, loop 2's
32 shfl.idx per iteration became 8 vectorized ld.shared.v4, and output was
bitwise identical including the short-tail case.

Usage: modal run verify_smem_b200.py
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

app = modal.App("cholesky-verify-smem-b200", image=image)

SHAPES = [
    (1, 8192), (1, 4096), (2, 2048), (16, 512), (64, 256), (1024, 64), (4096, 32),
    (1, 8196), (4, 100), (1, 36),
]

# the gather-only loop 2, as it was before the smem change
OLD_LOOP2 = '''    i_idx = gl.arange(0, BLOCK_I, layout=ROW_LAYOUT)
    i = bk + pid_i * BLOCK_I + i_idx
    row_mask = i < n

    # L21
    L_rows = gl.zeros((BLOCK_I, NB), dtype=gl.float32, layout=TILE_LAYOUT)
    for jp in range(nb_valid):
        jp_idx = gl.zeros([1, NB], gl.int32, layout=TILE_LAYOUT) + jp
        row_jp = gl.sum(gl.gather(Lp, jp_idx, 0), axis=0)
        contrib = gl.sum(L_rows * row_jp[None, :], axis=1)

        a_val = gl.load(
            A_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
            mask=row_mask, other=0.0,
        )
        diff = a_val - contrib
        ljj = gl.sum(gl.where(idx_col == jp, row_jp, 0.0), axis=0)

        is_diag = i == (bk + jp)
        val = gl.where(is_diag, ljj, diff / ljj)
        L_rows = gl.where(col_idx == jp, val[:, None], L_rows)

        gl.store(
            L_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
            val, mask=row_mask,
        )
'''


def _make_orig():
    """Replace the smem loop 2 with the gather version."""
    import re
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert "Lp_smem" in src, "expected the smem-patched working tree"
    start = src.index("    # Stage the finished factor into shared memory")
    end = src.index("@gluon.jit\ndef _round_tf32")
    o = src[:start] + OLD_LOOP2 + "\n\n" + src[end:]
    assert "Lp_smem" not in o and "static_range" not in o, "reverse-patch failed"
    open("/root/python_standalone/_orig_gather.py", "w").write(o)


@app.function(gpu="B200", timeout=3600)
def verify():
    import importlib
    import json
    import statistics
    import sys
    import time

    import torch

    sys.path.insert(0, "/root/python_standalone")
    _make_orig()

    new_mod = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    old_mod = importlib.import_module("_orig_gather")
    import bench_leaderboard

    rows = []
    for batch, n in SHAPES:
        A = bench_leaderboard.generate_input(batch, n, 2, 1234 + n)
        row = {"batch": batch, "n": n}
        for tag, mod in (("old", old_mod), ("new", new_mod)):
            L = mod.custom_kernel(A)
            torch.cuda.synchronize()
            row[f"{tag}_err"] = (L @ L.transpose(-1, -2) - A).abs().amax().item()
            row[f"{tag}_nan"] = int((~torch.isfinite(L)).sum().item())
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
        pk = next(e for e in ks if "panel" in e["name"])
        ktimes[tag]["regs"] = pk["args"].get("registers per thread")
        ktimes[tag]["smem"] = pk["args"].get("shared memory")

    return {"gpu": torch.cuda.get_device_name(0), "rows": rows, "ktimes": ktimes}


@app.local_entrypoint()
def main():
    r = verify.remote()
    print("=" * 92)
    print(r["gpu"], "  old = gather (registers),  new = smem staging")
    print("=" * 92)
    print(f"{'batch':>6} {'n':>6} {'old err':>11} {'new err':>11} {'NaN':>5}"
          f" {'old ms':>9} {'new ms':>9} {'speedup':>9} {'L diff':>9}")
    tot_o = tot_n = 0.0
    import math
    lo, ln = [], []
    for x in r["rows"]:
        sp = x["old_ms"] / x["new_ms"]
        lo.append(x["old_ms"]); ln.append(x["new_ms"])
        print(f"{x['batch']:6} {x['n']:6} {x['old_err']:11.3e} {x['new_err']:11.3e}"
              f" {x['old_nan']+x['new_nan']:5} {x['old_ms']:9.3f} {x['new_ms']:9.3f}"
              f" {sp:8.2f}x {x['max_diff']:9.2e}")
    go = math.exp(sum(math.log(t) for t in lo) / len(lo))
    gn = math.exp(sum(math.log(t) for t in ln) / len(ln))
    print("-" * 92)
    print(f"{'GEOMEAN(10)':>13} {'':>21} {'':>5} {go:9.3f} {gn:9.3f} {go/gn:8.2f}x")
    print()
    print("--- per-kernel, batch=1 n=8192 ---")
    for tag in ("old", "new"):
        t = r["ktimes"][tag]
        print(f"  {tag}: panel {t['panel']['ms']:7.3f} ms x{t['panel']['n']:4}"
              f"   syrk {t['syrk']['ms']:7.3f} ms x{t['syrk']['n']:4}"
              f"   panel regs {t['regs']}  smem {t['smem']}B")
    op, np_ = r["ktimes"]["old"]["panel"]["ms"], r["ktimes"]["new"]["panel"]["ms"]
    print(f"  panel: {op:.3f} -> {np_:.3f} ms  ({100*(np_-op)/op:+.1f}%, {op/np_:.2f}x)")
