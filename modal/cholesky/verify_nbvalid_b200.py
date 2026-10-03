"""
A/B the nb_valid removal in cholesky_gluon_tcgen05_blocked.py on a real B200.

nb_valid was a gl.constexpr, so dropping it turns both panel loops' trip
count from a compile-time constant into a runtime value. That can change
codegen, and the panel kernel is 54% of this factorization's runtime, so
the change needs measuring, not assuming.

Compares the working-tree version against a pristine copy of the original
(reconstructed here) for correctness and per-kernel time, including shapes
where n is NOT a multiple of NB=32 -- the only case where nb_valid != NB
and therefore the only case the old code's tail path ever exercised.

Usage: modal run verify_nbvalid_b200.py
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

app = modal.App("cholesky-verify-nbvalid-b200", image=image)

# Shapes: leaderboard-style (n % 32 == 0) plus tail shapes (n % 32 != 0, the
# only case where nb_valid != NB). Tail shapes must still keep n % 4 == 0:
# the SYRK path's tma.make_tensor_descriptor requires a 16-byte-multiple row
# stride, so odd/unaligned n cannot run this kernel at all (pre-existing).
SHAPES = [
    (1, 8192), (1, 4096), (2, 2048), (64, 256),
    (1, 8196), (1, 4100), (2, 2052), (64, 260), (4, 100), (1, 36),
]


@app.function(gpu="B200", timeout=3600)
def verify():
    import importlib
    import json
    import re
    import statistics
    import sys
    import time

    import torch
    import triton

    sys.path.insert(0, "/root/python_standalone")

    # Rebuild the ORIGINAL (pre-change) module from the working-tree source by
    # reversing the edit, so both variants come from the same file and differ
    # only in the nb_valid handling.
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    orig = src
    orig = orig.replace(
        "    NB: gl.constexpr,\n    BLOCK_I: gl.constexpr,\n):",
        "    NB: gl.constexpr,\n    nb_valid: gl.constexpr,\n    BLOCK_I: gl.constexpr,\n):", 1)
    orig = re.sub(r"\n *# Width of this panel.*?\n *nb_valid = gl\.minimum\(NB, n - bk\)\n",
                  "\n", orig, flags=re.S)
    orig = orig.replace(
        "    for bk in range(0, n, NB):\n        grid_panel",
        "    for bk in range(0, n, NB):\n        nb_valid = min(NB, n - bk)\n        grid_panel", 1)
    orig = orig.replace("            NB=NB, BLOCK_I=BLOCK_I,", "            NB=NB, nb_valid=nb_valid, BLOCK_I=BLOCK_I,", 1)
    orig = re.sub(r"\n *# Only a full NB-wide panel.*?\n *trailing = n - bk - NB\n *if trailing > 0:\n",
                  "\n        trailing = n - (bk + nb_valid)\n        if trailing > 0:\n"
                  "            assert nb_valid == NB\n", orig, flags=re.S)
    orig = orig.replace("            start = bk + NB", "            start = bk + nb_valid", 1)
    assert "nb_valid: gl.constexpr" in orig and "gl.minimum" not in orig, "reverse-patch failed"
    assert orig != src
    with open("/root/python_standalone/_orig_blocked.py", "w") as f:
        f.write(orig)

    new_mod = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    old_mod = importlib.import_module("_orig_blocked")
    import bench_leaderboard

    results = []
    for batch, n in SHAPES:
        A = bench_leaderboard.generate_input(batch, n, 2, 1234 + n)
        row = {"batch": batch, "n": n, "tail": n % 32 != 0}
        for tag, mod in (("old", old_mod), ("new", new_mod)):
            L = mod.custom_kernel(A)
            torch.cuda.synchronize()
            row[f"{tag}_err"] = (L @ L.transpose(-1, -2) - A).abs().amax().item()
            # lower triangle only: L must be lower-triangular
            row[f"{tag}_triu"] = L.triu(1).abs().amax().item()
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
        results.append(row)

    # per-kernel timing at the headline shape, from a Kineto trace
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
            k = e["name"].split("<")[0]
            k = "panel" if "panel" in k else ("syrk" if "syrk" in k else "other")
            a = agg.setdefault(k, [0, 0.0])
            a[0] += 1
            a[1] += e["dur"]
        ktimes[tag] = {k: {"n": v[0], "ms": v[1] / 1000} for k, v in agg.items()}
        ktimes[tag]["regs_panel"] = next(
            e["args"].get("registers per thread") for e in ks if "panel" in e["name"])

    return {"gpu": torch.cuda.get_device_name(0), "rows": results, "ktimes": ktimes}


@app.function(gpu="B200", timeout=900)
def check_unaligned_is_preexisting(n: int = 8191):
    """Runs ONLY the original (pre-change) kernel at an unaligned n, in its own
    container, to confirm the misaligned-address failure predates this edit."""
    import sys, torch
    sys.path.insert(0, "/root/python_standalone")
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert "gl.minimum" in src, "expected the edited working-tree source"
    import importlib, re
    orig = src
    orig = orig.replace("    NB: gl.constexpr,\n    BLOCK_I: gl.constexpr,\n):",
                        "    NB: gl.constexpr,\n    nb_valid: gl.constexpr,\n    BLOCK_I: gl.constexpr,\n):", 1)
    orig = re.sub(r"\n *# Width of this panel.*?\n *nb_valid = gl\.minimum\(NB, n - bk\)\n", "\n", orig, flags=re.S)
    orig = orig.replace("    for bk in range(0, n, NB):\n        grid_panel",
                        "    for bk in range(0, n, NB):\n        nb_valid = min(NB, n - bk)\n        grid_panel", 1)
    orig = orig.replace("            NB=NB, BLOCK_I=BLOCK_I,", "            NB=NB, nb_valid=nb_valid, BLOCK_I=BLOCK_I,", 1)
    orig = re.sub(r"\n *# Only a full NB-wide panel.*?\n *trailing = n - bk - NB\n *if trailing > 0:\n",
                  "\n        trailing = n - (bk + nb_valid)\n        if trailing > 0:\n", orig, flags=re.S)
    orig = orig.replace("            start = bk + NB", "            start = bk + nb_valid", 1)
    open("/root/python_standalone/_orig_only.py", "w").write(orig)
    old = importlib.import_module("_orig_only")
    import bench_leaderboard
    A = bench_leaderboard.generate_input(1, n, 2, 7)
    try:
        old.custom_kernel(A)
        torch.cuda.synchronize()
        return f"ORIGINAL kernel SUCCEEDED at n={n}"
    except Exception as e:
        return f"ORIGINAL kernel FAILED at n={n}: {type(e).__name__}: {str(e).splitlines()[0]}"


@app.local_entrypoint()
def main():
    r = verify.remote()
    print("=" * 92)
    print(r["gpu"])
    print("=" * 92)
    print(f"{'batch':>6} {'n':>6} {'tail':>5} {'old err':>11} {'new err':>11} {'triu':>8}"
          f" {'old ms':>9} {'new ms':>9} {'delta':>8} {'L diff':>9}")
    for x in r["rows"]:
        d = 100 * (x["new_ms"] - x["old_ms"]) / x["old_ms"]
        print(f"{x['batch']:6} {x['n']:6} {str(x['tail']):>5} {x['old_err']:11.3e}"
              f" {x['new_err']:11.3e} {max(x['old_triu'], x['new_triu']):8.1e}"
              f" {x['old_ms']:9.3f} {x['new_ms']:9.3f} {d:+7.2f}% {x['max_diff']:9.2e}")
    print()
    print("--- per-kernel, batch=1 n=8192 ---")
    for tag in ("old", "new"):
        t = r["ktimes"][tag]
        print(f"  {tag}: panel {t['panel']['ms']:6.3f} ms x{t['panel']['n']:4}"
              f"   syrk {t['syrk']['ms']:6.3f} ms x{t['syrk']['n']:4}"
              f"   panel regs {t['regs_panel']}")
    op, np_ = r["ktimes"]["old"]["panel"]["ms"], r["ktimes"]["new"]["panel"]["ms"]
    print(f"  panel delta: {100*(np_-op)/op:+.2f}%")
    print()
    print("--- is the unaligned-n failure pre-existing? ---")
    print(" ", check_unaligned_is_preexisting.remote(8191))
