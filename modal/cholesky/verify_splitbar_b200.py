"""
A/B the split-arrival-barrier change in _syrk_tile_tcgen05 on a real B200.

old: one mbarrier over all three TMA loads (l_m 8KB + l_n 8KB + a 16KB), so
     the kernel blocks on the LARGEST transfer before starting work that does
     not depend on it. ncu: 19.4% + 5.2% of samples on that single wait.
new: two mbarriers. Wait on the L tiles, do the hi/lo split and all three
     MMAs, and only then wait for A -- which is not read until the epilogue.

Interleaved per shape (old, new, old, new, ...) on a warmed input: the
earlier lesson was that separate passes cannot resolve sub-1% effects,
because drift on the small shapes exceeds 100%.

Usage: modal run verify_splitbar_b200.py
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

app = modal.App("cholesky-verify-splitbar-b200", image=image)

REPS = 9

# the single-barrier form, for reconstructing the pre-change kernel
OLD_LOADS = '''    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    l_m_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_n_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)

    mbarrier.expect(bar, 2 * l_desc.block_type.nbytes + a_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(l_desc, [off_m, bk], bar, l_m_smem)
    tma.async_copy_global_to_shared(l_desc, [off_n, bk], bar, l_n_smem)
    tma.async_copy_global_to_shared(a_desc, [off_m, off_n], bar, a_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
'''


def _write_old():
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert "bar_a" in src, "expected the split-barrier working tree"
    s = src.index("    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())")
    e = src.index("    # tf32x3: split each operand")
    o = src[:s] + OLD_LOADS + "\n" + src[e:]
    # drop the deferred A wait
    o = o.replace(
        "    # Only NOW is the A tile needed -- it has had the whole hi/lo split plus all\n"
        "    # three MMAs to arrive.\n"
        "    mbarrier.wait(bar_a, phase=0)\n"
        "    mbarrier.invalidate(bar_a)\n\n", "")
    assert "bar_a" not in o, "reverse-patch failed"
    open("/root/python_standalone/_v_onebar.py", "w").write(o)


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
    _write_old()

    new = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    old = importlib.import_module("_v_onebar")
    import bench_leaderboard

    rows = []
    for spec in bench_leaderboard.BENCHMARKS:
        A = bench_leaderboard.generate_input(spec["batch"], spec["n"], spec["cond"], spec["seed"])
        for m in (old, new):
            for _ in range(2):
                m.custom_kernel(A)
        torch.cuda.synchronize()
        to, tn = [], []
        for _ in range(REPS):                       # interleaved
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
            "old_sp": (max(to) - min(to)) / statistics.median(to) * 100,
            "new_sp": (max(tn) - min(tn)) / statistics.median(tn) * 100,
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
            a[0] += 1
            a[1] += e["dur"]
        sk = next(e for e in ks if "syrk" in e["name"])
        k[tag] = {kk: v[1] / 1000 for kk, v in agg.items()}
        k[tag]["smem"] = sk["args"].get("shared memory")
        k[tag]["regs"] = sk["args"].get("registers per thread")
    return {"gpu": torch.cuda.get_device_name(0), "rows": rows, "k": k}


@app.local_entrypoint()
def main():
    import math
    r = bench.remote()
    print("=" * 100)
    print(r["gpu"], "  old = one barrier for all 3 TMA loads   new = split L / A barriers")
    print("=" * 100)
    print(f"{'batch':>6} {'n':>6} | {'old ms':>9} {'new ms':>9} {'speedup':>8} |"
          f" {'old sp':>7} {'new sp':>7} | {'new err':>10} {'L diff':>9}")
    lo, ln = [], []
    for x in r["rows"]:
        lo.append(x["old"]); ln.append(x["new"])
        print(f"{x['batch']:6} {x['n']:6} | {x['old']:9.3f} {x['new']:9.3f} {x['old']/x['new']:7.3f}x |"
              f" {x['old_sp']:6.1f}% {x['new_sp']:6.1f}% | {x['new_err']:10.2e} {x['diff']:9.2e}")
    g = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 100)
    print(f"{'GEOMEAN':>13} | {g(lo):9.3f} {g(ln):9.3f} {g(lo)/g(ln):7.3f}x")
    print()
    for tag in ("old", "new"):
        t = r["k"][tag]
        print(f"  {tag}: panel {t['panel']:7.3f} ms  syrk {t['syrk']:7.3f} ms  "
              f"smem {t['smem']}B  regs {t['regs']}")
    o, n = r["k"]["old"]["syrk"], r["k"]["new"]["syrk"]
    print(f"  syrk: {o:.3f} -> {n:.3f} ms ({100*(n-o)/o:+.2f}%)")
    print("  NaNs:", sum(x["nan"] for x in r["rows"]))
