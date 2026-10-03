"""
Does explicitly reusing l_m_smem/l_n_smem as the `hi` buffers save anything,
or is triton's shared-memory allocator already aliasing them?

l_m_smem is dead once `l_m_reg = l_m_smem.load(REG_LAYOUT)` completes, so
storing l_m_hi back into it is legal. And there is no cross-warp WAR hazard:
under REG_LAYOUT the [64,32] tile splits along dim 0, so warp w owns rows
[16w,16w+16) exclusively -- each warp rewrites only rows it read itself.

Nominal footprint says the allocator already recovers ~33KB (81,936 -> 49,176),
and 49,152 = 16,384(a) + 4x8,192(hi/lo) exactly, which is what you get if
l_m_smem/l_n_smem are ALREADY aliased onto two of the hi/lo tiles and
out_smem onto a_smem. If so this change is a no-op.

Usage: modal run verify_inplace_hi_b200.py
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

app = modal.App("cholesky-inplace-hi-b200", image=image)
REPS = 9

OLD = """    l_m_hi_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_m_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_n_hi_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_n_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_m_hi_smem.store(l_m_hi)
    l_m_lo_smem.store(l_m_lo)
    l_n_hi_smem.store(l_n_hi)
    l_n_lo_smem.store(l_n_lo)"""

NEW = """    # reuse the source tiles as the `hi` buffers: l_m_smem/l_n_smem are dead
    # once their contents are in registers, and each warp rewrites only the
    # rows it read itself (REG_LAYOUT splits dim 0 by warp), so no barrier.
    l_m_hi_smem = l_m_smem
    l_n_hi_smem = l_n_smem
    l_m_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_n_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_m_hi_smem.store(l_m_hi)
    l_m_lo_smem.store(l_m_lo)
    l_n_hi_smem.store(l_n_hi)
    l_n_lo_smem.store(l_n_lo)"""


def _write_variant():
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert OLD in src, "staging block not found"
    open("/root/python_standalone/_v_inplace.py", "w").write(src.replace(OLD, NEW, 1))


@app.function(gpu="B200", timeout=5400)
def bench():
    import importlib, json, statistics, sys, time, traceback
    import torch
    sys.path.insert(0, "/root/python_standalone")
    _write_variant()
    old = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    try:
        new = importlib.import_module("_v_inplace")
    except Exception:
        return {"fail": traceback.format_exc()[-3000:]}

    import bench_leaderboard
    rows = []
    for spec in bench_leaderboard.BENCHMARKS:
        A = bench_leaderboard.generate_input(spec["batch"], spec["n"], spec["cond"], spec["seed"])
        try:
            for m in (old, new):
                for _ in range(2):
                    m.custom_kernel(A)
            torch.cuda.synchronize()
        except Exception:
            return {"fail": f"{spec}\n" + traceback.format_exc()[-2500:]}
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
        rows.append({"batch": spec["batch"], "n": spec["n"],
                     "old": statistics.median(to), "new": statistics.median(tn),
                     "diff": (Lo - Ln).abs().amax().item(),
                     "err": (Ln @ Ln.transpose(-1, -2) - A).abs().amax().item(),
                     "nan": int((~torch.isfinite(Ln)).sum().item())})
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
            key = "panel" if "panel" in e["name"] else ("syrk" if "syrk" in e["name"] else "o")
            agg.setdefault(key, 0.0); agg[key] += e["dur"]
        sk = next(e for e in ks if "syrk" in e["name"])
        k[tag] = {kk: v / 1000 for kk, v in agg.items()}
        k[tag]["smem"] = sk["args"].get("shared memory")
        k[tag]["regs"] = sk["args"].get("registers per thread")
    return {"gpu": torch.cuda.get_device_name(0), "rows": rows, "k": k}


@app.local_entrypoint()
def main():
    import math
    r = bench.remote()
    if "fail" in r:
        print("VARIANT FAILED:\n", r["fail"]); return
    print("=" * 84)
    print(r["gpu"], "  old = 4 hi/lo buffers   new = hi written back into the source tile")
    print("=" * 84)
    lo, ln = [], []
    for x in r["rows"]:
        lo.append(x["old"]); ln.append(x["new"])
        print(f"{x['batch']:6} {x['n']:6} | {x['old']:9.3f} {x['new']:9.3f} "
              f"{x['old']/x['new']:7.3f}x | L diff {x['diff']:9.2e}")
    g = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 84)
    print(f"{'GEOMEAN':>13} | {g(lo):9.3f} {g(ln):9.3f} {g(lo)/g(ln):7.3f}x")
    print()
    for tag in ("old", "new"):
        t = r["k"][tag]
        print(f"  {tag}: panel {t['panel']:7.3f}  syrk {t['syrk']:7.3f} ms  "
              f"smem {t['smem']}B  regs {t['regs']}")
    print("  NaNs:", sum(x["nan"] for x in r["rows"]))
