"""
A/B replacing out_smem with a_smem in the SYRK epilogue.

a_smem is dead once `a_reg = a_smem.load(acc_reg_layout)` completes, and the
epilogue is a read-modify-write of the same global tile, so the same 16KB can
serve both directions.

Expected no-op: the byte accounting says triton already aliases out_smem onto
a_smem (49,152 = 16,384 + 4x8,192 accounts for every byte with only five
distinct buffers), and the analogous l_m_hi_smem = l_m_smem test came back
byte-identical.

The one thing worth MEASURING rather than assuming: a_smem is now read by all
warps and then rewritten by all warps. Whether that is a WAR hazard depends on
acc_reg_layout distributing the [64,64] tile disjointly across the 4 warps. If
it does not, triton must insert a barrier -- and barriers are ~27% of this
kernel, so an extra one would show up.

Usage: modal run verify_outsmem_b200.py
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

app = modal.App("cholesky-outsmem-b200", image=image)
REPS = 9

# undo the whole "minimal names" cleanup, so the A/B is cleanup vs no-cleanup
SUBS = [
 ("""    l_m_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_n_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_m_smem.store(l_m_hi)
    l_m_lo_smem.store(l_m_lo)
    l_n_smem.store(l_n_hi)
    l_n_lo_smem.store(l_n_lo)""",
  """    l_m_hi_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_m_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_n_hi_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_n_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_m_hi_smem.store(l_m_hi)
    l_m_lo_smem.store(l_m_lo)
    l_n_hi_smem.store(l_n_hi)
    l_n_lo_smem.store(l_n_lo)"""),
 ("tcgen05_mma(l_m_smem, l_n_smem.permute((1, 0)), acc0",
  "tcgen05_mma(l_m_hi_smem, l_n_hi_smem.permute((1, 0)), acc0"),
 ("tcgen05_mma(l_m_smem, l_n_lo_smem.permute((1, 0)), acc1",
  "tcgen05_mma(l_m_hi_smem, l_n_lo_smem.permute((1, 0)), acc1"),
 ("tcgen05_mma(l_m_lo_smem, l_n_smem.permute((1, 0)), acc2",
  "tcgen05_mma(l_m_lo_smem, l_n_hi_smem.permute((1, 0)), acc2"),
 ("""    result = a_smem.load(acc_reg_layout) - update""",
  """    a_reg = a_smem.load(acc_reg_layout)
    result = a_reg - update"""),
 ("""    a_smem.store(result)
    fence_async_shared()
    tma.async_copy_shared_to_global(a_desc, [off_m, off_n], a_smem)""",
  """    out_smem = gl.allocate_shared_memory(gl.float32, a_desc.block_type.shape, a_desc.layout)
    out_smem.store(result)
    fence_async_shared()
    tma.async_copy_shared_to_global(a_desc, [off_m, off_n], out_smem)"""),
 ("    i = bk + pid_i * BLOCK_I + gl.arange(0, BLOCK_I, layout=ROW_LAYOUT)",
  """    i_idx = gl.arange(0, BLOCK_I, layout=ROW_LAYOUT)
    i = bk + pid_i * BLOCK_I + i_idx"""),
]
NEW = OLD = None


def _write_old():
    o = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    for new, old in SUBS:
        assert new in o, f"missing: {new[:60]}"
        o = o.replace(new, old, 1)
    assert "out_smem = gl.allocate" in o and "l_m_hi_smem = gl.allocate" in o and "i_idx" in o
    open("/root/python_standalone/_v_outsmem.py", "w").write(o)


@app.function(gpu="B200", timeout=5400)
def bench():
    import importlib, json, statistics, sys, time, traceback
    import torch
    sys.path.insert(0, "/root/python_standalone")
    _write_old()
    new = importlib.import_module("cholesky_gluon_tcgen05_blocked")   # a_smem
    try:
        old = importlib.import_module("_v_outsmem")                    # out_smem
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
    for tag, m in (("pre", old), ("minimal", new)):
        m.custom_kernel(A)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            m.custom_kernel(A)
            torch.cuda.synchronize()
        prof.export_chrome_trace("/tmp/t.json")
        ev = json.load(open("/tmp/t.json"))["traceEvents"]
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
    print("=" * 86)
    print(r["gpu"], "  old = pre-cleanup (10 allocs, extra names)   new = minimal names (7 allocs)")
    print("=" * 86)
    lo, ln = [], []
    for x in r["rows"]:
        lo.append(x["old"]); ln.append(x["new"])
        print(f"{x['batch']:6} {x['n']:6} | {x['old']:9.3f} {x['new']:9.3f} "
              f"{x['old']/x['new']:7.3f}x | L diff {x['diff']:9.2e}  err {x['err']:9.2e}")
    g = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 86)
    print(f"{'GEOMEAN':>13} | {g(lo):9.3f} {g(ln):9.3f} {g(lo)/g(ln):7.3f}x")
    print()
    for tag in ("pre", "minimal"):
        t = r["k"][tag]
        print(f"  {tag:9}: panel {t['panel']:7.3f}  syrk {t['syrk']:7.3f} ms  "
              f"smem {t['smem']}B  regs {t['regs']}")
    print("  NaNs:", sum(x["nan"] for x in r["rows"]))
