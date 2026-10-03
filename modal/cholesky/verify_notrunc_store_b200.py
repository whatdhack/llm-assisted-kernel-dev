"""Drop the two `hi` stores: feed the untouched source tile to the MMA.

l_m_smem/l_n_smem already hold the TMA'd fp32, and tcgen05 narrows fp32 ->
tf32 in hardware, so the hi operand needs no instruction and no store. The
split must then be consistent: lo is the residual against whatever narrowing
the hardware performs.

Removes, per element: the `+ 0x1000` forming hi (1.05M warp inst, 6.3% of
the syrk stream) and the two hi stores (262K, 1.6%). Total 7.8%.
Costs ~2 bits: hi+lo represents x to 2^-21 instead of 2^-23.

TWO VARIANTS, because the hardware's narrowing mode is not documented here:
  A  lo = round(x - trunc(x))   consistent iff tcgen05 TRUNCATES
  B  lo = round(x - rn(x))      consistent iff tcgen05 ROUNDS-TO-NEAREST
The inconsistent one mismatches hi by up to 1 ulp of hi, so its error lands
near 2^-11 instead of 2^-21. The accuracy column identifies which.

Usage: modal run verify_notrunc_store_b200.py
"""
import modal

import os as _os
# Kernel sources live one level up, in the cholesky_py problem directory.
LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "triton")
         .add_local_dir(LOCAL_DIR, "/root/python_standalone"))
app = modal.App("cholesky-notrunc-store", image=image)

TRUNC_DEF = '''
@gluon.jit
def _trunc_tf32(x):
    # fp32 -> tf32 by truncation: drop the low 13 mantissa bits, no round bit.
    bits = x.to(gl.uint32, bitcast=True)
    return (bits & 0xFFFFE000).to(gl.float32, bitcast=True)

'''

OLD_COMPUTE = """    l_m_reg = l_m_smem.load(REG_LAYOUT)
    l_m_hi = _round_tf32(l_m_reg)
    l_m_lo = _round_tf32(l_m_reg - l_m_hi)
    l_n_reg = l_n_smem.load(REG_LAYOUT)
    l_n_hi = _round_tf32(l_n_reg)
    l_n_lo = _round_tf32(l_n_reg - l_n_hi)"""

NEW_COMPUTE = """    # hi is the UNTOUCHED source tile: tcgen05 narrows fp32 -> tf32 itself, so
    # forming and storing hi is redundant. lo is the residual against that
    # same narrowing.
    l_m_reg = l_m_smem.load(REG_LAYOUT)
    l_m_lo = _round_tf32(l_m_reg - _HI_(l_m_reg))
    l_n_reg = l_n_smem.load(REG_LAYOUT)
    l_n_lo = _round_tf32(l_n_reg - _HI_(l_n_reg))"""


def _write(tag, hi_fn):
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert OLD_COMPUTE in src
    src = src.replace(OLD_COMPUTE, NEW_COMPUTE.replace("_HI_", hi_fn), 1)
    for dead in ("    l_m_smem.store(l_m_hi)\n", "    l_n_smem.store(l_n_hi)\n"):
        assert dead in src, dead
        src = src.replace(dead, "", 1)
    i = src.index("@gluon.jit\ndef _syrk_tile_tcgen05(")
    src = src[:i] + TRUNC_DEF.lstrip("\n") + "\n" + src[i:]
    open(f"/root/python_standalone/_v_{tag}.py", "w").write(src)
    return f"_v_{tag}"


@app.function(gpu="B200", timeout=7200, max_containers=2)
def bench():
    import importlib, json, sys, traceback
    import torch
    sys.path.insert(0, "/root/python_standalone")
    import bench_leaderboard
    base = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    mods = [("base", base)]
    for tag, fn in (("A_trunc", "_trunc_tf32"), ("B_rn", "_round_tf32")):
        try:
            mods.append((tag, importlib.import_module(_write(tag, fn))))
        except Exception:
            return {"fail": f"{tag}\n" + traceback.format_exc()[-3500:]}

    acc = []
    for batch, n in [(1, 512), (1, 2048), (1, 8192)]:
        A = bench_leaderboard.generate_input(batch, n, 2, 40000 + n)
        row = {"batch": batch, "n": n}
        for tag, m in mods:
            L = m.custom_kernel(A); torch.cuda.synchronize()
            row[tag] = (L @ L.transpose(-1, -2) - A).abs().amax().item()
            row[tag + "_nan"] = int((~torch.isfinite(L)).sum().item())
            del L
        acc.append(row); del A; torch.cuda.empty_cache()

    rows = []
    for spec in bench_leaderboard.BENCHMARKS:
        r = {"batch": spec["batch"], "n": spec["n"]}
        for tag, m in mods:
            try:
                r[tag] = bench_leaderboard.bench_one(m.custom_kernel, spec) * 1e3
            except Exception:
                return {"fail": f"{spec['n']} {tag}\n" + traceback.format_exc()[-2500:], "acc": acc}
        rows.append(r); torch.cuda.empty_cache()

    res = {}
    try:
        A = bench_leaderboard.generate_input(1, 8192, 2, 48192)
        for tag, m in mods:
            m.custom_kernel(A); torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
                m.custom_kernel(A); torch.cuda.synchronize()
            p.export_chrome_trace(f"/tmp/{tag}.json")
            a = {}
            for e in json.load(open(f"/tmp/{tag}.json"))["traceEvents"]:
                if e.get("cat") != "kernel": continue
                k = e["name"].split("(")[0][:30]
                d = a.setdefault(k, {"n": 0, "ms": 0.0, "regs": e.get("args", {}).get("registers per thread")})
                d["n"] += 1; d["ms"] += float(e.get("dur", 0)) / 1e3
            res[tag] = a
        del A
    except Exception:
        res = {"fail": traceback.format_exc()[-1500:]}
    return {"gpu": torch.cuda.get_device_name(0), "acc": acc, "rows": rows, "res": res}


@app.local_entrypoint()
def main():
    import json, math
    r = bench.remote()
    if "fail" in r:
        print("FAILED:\n", r["fail"]); print(json.dumps(r.get("acc"), indent=1)); return
    print("RAW", json.dumps({"acc": r["acc"], "rows": r["rows"]}))
    tags = ["base", "A_trunc", "B_rn"]
    print("\nACCURACY  max|LL^T - A|   (consistent variant ~ base; inconsistent ~2^-11 worse)")
    for x in r["acc"]:
        print(f"  n={x['n']:6} " + "  ".join(f"{t}={x[t]:.3e}" for t in tags)
              + "   nan=" + ",".join(str(x[t + "_nan"]) for t in tags))
    print(f"\n{'batch':>6} {'n':>6} |" + "".join(f"{t:>10}" for t in tags) + " | A/base  B/base")
    g = {t: [] for t in tags}
    for x in r["rows"]:
        for t in tags: g[t].append(x[t])
        print(f"{x['batch']:6} {x['n']:6} |" + "".join(f"{x[t]:10.3f}" for t in tags)
              + f" | {x['base']/x['A_trunc']:6.3f}x {x['base']/x['B_rn']:6.3f}x")
    gm = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 62)
    print(f"{'GEOMEAN':>13} |" + "".join(f"{gm(g[t]):10.3f}" for t in tags)
          + f" | {gm(g['base'])/gm(g['A_trunc']):6.3f}x {gm(g['base'])/gm(g['B_rn']):6.3f}x")
    if "fail" not in r["res"]:
        print("\nPER-KERNEL n=8192")
        for t in tags:
            for k, d in sorted(r["res"][t].items(), key=lambda kv: -kv[1]["ms"])[:2]:
                print(f"  {t:<9} {k:<28} {d['n']:5} {d['ms']:8.3f} ms  regs {d['regs']}")
