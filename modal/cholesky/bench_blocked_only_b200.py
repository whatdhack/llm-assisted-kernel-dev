"""Official 15 cases: cuSOLVER vs _blocked only."""
import modal

import os as _os
# Kernel sources live one level up, in the cholesky_py problem directory.
LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "triton")
         .add_local_dir(LOCAL_DIR, "/root/python_standalone"))
app = modal.App("cholesky-blocked-only", image=image)


@app.function(gpu="B200", timeout=7200, max_containers=2)
def bench():
    import importlib, json, sys, traceback
    import torch
    sys.path.insert(0, "/root/python_standalone")
    import bench_leaderboard
    mods = [("cus", importlib.import_module("starter")),
            ("blk", importlib.import_module("cholesky_gluon_tcgen05_blocked"))]
    rows = []
    for spec in bench_leaderboard.BENCHMARKS:
        r = {"batch": spec["batch"], "n": spec["n"]}
        for t, m in mods:
            try:
                r[t] = bench_leaderboard.bench_one(m.custom_kernel, spec) * 1e3
            except Exception:
                return {"fail": f"{spec['n']} {t}\n" + traceback.format_exc()[-2500:]}
        A = bench_leaderboard.generate_input(spec["batch"], spec["n"], spec["cond"], spec["seed"])
        L = mods[1][1].custom_kernel(A); torch.cuda.synchronize()
        r["err"] = (L @ L.transpose(-1, -2) - A).abs().amax().item()
        r["nan"] = int((~torch.isfinite(L)).sum().item())
        del A, L; torch.cuda.empty_cache()
        rows.append(r)
    res = {}
    try:
        A = bench_leaderboard.generate_input(1, 8192, 2, 48192)
        m = mods[1][1]; m.custom_kernel(A); torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            m.custom_kernel(A); torch.cuda.synchronize()
        p.export_chrome_trace("/tmp/t.json")
        for e in json.load(open("/tmp/t.json"))["traceEvents"]:
            if e.get("cat") != "kernel": continue
            k = e["name"].split("(")[0][:30]
            d = res.setdefault(k, {"n": 0, "ms": 0.0,
                                   "smem": e.get("args", {}).get("shared memory"),
                                   "regs": e.get("args", {}).get("registers per thread")})
            d["n"] += 1; d["ms"] += float(e.get("dur", 0)) / 1e3
        del A
    except Exception:
        res = {"fail": traceback.format_exc()[-1200:]}
    return {"rows": rows, "res": res}


@app.local_entrypoint()
def main():
    import json, math
    r = bench.remote()
    if "fail" in r:
        print("FAILED:\n", r["fail"]); return
    print("RAW", json.dumps(r["rows"]))
    print(f"{'batch':>6} {'n':>6} | {'cuSOLVER':>9} {'_blocked':>9} | {'speedup':>8} | {'err':>9} {'nan':>4}")
    c, b = [], []
    for x in r["rows"]:
        c.append(x["cus"]); b.append(x["blk"])
        print(f"{x['batch']:6} {x['n']:6} | {x['cus']:9.3f} {x['blk']:9.3f} |"
              f" {x['cus']/x['blk']:7.3f}x | {x['err']:9.2e} {x['nan']:4}")
    g = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 60)
    print(f"{'GEOMEAN':>13} | {g(c):9.3f} {g(b):9.3f} | {g(c)/g(b):7.3f}x   blk/cuS {g(b)/g(c):.3f}")
    print(f"  wins {sum(1 for x in r['rows'] if x['blk'] < x['cus'])}/15"
          f"   max err {max(x['err'] for x in r['rows']):.2e}"
          f"   nonfinite {sum(x['nan'] for x in r['rows'])}")
    if "fail" not in r["res"]:
        for k, d in sorted(r["res"].items(), key=lambda kv: -kv[1]["ms"]):
            print(f"  {k:<30} {d['n']:5} {d['ms']:8.3f} ms  smem {d['smem']}  regs {d['regs']}")
