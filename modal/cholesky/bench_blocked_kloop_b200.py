"""Official 15-case benchmark of _blocked and _kloop, post-cleanup.

Re-run after the dead-variable/stale-comment cleanup of 2026-08-17. Both
edits are codegen-neutral by construction (an assigned-but-never-read name,
an inlined single-use intermediate, and comment text), and the panel output
was already shown bitwise identical before/after on the local GPU. This run
confirms that end to end on the B200 and refreshes both columns.

The registers/smem block doubles as a codegen-identity check: if it matches
the 2026-08-16 run exactly, the cleanup provably changed no instructions.
    kloop panel 16,384B / 168 regs      kloop syrk 65,544B / 58 regs
    base  panel  4,096B /  64 regs      base  syrk 49,176B / 99 regs

PROTOCOL: bench_leaderboard.bench_one (warmup 2, median of 5, fresh input),
all 15 official specs, variants INTERLEAVED within each shape.

Usage: modal run bench_blocked_kloop_b200.py
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

app = modal.App("cholesky-blocked-kloop", image=image)


def _resources(mods, spec):
    """Per-kernel smem/regs/blocks-per-SM from the Kineto trace args."""
    import json, os, torch
    import bench_leaderboard

    props = torch.cuda.get_device_properties(0)
    smem_per_sm = getattr(props, "shared_memory_per_multiprocessor", 233472)
    regs_per_sm = getattr(props, "regs_per_multiprocessor", 65536)
    thr_per_sm = getattr(props, "max_threads_per_multi_processor", 2048)

    A = bench_leaderboard.generate_input(spec["batch"], spec["n"], spec["cond"], spec["seed"])
    out = {"smem_per_sm": smem_per_sm, "regs_per_sm": regs_per_sm,
           "threads_per_sm": thr_per_sm, "kernels": {}}
    for tag, m in mods:
        m.custom_kernel(A)
        torch.cuda.synchronize()
        path = f"/tmp/trace_{tag}.json"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], with_stack=False,
        ) as prof:
            m.custom_kernel(A)
            torch.cuda.synchronize()
        prof.export_chrome_trace(path)
        agg = {}
        for e in json.load(open(path))["traceEvents"]:
            if e.get("cat") != "kernel":
                continue
            a = e.get("args", {})
            smem, regs = a.get("shared memory"), a.get("registers per thread")
            if smem is None or regs is None:
                continue
            blk = a.get("block", [0, 0, 0])
            threads = int(blk[0]) * max(int(blk[1]), 1) * max(int(blk[2]), 1)
            k = (e["name"].split("(")[0][:40], int(smem), int(regs), threads)
            d = agg.setdefault(k, {"launches": 0, "us": 0.0})
            d["launches"] += 1
            d["us"] += float(e.get("dur", 0))
        rows = []
        for (name, smem, regs, threads), d in sorted(agg.items(), key=lambda kv: -kv[1]["us"]):
            by_smem = smem_per_sm // smem if smem else 99
            by_regs = regs_per_sm // (regs * threads) if regs * threads else 99
            by_thr = thr_per_sm // threads if threads else 99
            rows.append({"name": name, "smem": smem, "regs": regs, "threads": threads,
                         "launches": d["launches"], "ms": d["us"] / 1e3,
                         "blocks_sm": min(by_smem, by_regs, by_thr),
                         "by": f"{by_smem}/{by_regs}/{by_thr}"})
        out["kernels"][tag] = rows
        os.remove(path)
    del A
    torch.cuda.empty_cache()
    return out


@app.function(gpu="B200", timeout=7200, max_containers=2)
def bench():
    import importlib, sys, traceback
    import torch
    sys.path.insert(0, "/root/python_standalone")
    import bench_leaderboard
    try:
        mods = [
            ("starter", importlib.import_module("starter")),
            ("base", importlib.import_module("cholesky_gluon_tcgen05_blocked")),
            ("kloop", importlib.import_module("cholesky_gluon_tcgen05_kloop")),
        ]
    except Exception:
        return {"fail": traceback.format_exc()[-3000:]}
    mod = dict(mods)

    rows = []
    for spec in bench_leaderboard.BENCHMARKS:
        row = {"batch": spec["batch"], "n": spec["n"]}
        for tag, m in mods:                      # interleaved within the shape
            try:
                row[f"{tag}_ms"] = bench_leaderboard.bench_one(m.custom_kernel, spec) * 1e3
            except Exception:
                return {"fail": f"{spec['batch']}x{spec['n']} {tag}\n"
                                + traceback.format_exc()[-2500:]}
        A = bench_leaderboard.generate_input(spec["batch"], spec["n"], spec["cond"], spec["seed"])
        for tag in ("base", "kloop"):
            L = mod[tag].custom_kernel(A)
            torch.cuda.synchronize()
            row[f"{tag}_err"] = (L @ L.transpose(-1, -2) - A).abs().amax().item()
            row[f"{tag}_nan"] = int((~torch.isfinite(L)).sum().item())
            del L
        rows.append(row)
        del A
        torch.cuda.empty_cache()

    try:
        res = _resources([("kloop", mod["kloop"]), ("base", mod["base"])],
                         {"batch": 1, "n": 8192, "cond": 2, "seed": 48192})
    except Exception:
        res = {"fail": traceback.format_exc()[-2000:]}
    return {"gpu": torch.cuda.get_device_name(0), "rows": rows, "res": res}


@app.local_entrypoint()
def main():
    import json, math
    r = bench.remote()
    if "fail" in r:
        print("RUN FAILED:\n", r["fail"])
        return
    print("RAW_ROWS", json.dumps(r["rows"]))
    print("RAW_RES", json.dumps(r["res"]))

    print("=" * 92)
    print(f"{r['gpu']}   post-cleanup (2026-08-17)   _blocked NB=32   _kloop NB_INNER=64")
    print("=" * 92)
    print(f"{'batch':>6} {'n':>6} | {'cuSOLVER':>9} {'_blocked':>9} {'_kloop':>9} |"
          f" {'blk/cuS':>8} {'kl/blk':>7} | {'err(blk)':>9} {'err(kl)':>9}")
    ss, bs, ks, ds = [], [], [], []
    for x in r["rows"]:
        ss.append(x["starter_ms"]); bs.append(x["base_ms"]); ks.append(x["kloop_ms"])
        ds.append(min(x["base_ms"], x["kloop_ms"]))
        win = "*" if x["base_ms"] < x["starter_ms"] else " "
        print(f"{x['batch']:6} {x['n']:6} | {x['starter_ms']:9.3f} {x['base_ms']:9.3f}{win}"
              f"{x['kloop_ms']:9.3f} | {x['starter_ms'] / x['base_ms']:7.3f}x"
              f" {x['base_ms'] / x['kloop_ms']:6.3f}x | {x['base_err']:9.2e} {x['kloop_err']:9.2e}")
    g = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 92)
    print(f"{'GEOMEAN':>13} | {g(ss):9.3f} {g(bs):9.3f} {g(ks):10.3f} |"
          f" {g(ss) / g(bs):7.3f}x {g(bs) / g(ks):6.3f}x")
    print()
    print(f"  _blocked {g(bs):.3f} ms = {g(ss) / g(bs):.3f}x cuSOLVER, wins "
          f"{sum(1 for x in r['rows'] if x['base_ms'] < x['starter_ms'])}/15")
    print(f"  _kloop   {g(ks):.3f} ms = {g(ss) / g(ks):.3f}x cuSOLVER, beats _blocked on "
          f"{sum(1 for x in r['rows'] if x['kloop_ms'] < x['base_ms'])}/15")
    print(f"  oracle dispatch min(): {g(ds):.3f} ms = {g(ss) / g(ds):.3f}x cuSOLVER")
    print(f"  max err: _blocked {max(x['base_err'] for x in r['rows']):.2e}"
          f"  _kloop {max(x['kloop_err'] for x in r['rows']):.2e}"
          f"   nonfinite: {sum(x['base_nan'] + x['kloop_nan'] for x in r['rows'])}")

    res = r["res"]
    if "fail" in res:
        print("\nRESOURCE CAPTURE FAILED (timings unaffected):\n", res["fail"])
        return
    print()
    print("CODEGEN IDENTITY CHECK vs the 2026-08-16 pre-cleanup run")
    print(f"{'variant':>7} {'kernel':<26} {'smem':>8} {'regs':>5} {'launch':>7} {'ms':>8}"
          f" {'blk/SM':>7}   expected")
    exp = {("kloop", "_panel_trsm_kernel"): "16384B/168r",
           ("kloop", "_syrk_kernel_tcgen05"): "65544B/58r",
           ("base", "_panel_trsm_kernel"): "4096B/64r",
           ("base", "_syrk_kernel_tcgen05"): "49176B/99r"}
    for tag in ("kloop", "base"):
        for k in res["kernels"].get(tag, []):
            e = exp.get((tag, k["name"]), "")
            got = f"{k['smem']}B/{k['regs']}r"
            mark = "  MATCH" if e and got == e else ("  DIFFERS!" if e else "")
            print(f"{tag:>7} {k['name']:<26} {k['smem']:8} {k['regs']:5} {k['launches']:7}"
                  f" {k['ms']:8.3f} {k['blocks_sm']:7}   {e}{mark}")
