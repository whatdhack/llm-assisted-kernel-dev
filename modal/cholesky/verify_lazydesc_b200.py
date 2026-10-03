"""
A/B building the TMA descriptors lazily, i.e. only in blocks that have work.

_syrk_kernel_tcgen05 calls tma.make_tensor_descriptor twice BEFORE testing
the triangular guards, so all 8,192 blocks build descriptors -- but at
bk=0/n=8192, 2,016 of them (24.6%) then fail both guards and exit without
running a single tile.

On-device descriptor construction is not free: it writes through global
scratch and requires constant-cache invalidation (CCTL.E.C.LDCU.IV.DEEP,
4.23% of syrk samples), which is not private to the issuing block.

`pid_n_lo <= pid_m` is exactly "at least one tile runs" (pid_n_lo <
pid_n_hi always for even NUM_N_TILES), so hoisting the construction inside
that guard skips it for the empty blocks.

Usage: modal run verify_lazydesc_b200.py
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

app = modal.App("cholesky-verify-lazydesc-b200", image=image)
REPS = 9

OLD = """    l_desc = tma.make_tensor_descriptor(
        L_ptr + b * stride_b, [n, n], [stride_r, stride_c], [BLOCK_M, NB], l_layout,
    )
    a_desc = tma.make_tensor_descriptor(
        A_ptr + b * stride_b, [n, n], [stride_r, stride_c], [BLOCK_M, BLOCK_N], a_layout,
    )

    off_m = start + pid_m * BLOCK_M

    if pid_n_lo <= pid_m:
        off_n = start + pid_n_lo * BLOCK_N
        _syrk_tile_tcgen05(l_desc, a_desc, bk, off_m, off_n, num_warps=num_warps)
    if pid_n_hi <= pid_m and pid_n_hi != pid_n_lo:
        off_n = start + pid_n_hi * BLOCK_N
        _syrk_tile_tcgen05(l_desc, a_desc, bk, off_m, off_n, num_warps=num_warps)"""

NEW = """    off_m = start + pid_m * BLOCK_M

    # pid_n_lo < pid_n_hi always, so this is exactly "at least one tile runs".
    # Building the descriptors inside it skips them for the 2,016 of 8,192
    # blocks (bk=0, n=8192) that fail both triangular guards and exit.
    if pid_n_lo <= pid_m:
        l_desc = tma.make_tensor_descriptor(
            L_ptr + b * stride_b, [n, n], [stride_r, stride_c], [BLOCK_M, NB], l_layout,
        )
        a_desc = tma.make_tensor_descriptor(
            A_ptr + b * stride_b, [n, n], [stride_r, stride_c], [BLOCK_M, BLOCK_N], a_layout,
        )
        off_n = start + pid_n_lo * BLOCK_N
        _syrk_tile_tcgen05(l_desc, a_desc, bk, off_m, off_n, num_warps=num_warps)
        if pid_n_hi <= pid_m and pid_n_hi != pid_n_lo:
            off_n = start + pid_n_hi * BLOCK_N
            _syrk_tile_tcgen05(l_desc, a_desc, bk, off_m, off_n, num_warps=num_warps)"""


def _write_variant():
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert OLD in src, "call-site pattern not found"
    open("/root/python_standalone/_v_lazydesc.py", "w").write(src.replace(OLD, NEW, 1))


@app.function(gpu="B200", timeout=7200)
def bench():
    import importlib, json, statistics, sys, time, traceback
    import torch
    sys.path.insert(0, "/root/python_standalone")
    _write_variant()
    old = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    try:
        new = importlib.import_module("_v_lazydesc")
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
                     "new_err": (Ln @ Ln.transpose(-1, -2) - A).abs().amax().item(),
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
            key = "panel" if "panel" in e["name"] else ("syrk" if "syrk" in e["name"] else "other")
            agg.setdefault(key, 0.0)
            agg[key] += e["dur"]
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
    print("=" * 92)
    print(r["gpu"], "  old = descriptors always built   new = only in blocks with work")
    print("=" * 92)
    print(f"{'batch':>6} {'n':>6} | {'old ms':>9} {'new ms':>9} {'speedup':>8} | {'L diff':>9}")
    lo, ln = [], []
    for x in r["rows"]:
        lo.append(x["old"]); ln.append(x["new"])
        print(f"{x['batch']:6} {x['n']:6} | {x['old']:9.3f} {x['new']:9.3f} "
              f"{x['old']/x['new']:7.3f}x | {x['diff']:9.2e}")
    g = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 92)
    print(f"{'GEOMEAN':>13} | {g(lo):9.3f} {g(ln):9.3f} {g(lo)/g(ln):7.3f}x")
    print()
    for tag in ("old", "new"):
        t = r["k"][tag]
        print(f"  {tag}: panel {t['panel']:7.3f}  syrk {t['syrk']:7.3f} ms  "
              f"smem {t['smem']}B  regs {t['regs']}")
    o, n = r["k"]["old"]["syrk"], r["k"]["new"]["syrk"]
    print(f"  syrk: {o:.3f} -> {n:.3f} ms ({100*(n-o)/o:+.2f}%)")
    print("  NaNs:", sum(x["nan"] for x in r["rows"]))
