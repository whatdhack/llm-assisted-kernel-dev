"""
A/B hoisting the l_m load + hi/lo split out of _syrk_tile_tcgen05.

Both tile calls in _syrk_kernel_tcgen05 pass the SAME off_m, so l_m and its
tf32 hi/lo split are computed twice by every block that runs two tiles.
At bk=0, n=8192 that is 2,080 of 8,192 blocks (25%); 4,096 run one tile and
2,016 run none.

Hoisting does it once per block. Note `pid_n_lo <= pid_m` is exactly the
"at least one tile runs" test (pid_n_lo < pid_n_hi always for even
NUM_N_TILES), so guarding on it also keeps the 2,016 empty blocks from
doing the hoisted work.

RISK: l_m_hi_smem / l_m_lo_smem must now stay live across both tile calls,
whereas today triton can alias them with the per-call allocations. That may
raise peak shared memory and drop residency below 4 blocks/SM -- which the
run reports so it is visible either way.

Working tree keeps the current version; the variant is built here.

Usage: modal run verify_hoistlm_b200.py
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

app = modal.App("cholesky-verify-hoistlm-b200", image=image)

REPS = 9


def _write_variant():
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()

    # 1. helper takes the pre-split l_m tiles
    o = src.replace(
        "def _syrk_tile_tcgen05(l_desc, a_desc, bk, off_m, off_n, num_warps: gl.constexpr):",
        "def _syrk_tile_tcgen05(l_desc, a_desc, bk, off_m, off_n,\n"
        "                       l_m_hi_smem, l_m_lo_smem, num_warps: gl.constexpr):", 1)

    # 2. helper no longer loads or allocates l_m
    o = o.replace(
        "    l_m_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)\n"
        "    l_n_smem", "    l_n_smem", 1)
    o = o.replace(
        "    mbarrier.expect(bar, 2 * l_desc.block_type.nbytes)\n"
        "    tma.async_copy_global_to_shared(l_desc, [off_m, bk], bar, l_m_smem)\n",
        "    mbarrier.expect(bar, l_desc.block_type.nbytes)\n", 1)

    # 3. helper no longer splits l_m
    o = o.replace(
        "    l_m_reg = l_m_smem.load(REG_LAYOUT)\n"
        "    l_m_hi = _round_tf32(l_m_reg)\n"
        "    l_m_lo = _round_tf32(l_m_reg - l_m_hi)\n", "", 1)
    o = o.replace(
        "    l_m_hi_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)\n"
        "    l_m_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)\n",
        "", 1)
    o = o.replace("    l_m_hi_smem.store(l_m_hi)\n    l_m_lo_smem.store(l_m_lo)\n", "", 1)

    # 4. kernel does it once, guarded by "at least one tile runs"
    old_calls = """    if pid_n_lo <= pid_m:
        off_n = start + pid_n_lo * BLOCK_N
        _syrk_tile_tcgen05(l_desc, a_desc, bk, off_m, off_n, num_warps=num_warps)
    if pid_n_hi <= pid_m and pid_n_hi != pid_n_lo:
        off_n = start + pid_n_hi * BLOCK_N
        _syrk_tile_tcgen05(l_desc, a_desc, bk, off_m, off_n, num_warps=num_warps)"""
    new_calls = """    # pid_n_lo < pid_n_hi always, so this is exactly "at least one tile runs":
    # the 2,016 blocks that fail it skip the hoisted work entirely.
    if pid_n_lo <= pid_m:
        REG_LAYOUT_M: gl.constexpr = gl.BlockedLayout([1, 16], [16, 2], [num_warps, 1], [1, 0])
        bar_m = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar_m, count=1)
        l_m_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
        mbarrier.expect(bar_m, l_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(l_desc, [off_m, bk], bar_m, l_m_smem)
        mbarrier.wait(bar_m, phase=0)
        mbarrier.invalidate(bar_m)

        l_m_reg = l_m_smem.load(REG_LAYOUT_M)
        l_m_hi = _round_tf32(l_m_reg)
        l_m_lo = _round_tf32(l_m_reg - l_m_hi)
        l_m_hi_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
        l_m_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
        l_m_hi_smem.store(l_m_hi)
        l_m_lo_smem.store(l_m_lo)
        fence_async_shared()

        off_n = start + pid_n_lo * BLOCK_N
        _syrk_tile_tcgen05(l_desc, a_desc, bk, off_m, off_n,
                           l_m_hi_smem, l_m_lo_smem, num_warps=num_warps)
        if pid_n_hi <= pid_m and pid_n_hi != pid_n_lo:
            off_n = start + pid_n_hi * BLOCK_N
            _syrk_tile_tcgen05(l_desc, a_desc, bk, off_m, off_n,
                               l_m_hi_smem, l_m_lo_smem, num_warps=num_warps)"""
    assert old_calls in o, "call-site pattern not found"
    o = o.replace(old_calls, new_calls, 1)

    assert "l_m_smem.load(REG_LAYOUT)" not in o, "helper still splits l_m"
    assert o.count("_round_tf32(l_m_reg)") == 1, "l_m split should appear once"
    open("/root/python_standalone/_v_hoistlm.py", "w").write(o)


@app.function(gpu="B200", timeout=7200)
def bench():
    import importlib, json, math, statistics, sys, time
    import torch

    sys.path.insert(0, "/root/python_standalone")
    _write_variant()
    old = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    try:
        new = importlib.import_module("_v_hoistlm")
    except Exception as e:
        import traceback
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
        except Exception as e:
            return {"fail": f"{spec}: {type(e).__name__}: {str(e).splitlines()[0][:200]}"}
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
                     "old_err": (Lo @ Lo.transpose(-1, -2) - A).abs().amax().item(),
                     "new_err": (Ln @ Ln.transpose(-1, -2) - A).abs().amax().item(),
                     "diff": (Lo - Ln).abs().amax().item(),
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
        print("VARIANT FAILED:\n", r["fail"])
        return
    print("=" * 96)
    print(r["gpu"], "  old = l_m split per tile   new = l_m hoisted, split once per block")
    print("=" * 96)
    print(f"{'batch':>6} {'n':>6} | {'old ms':>9} {'new ms':>9} {'speedup':>8} | {'L diff':>9} {'new err':>10}")
    lo, ln = [], []
    for x in r["rows"]:
        lo.append(x["old"]); ln.append(x["new"])
        print(f"{x['batch']:6} {x['n']:6} | {x['old']:9.3f} {x['new']:9.3f} {x['old']/x['new']:7.3f}x |"
              f" {x['diff']:9.2e} {x['new_err']:10.2e}")
    g = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 96)
    print(f"{'GEOMEAN':>13} | {g(lo):9.3f} {g(ln):9.3f} {g(lo)/g(ln):7.3f}x")
    print()
    for tag in ("old", "new"):
        t = r["k"][tag]
        blocks = 233472 // (t["smem"] + 1024)
        print(f"  {tag}: panel {t['panel']:7.3f}  syrk {t['syrk']:7.3f} ms  "
              f"smem {t['smem']}B (~{blocks} blk/SM)  regs {t['regs']}")
    o, n = r["k"]["old"]["syrk"], r["k"]["new"]["syrk"]
    print(f"  syrk: {o:.3f} -> {n:.3f} ms ({100*(n-o)/o:+.2f}%)")
    print("  NaNs:", sum(x["nan"] for x in r["rows"]))
