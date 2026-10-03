"""
Official 15-case leaderboard suite, chained accumulator vs 3 independent
accumulators, on one Modal B200.

The earlier 12-shape A/B was contaminated by run-to-run drift: batch=4096/
n=32 showed a 1.13x "speedup" despite launching ZERO syrk kernels and
producing byte-identical output. So here the two variants are INTERLEAVED
within each shape (old, new, old, new, ...) rather than run in separate
passes, which cancels clock/thermal drift, and each gets more samples.

The working tree keeps the chained version; the independent-accumulator
variant is forward-patched into a scratch module inside the container.

Usage: modal run bench_indepacc_b200.py
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

app = modal.App("cholesky-bench-indepacc-b200", image=image)

REPS = 9          # interleaved reps per variant per shape

INDEP_BLOCK = '''    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc0 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    acc1 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    acc2 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    mbarrier.invalidate(bar)
    mbarrier.init(bar, count=3)
    tcgen05_mma(l_m_hi_smem, l_n_hi_smem.permute((1, 0)), acc0, use_acc=False, mbarriers=[bar], mbarrier_preds=[True])
    tcgen05_mma(l_m_hi_smem, l_n_lo_smem.permute((1, 0)), acc1, use_acc=False, mbarriers=[bar], mbarrier_preds=[True])
    tcgen05_mma(l_m_lo_smem, l_n_hi_smem.permute((1, 0)), acc2, use_acc=False, mbarriers=[bar], mbarrier_preds=[True])
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, num_warps)
    update = acc0.load(acc_reg_layout) + acc1.load(acc_reg_layout) + acc2.load(acc_reg_layout)

'''


def _write_indep():
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert "acc_tmem" in src and "acc0" not in src, "expected the chained working tree"
    start = src.index("    # The three tf32x3 products are mathematically INDEPENDENT")
    end = src.index("    a_reg = a_smem.load(acc_reg_layout)")
    o = src[:start] + INDEP_BLOCK + src[end:]
    assert "acc0" in o and "count=3" in o and "use_acc=True" not in o, "forward-patch failed"
    open("/root/python_standalone/_v_indep.py", "w").write(o)


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
    _write_indep()

    old = importlib.import_module("cholesky_gluon_tcgen05_blocked")   # chained
    new = importlib.import_module("_v_indep")                          # independent
    starter = importlib.import_module("starter")
    import bench_leaderboard

    rows = []
    for spec in bench_leaderboard.BENCHMARKS:
        A = bench_leaderboard.generate_input(spec["batch"], spec["n"],
                                             spec["cond"], spec["seed"])
        # warm up both (JIT + allocator) before any timing
        for m in (old, new, starter):
            for _ in range(2):
                m.custom_kernel(A)
        torch.cuda.synchronize()

        to, tn, ts = [], [], []
        for _ in range(REPS):                       # INTERLEAVED
            for m, acc in ((old, to), (new, tn), (starter, ts)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                m.custom_kernel(A)
                torch.cuda.synchronize()
                acc.append((time.perf_counter() - t0) * 1e3)

        Lo, Ln = old.custom_kernel(A), new.custom_kernel(A)
        torch.cuda.synchronize()
        row = {
            "batch": spec["batch"], "n": spec["n"],
            "old": statistics.median(to), "new": statistics.median(tn),
            "starter": statistics.median(ts),
            "old_min": min(to), "new_min": min(tn),
            "old_iqr": (max(to) - min(to)) / statistics.median(to) * 100,
            "new_iqr": (max(tn) - min(tn)) / statistics.median(tn) * 100,
            "old_err": (Lo @ Lo.transpose(-1, -2) - A).abs().amax().item(),
            "new_err": (Ln @ Ln.transpose(-1, -2) - A).abs().amax().item(),
            "nan": int((~torch.isfinite(Ln)).sum().item()),
        }
        del Lo, Ln, A
        torch.cuda.empty_cache()
        rows.append(row)

    # per-kernel split at the headline shape
    A = bench_leaderboard.generate_input(1, 8192, 2, 48192)
    ktimes = {}
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
            k = "panel" if "panel" in e["name"] else ("syrk" if "syrk" in e["name"] else "other")
            a = agg.setdefault(k, [0, 0.0])
            a[0] += 1
            a[1] += e["dur"]
        sk = next(e for e in ks if "syrk" in e["name"])
        ktimes[tag] = {k: v[1] / 1000 for k, v in agg.items()}
        ktimes[tag]["regs"] = sk["args"].get("registers per thread")
        kk = m._syrk_kernel_tcgen05
        cache = list(kk.device_caches.values())[0][0]
        ktimes[tag]["tmem"] = getattr(next(iter(cache.values())).metadata, "tmem_size", None)

    return {"gpu": torch.cuda.get_device_name(0), "rows": rows, "ktimes": ktimes,
            "reps": REPS}


@app.local_entrypoint()
def main():
    import math
    r = bench.remote()
    print("=" * 104)
    print(f"{r['gpu']}  official 15-case suite, INTERLEAVED, median of {r['reps']}")
    print("  old = chained accumulator (3 waits)   new = 3 independent accs (1 wait)")
    print("=" * 104)
    print(f"{'batch':>6} {'n':>6} | {'starter':>9} {'old':>9} {'new':>9} | {'speedup':>8} |"
          f" {'old spread':>10} {'new spread':>10} | {'new err':>10}")
    print("-" * 104)
    lo, ln, ls = [], [], []
    for x in r["rows"]:
        lo.append(x["old"]); ln.append(x["new"]); ls.append(x["starter"])
        print(f"{x['batch']:6} {x['n']:6} | {x['starter']:9.3f} {x['old']:9.3f} {x['new']:9.3f} |"
              f" {x['old']/x['new']:7.3f}x | {x['old_iqr']:9.1f}% {x['new_iqr']:9.1f}% |"
              f" {x['new_err']:10.2e}")
    g = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 104)
    print(f"{'GEOMEAN':>13} | {g(ls):9.3f} {g(lo):9.3f} {g(ln):9.3f} | {g(lo)/g(ln):7.3f}x")
    print()
    print("--- per-kernel, batch=1 n=8192 ---")
    for tag in ("old", "new"):
        t = r["ktimes"][tag]
        print(f"  {tag}: panel {t['panel']:7.3f} ms  syrk {t['syrk']:7.3f} ms  "
              f"syrk regs {t['regs']}  tmem {t['tmem']} cols")
    o, n = r["ktimes"]["old"]["syrk"], r["ktimes"]["new"]["syrk"]
    print(f"  syrk: {o:.3f} -> {n:.3f} ms ({100*(n-o)/o:+.2f}%)")
    print()
    print("NaNs:", sum(x["nan"] for x in r["rows"]))
