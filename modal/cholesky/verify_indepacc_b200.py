"""
A/B the independent-accumulator SYRK change on a real B200.

old: 3 MMAs chained through ONE acc_tmem (use_acc=True), each followed by a
     full mbarrier round-trip -> 27.9% barrier stall
new: 3 accumulators, 3 MMAs issued back to back, ONE wait, summed after

The three products (hi_m*hi_n, hi_m*lo_n, lo_m*hi_n) are independent; the
serialisation was an artefact of accumulator reuse. TMEM cost 64 -> 128
columns, which still admits 4 blocks/SM, so occupancy is unchanged.

Usage: modal run verify_indepacc_b200.py
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

app = modal.App("cholesky-verify-indepacc-b200", image=image)

SHAPES = [
    (1, 8192), (1, 16384), (1, 4096), (2, 4096), (2, 2048), (8, 2048),
    (16, 512), (640, 512), (64, 256), (1024, 64), (4096, 32), (4, 100),
]

OLD_BLOCK = '''    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    tcgen05_mma(l_m_hi_smem, l_n_hi_smem.permute((1, 0)), acc_tmem, use_acc=False, mbarriers=[bar], mbarrier_preds=[True])
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    mbarrier.init(bar, count=1)
    tcgen05_mma(l_m_hi_smem, l_n_lo_smem.permute((1, 0)), acc_tmem, use_acc=True, mbarriers=[bar], mbarrier_preds=[True])
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    mbarrier.init(bar, count=1)
    tcgen05_mma(l_m_lo_smem, l_n_hi_smem.permute((1, 0)), acc_tmem, use_acc=True, mbarriers=[bar], mbarrier_preds=[True])
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, num_warps)
    update = acc_tmem.load(acc_reg_layout)
'''


def _make_orig():
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert "acc0" in src, "expected the independent-accumulator working tree"
    start = src.index("    # The three tf32x3 products are mathematically INDEPENDENT")
    end = src.index("    a_reg = a_smem.load(acc_reg_layout)")
    o = src[:start] + OLD_BLOCK + "\n" + src[end:]
    assert "acc0" not in o and "count=3" not in o, "reverse-patch failed"
    open("/root/python_standalone/_v_chained.py", "w").write(o)


@app.function(gpu="B200", timeout=5400)
def verify():
    import importlib
    import json
    import math
    import statistics
    import sys
    import time

    import torch

    sys.path.insert(0, "/root/python_standalone")
    _make_orig()

    new_mod = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    old_mod = importlib.import_module("_v_chained")
    import bench_leaderboard

    rows = []
    for batch, n in SHAPES:
        A = bench_leaderboard.generate_input(batch, n, 2, 1234 + n)
        row = {"batch": batch, "n": n}
        for tag, mod in (("old", old_mod), ("new", new_mod)):
            L = mod.custom_kernel(A)
            torch.cuda.synchronize()
            row[f"{tag}_err"] = (L @ L.transpose(-1, -2) - A).abs().amax().item()
            row[f"{tag}_nan"] = int((~torch.isfinite(L)).sum().item())
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
        rows.append(row)

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
            k = "panel" if "panel" in e["name"] else ("syrk" if "syrk" in e["name"] else "other")
            a = agg.setdefault(k, [0, 0.0])
            a[0] += 1
            a[1] += e["dur"]
        sk = next(e for e in ks if "syrk" in e["name"])
        ktimes[tag] = {k: {"n": v[0], "ms": v[1] / 1000} for k, v in agg.items()}
        ktimes[tag]["smem"] = sk["args"].get("shared memory")
        ktimes[tag]["regs"] = sk["args"].get("registers per thread")
        k = mod._syrk_kernel_tcgen05
        cache = list(k.device_caches.values())[0][0]
        ktimes[tag]["tmem"] = getattr(next(iter(cache.values())).metadata, "tmem_size", None)

    return {"gpu": torch.cuda.get_device_name(0), "rows": rows, "ktimes": ktimes}


@app.local_entrypoint()
def main():
    import math
    r = verify.remote()
    print("=" * 96)
    print(r["gpu"], "  old = chained accumulator (3 waits),  new = 3 independent accs (1 wait)")
    print("=" * 96)
    print(f"{'batch':>6} {'n':>6} {'old err':>11} {'new err':>11} {'NaN':>4}"
          f" {'old ms':>9} {'new ms':>9} {'speedup':>9} {'L diff':>9}")
    lo, ln = [], []
    for x in r["rows"]:
        lo.append(x["old_ms"]); ln.append(x["new_ms"])
        print(f"{x['batch']:6} {x['n']:6} {x['old_err']:11.3e} {x['new_err']:11.3e}"
              f" {x['old_nan']+x['new_nan']:4} {x['old_ms']:9.3f} {x['new_ms']:9.3f}"
              f" {x['old_ms']/x['new_ms']:8.2f}x {x['max_diff']:9.2e}")
    go = math.exp(sum(math.log(t) for t in lo) / len(lo))
    gn = math.exp(sum(math.log(t) for t in ln) / len(ln))
    print("-" * 96)
    print(f"{'GEOMEAN':>13} {'':>28} {go:9.3f} {gn:9.3f} {go/gn:8.2f}x")
    print()
    print("--- per-kernel, batch=1 n=8192 ---")
    for tag in ("old", "new"):
        t = r["ktimes"][tag]
        print(f"  {tag}: panel {t['panel']['ms']:7.3f} ms   syrk {t['syrk']['ms']:7.3f} ms x{t['syrk']['n']:4}"
              f"   syrk smem {t['smem']}B regs {t['regs']} tmem {t['tmem']} cols")
    op, np_ = r["ktimes"]["old"]["syrk"]["ms"], r["ktimes"]["new"]["syrk"]["ms"]
    print(f"  syrk: {op:.3f} -> {np_:.3f} ms  ({100*(np_-op)/op:+.1f}%, {op/np_:.2f}x)")
