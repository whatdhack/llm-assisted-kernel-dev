"""
Sweep NB for cholesky_gluon_tcgen05_blocked.py on a Modal B200.

NB was pinned at 32 by two walls (see leaderboard_benchmark_results.txt):
  1. SYRK shared memory, linear in NB
  2. panel registers, quadratic in NB -- Lp = gl.zeros((NB,NB)) register-resident

Staging Lp into smem halved wall 2 (peak is now one register tile, L_rows,
not two). This measures whether NB can actually move, and whether it helps.

Expect a genuine tradeoff, not a free win:
  panel total work ~ (n/NB steps) x O(NB^2) per step  = O(n*NB)   -> LINEAR in NB
  SYRK traffic     ~ (n/NB steps) x O(n^2) per step   = O(n^3/NB) -> INVERSE in NB
so larger NB should help where SYRK dominates (large n, batch=1) and hurt
where the panel dominates (small n, high batch).

Usage: modal run sweep_nb_b200.py
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

app = modal.App("cholesky-sweep-nb-b200", image=image)

NB_VALUES = [32, 64, 128]
SHAPES = [(1, 32768), (1, 16384), (1, 8192), (2, 4096), (640, 512), (4096, 32)]


@app.function(gpu="B200", timeout=7200)
def sweep():
    import importlib
    import json
    import statistics
    import sys
    import time
    import traceback

    import torch

    sys.path.insert(0, "/root/python_standalone")
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert "    NB = 32\n" in src, "expected NB = 32 in custom_kernel"

    mods = {}
    for nb in NB_VALUES:
        s = src.replace("    NB = 32\n", f"    NB = {nb}\n", 1)
        name = f"_nb{nb}"
        open(f"/root/python_standalone/{name}.py", "w").write(s)
        mods[nb] = importlib.import_module(name)

    import bench_leaderboard

    out = {}
    for batch, n in SHAPES:
        A = bench_leaderboard.generate_input(batch, n, 2, 1234 + n)
        row = {}
        for nb in NB_VALUES:
            try:
                L = mods[nb].custom_kernel(A)
                torch.cuda.synchronize()
                err = (L @ L.transpose(-1, -2) - A).abs().amax().item()
                nan = int((~torch.isfinite(L)).sum().item())
                del L
                ts = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    mods[nb].custom_kernel(A)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                # per-kernel split
                with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    mods[nb].custom_kernel(A)
                    torch.cuda.synchronize()
                prof.export_chrome_trace("/tmp/t.json")
                ev = json.load(open("/tmp/t.json"))["traceEvents"]
                ks = [e for e in ev if e.get("cat") == "kernel"]
                agg = {}
                for e in ks:
                    k = "panel" if "panel" in e["name"] else (
                        "syrk" if "syrk" in e["name"] else "other")
                    a = agg.setdefault(k, [0, 0.0])
                    a[0] += 1
                    a[1] += e["dur"]
                pk = next(e for e in ks if "panel" in e["name"])
                sk = next((e for e in ks if "syrk" in e["name"]), None)
                row[nb] = {
                    "ok": True,
                    "ms": statistics.median(ts) * 1e3,
                    "err": err, "nan": nan,
                    "panel_ms": agg.get("panel", [0, 0])[1] / 1000,
                    "panel_n": agg.get("panel", [0, 0])[0],
                    "syrk_ms": agg.get("syrk", [0, 0])[1] / 1000,
                    "syrk_n": agg.get("syrk", [0, 0])[0],
                    "panel_regs": pk["args"].get("registers per thread"),
                    "panel_smem": pk["args"].get("shared memory"),
                    "syrk_smem": sk["args"].get("shared memory") if sk else None,
                }
            except Exception as e:
                row[nb] = {"ok": False,
                           "err_msg": f"{type(e).__name__}: {str(e).splitlines()[0][:140]}"}
                torch.cuda.empty_cache()
        out[(batch, n)] = row
        del A
        torch.cuda.empty_cache()

    return {"gpu": torch.cuda.get_device_name(0),
            "rows": {f"{b}x{n}": v for (b, n), v in out.items()}}


@app.local_entrypoint()
def main():
    r = sweep.remote()
    print("=" * 100)
    print(r["gpu"], " NB sweep")
    print("=" * 100)
    hdr = f"{'shape':>12} |" + "".join(f"{'NB=' + str(nb):>26}" for nb in NB_VALUES)
    print(hdr)
    print(f"{'':>12} |" + "".join(f"{'total':>9}{'panel':>8}{'syrk':>9}" for _ in NB_VALUES))
    print("-" * 100)
    for shape, row in r["rows"].items():
        line = f"{shape:>12} |"
        for nb in NB_VALUES:
            d = row.get(nb) or row.get(str(nb))
            if not d or not d.get("ok"):
                line += f"{'FAIL':>26}"
            else:
                line += f"{d['ms']:9.3f}{d['panel_ms']:8.2f}{d['syrk_ms']:9.2f}"
        print(line)
    print()
    print("failures / resource usage:")
    for shape, row in r["rows"].items():
        for nb in NB_VALUES:
            d = row.get(nb) or row.get(str(nb))
            if d and not d.get("ok"):
                print(f"  {shape} NB={nb}: {d['err_msg']}")
    for shape, row in r["rows"].items():
        for nb in NB_VALUES:
            d = row.get(nb) or row.get(str(nb))
            if d and d.get("ok"):
                print(f"  {shape:>12} NB={nb:<4} panel regs {d['panel_regs']:>4} "
                      f"smem {d['panel_smem']:>7}B | syrk smem {d['syrk_smem']}B | "
                      f"err {d['err']:.2e} nan {d['nan']} | "
                      f"launches panel {d['panel_n']} syrk {d['syrk_n']}")
        break  # resource numbers are shape-independent enough; show one shape
