"""
Attack the SYRK occupancy ceiling that ncu flags four different ways
(Est. 71.57% twice, 84.53%, 46.75% -- all one root cause).

occupancy = blocks/SM x warps/block
  blocks/SM   = 4, pinned by 50,184 B shared memory per block
  warps/block = 4, pinned by nothing but `num_warps=4` at the launch site

So sweep both axes:
  num_warps 4 -> 8 : doubles warps/SM at identical smem   (the cheap lever)
  BLOCK_M/N 32/64/128 : changes smem per block, hence blocks/SM

Usage: modal run sweep_syrk_b200.py
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

app = modal.App("cholesky-sweep-syrk-b200", image=image)

# (BLOCK_MN, num_warps)
CONFIGS = [(64, 4), (64, 8), (32, 4), (32, 8), (128, 4), (128, 8)]
SHAPES = [(1, 8192), (1, 16384), (2, 4096), (640, 512)]


@app.function(gpu="B200", timeout=7200)
def sweep():
    import importlib
    import json
    import statistics
    import sys
    import time

    import torch

    sys.path.insert(0, "/root/python_standalone")
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert "    BLOCK_M = BLOCK_N = 64\n" in src
    assert "NUM_N_TILES=num_n_tiles, num_warps=4,\n" in src

    mods = {}
    for bmn, nw in CONFIGS:
        s = src.replace("    BLOCK_M = BLOCK_N = 64\n",
                        f"    BLOCK_M = BLOCK_N = {bmn}\n", 1)
        s = s.replace("NUM_N_TILES=num_n_tiles, num_warps=4,\n",
                      f"NUM_N_TILES=num_n_tiles, num_warps={nw},\n", 1)
        name = f"_syrk_b{bmn}_w{nw}"
        open(f"/root/python_standalone/{name}.py", "w").write(s)
        try:
            mods[(bmn, nw)] = importlib.import_module(name)
        except Exception as e:
            mods[(bmn, nw)] = f"IMPORT FAIL: {type(e).__name__}: {str(e).splitlines()[0][:90]}"

    import bench_leaderboard

    out = {}
    for batch, n in SHAPES:
        A = bench_leaderboard.generate_input(batch, n, 2, 1234 + n)
        row = {}
        for cfg in CONFIGS:
            mod = mods[cfg]
            if isinstance(mod, str):
                row[str(cfg)] = {"ok": False, "msg": mod}
                continue
            try:
                L = mod.custom_kernel(A)
                torch.cuda.synchronize()
                err = (L @ L.transpose(-1, -2) - A).abs().amax().item()
                nan = int((~torch.isfinite(L)).sum().item())
                del L
                ts = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    mod.custom_kernel(A)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    mod.custom_kernel(A)
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
                sk = next((e for e in ks if "syrk" in e["name"]), None)
                row[str(cfg)] = {
                    "ok": True,
                    "ms": statistics.median(ts) * 1e3,
                    "err": err, "nan": nan,
                    "syrk_ms": agg.get("syrk", [0, 0])[1] / 1000,
                    "syrk_n": agg.get("syrk", [0, 0])[0],
                    "panel_ms": agg.get("panel", [0, 0])[1] / 1000,
                    "syrk_smem": sk["args"].get("shared memory") if sk else None,
                    "syrk_regs": sk["args"].get("registers per thread") if sk else None,
                    "syrk_block": sk["args"].get("block") if sk else None,
                }
            except Exception as e:
                row[str(cfg)] = {"ok": False,
                                 "msg": f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"}
                torch.cuda.empty_cache()
        out[f"{batch}x{n}"] = row
        del A
        torch.cuda.empty_cache()
    return {"gpu": torch.cuda.get_device_name(0), "rows": out}


@app.local_entrypoint()
def main():
    r = sweep.remote()
    print("=" * 112)
    print(r["gpu"], "  SYRK sweep (BLOCK_M=BLOCK_N / num_warps).  total ms | syrk ms")
    print("=" * 112)
    print(f"{'shape':>10} |" + "".join(f"{f'B{b}/w{w}':>17}" for b, w in CONFIGS))
    print(f"{'':>10} |" + "".join(f"{'total':>9}{'syrk':>8}" for _ in CONFIGS))
    print("-" * 112)
    for shape, row in r["rows"].items():
        line = f"{shape:>10} |"
        for cfg in CONFIGS:
            d = row[str(cfg)]
            line += f"{'FAIL':>17}" if not d["ok"] else f"{d['ms']:9.3f}{d['syrk_ms']:8.2f}"
        print(line)
    print()
    print("resources / correctness:")
    seen = set()
    for shape, row in r["rows"].items():
        for cfg in CONFIGS:
            d = row[str(cfg)]
            if d["ok"] and str(cfg) not in seen:
                seen.add(str(cfg))
                blocks_smem = 233472 // (d['syrk_smem'] + 1024)
                warps = d['syrk_block'][0] // 32 if d['syrk_block'] else 0
                print(f"  B{cfg[0]:<3}/w{cfg[1]:<2} smem {d['syrk_smem']:>7}B regs {d['syrk_regs']:>4} "
                      f"block {d['syrk_block']} | ~{blocks_smem} blk/SM x {warps} warp "
                      f"= {blocks_smem*warps} warps ({100*blocks_smem*warps/64:.0f}%) | "
                      f"err {d['err']:.2e} nan {d['nan']} | syrk launches {d['syrk_n']}")
    print()
    fails = set()
    for shape, row in r["rows"].items():
        for cfg in CONFIGS:
            d = row[str(cfg)]
            if not d["ok"] and str(cfg) not in fails:
                fails.add(str(cfg))
                print(f"  FAIL B{cfg[0]}/w{cfg[1]}: {d['msg']}")
