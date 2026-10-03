"""
Where the CuTe port's extra wall-clock time actually goes, at batch=1/n=8192.

Splits total wall time into panel-kernel GPU time, SYRK-kernel GPU time, and
the gap between kernels (i.e. host-side launch cost), and dumps the per-launch
SYRK duration for the first 24 launches so the "offset bk steps are slower"
claim can be checked across many launches rather than the two ncu captured.

Usage: modal run split_cute_vs_gluon_b200.py
"""
import os
import time

import modal

LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(LOCAL_DIR, "outputs", "traces")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch", "triton==3.7.1", "numpy")
    # PINNED. Modal caches image layers by the definition hash, so an
    # unpinned "nvidia-cutlass-dsl" would keep whatever version was
    # latest when the layer was FIRST built, silently, forever -- and
    # two of this port's bugs were version-sensitive DSL semantics
    # (internal_type=TFloat32 rounding inside the TMA; cute.copy and
    # cute.gemm electing per-warp). Pin it so the number is a fact in
    # the source and the recorded measurements stay reproducible.
    .pip_install("nvidia-cutlass-dsl==4.7.1")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib",
        "CUTE_DSL_ARCH": "sm_100a",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(LOCAL_DIR, "/root/python_standalone")
)

app = modal.App("cholesky-split-cute-vs-gluon")

def _print_versions(tag=""):
    """Record what actually ran. The image pins nvidia-cutlass-dsl, but pinning
    is only half of it -- printing the version is what makes a saved artifact
    self-describing months later."""
    import subprocess, sys
    import torch
    import cutlass
    v = [f"torch {torch.__version__}", f"cutlass-dsl {cutlass.__version__}"]
    try:
        import triton
        v.append(f"triton {triton.__version__}")
    except ImportError:
        pass
    v.append(f"gpu {torch.cuda.get_device_name(0)}")
    print(("VERSIONS " + tag + ": " if tag else "VERSIONS: ") + ", ".join(v),
          flush=True)



MODULES = [("cute", "cholesky_cute_tcgen05_blocked"),
           ("gluon", "cholesky_gluon_tcgen05_blocked")]


@app.function(gpu="B200", image=image, timeout=1800)
def split(batch: int = 1, n: int = 8192):
    import importlib
    import json
    import sys
    import time

    import torch

    sys.path.insert(0, "/root/python_standalone")
    import bench_leaderboard

    _print_versions()
    out = {}
    for label, name in MODULES:
        mod = importlib.import_module(name)
        A = bench_leaderboard.generate_input(batch, n, 2, 48192)
        mod.custom_kernel(A)          # warm up JIT / caches
        torch.cuda.synchronize()

        # plain wall clock, no profiler attached
        walls = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mod.custom_kernel(A)
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1e3)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
        ) as prof:
            mod.custom_kernel(A)
            torch.cuda.synchronize()
        prof.export_chrome_trace(f"/tmp/{label}.json")
        with open(f"/tmp/{label}.json", "rb") as f:
            blob = f.read()
        ev = [e for e in json.load(open(f"/tmp/{label}.json"))["traceEvents"]
              if e.get("cat") == "kernel"]
        ev.sort(key=lambda e: e["ts"])

        def kind(e):
            nm = e["name"].lower()
            if "panel" in nm:
                return "panel"
            if "syrk" in nm:
                return "syrk"
            return "other"

        agg = {}
        for e in ev:
            k = kind(e)
            a = agg.setdefault(k, [0, 0.0])
            a[0] += 1
            a[1] += e["dur"]
        span = (max(e["ts"] + e["dur"] for e in ev) - min(e["ts"] for e in ev)) / 1e3
        busy = sum(e["dur"] for e in ev) / 1e3

        syrks = [round(e["dur"], 1) for e in ev if kind(e) == "syrk"][:24]
        out[label] = {
            "wall_ms": walls,
            "span_ms": span,
            "busy_ms": busy,
            "gap_ms": span - busy,
            "split": {k: {"n": v[0], "ms": v[1] / 1e3} for k, v in agg.items()},
            "first_syrk_us": syrks,
            "all_syrk_us": [round(e["dur"], 1) for e in ev if kind(e) == "syrk"],
            "all_panel_us": [round(e["dur"], 1) for e in ev if kind(e) == "panel"],
            "module": name,
            "trace": blob,
        }
        del A
        torch.cuda.empty_cache()
    traces = {k: v.pop("trace") for k, v in out.items()}
    print("RESULT_JSON", json.dumps(out), flush=True)
    return json.dumps(out), traces


@app.local_entrypoint()
def main(batch: int = 1, n: int = 8192):
    import json

    payload, traces = split.remote(batch, n)
    r = json.loads(payload)

    os.makedirs(TRACE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for label, blob in traces.items():
        path = os.path.join(
            TRACE_DIR, f"{r[label]['module']}__custom_kernel__b{batch}n{n}__{stamp}.json")
        with open(path, "wb") as f:
            f.write(blob)
        print(f"saved {label:6} kineto -> {path}")
        jpath = os.path.join(
            TRACE_DIR, f"{r[label]['module']}__launch_durations__b{batch}n{n}__{stamp}.json")
        with open(jpath, "w") as f:
            json.dump({"panel_us": r[label]["all_panel_us"],
                       "syrk_us": r[label]["all_syrk_us"]}, f)
        print(f"saved {label:6} per-launch -> {jpath}")
    print(f"\nbatch={batch} n={n}\n")
    print(f"{'':10} {'wall(med)':>10} {'gpu span':>9} {'gpu busy':>9} {'gap':>8} "
          f"{'panel ms':>9} {'x':>5} {'syrk ms':>9} {'x':>5}")
    for label in ("cute", "gluon"):
        d = r[label]
        w = sorted(d["wall_ms"])[1]
        p, s_ = d["split"].get("panel", {}), d["split"].get("syrk", {})
        print(f"{label:10} {w:10.3f} {d['span_ms']:9.3f} {d['busy_ms']:9.3f} "
              f"{d['gap_ms']:8.3f} {p.get('ms',0):9.3f} {p.get('n',0):5} "
              f"{s_.get('ms',0):9.3f} {s_.get('n',0):5}")
    c, g = r["cute"], r["gluon"]
    print(f"\n  ratios  panel {c['split']['panel']['ms']/g['split']['panel']['ms']:.2f}x"
          f"   syrk {c['split']['syrk']['ms']/g['split']['syrk']['ms']:.2f}x"
          f"   gap {c['gap_ms']/max(g['gap_ms'],1e-9):.2f}x")
    print("\n  first 24 syrk launches (us), alternating bk = 0, 32, 64, ...")
    for label in ("cute", "gluon"):
        print(f"    {label:6}", r[label]["first_syrk_us"])
