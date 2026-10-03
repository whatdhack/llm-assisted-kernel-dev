"""
A/B of tmem.relinquish_alloc_permit() placement, in ONE container so the
comparison cannot be confounded by image/driver/harness drift.

  cute_early = cholesky_cute_tcgen05_blocked  (permit released right after
               the allocation -- current)
  cute_late  = cholesky_cute_permit_late      (permit held to the end of the
               CTA -- the previous code, iket events stripped so the two
               differ ONLY in permit placement)
  gluon      = control, unchanged code

Usage: modal run bench_permit_ab_b200.py
"""
import os

import modal

LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch", "triton==3.7.1", "numpy")
    .pip_install("nvidia-cutlass-dsl==4.7.1")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib",
        "CUTE_DSL_ARCH": "sm_100a",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(LOCAL_DIR, "/root/python_standalone")
)
app = modal.App("cholesky-permit-ab", image=image)


@app.function(gpu="B200", timeout=7200)
def bench():
    import importlib
    import sys
    import torch
    import cutlass
    sys.path.insert(0, "/root/python_standalone")
    import bench_leaderboard

    # str() matters: torch.__version__ is a TorchVersion instance, and pickling
    # it makes the RETURN VALUE require torch on the local side to deserialize.
    env = {"gpu": str(torch.cuda.get_device_name(0)), "torch": str(torch.__version__),
           "cutlass_dsl": str(cutlass.__version__)}
    mods = [("cute_early", "cholesky_cute_tcgen05_blocked"),
            ("cute_late", "cholesky_cute_permit_late"),
            ("gluon", "cholesky_gluon_tcgen05_blocked")]
    out, errs = {}, {}
    for label, name in mods:
        mod = importlib.import_module(name)
        ts = []
        for spec in bench_leaderboard.BENCHMARKS:
            print("bench", label, spec["batch"], spec["n"], flush=True)
            ts.append(bench_leaderboard.bench_one(mod.custom_kernel, spec) * 1e3)
        out[label] = [float(t) for t in ts]
        A = bench_leaderboard.generate_input(1, 2048, 2, 48192)
        L = mod.custom_kernel(A)
        errs[label] = (L @ L.transpose(-1, -2) - A).abs().amax().item()
    specs = [(int(x['batch']), int(x['n'])) for x in bench_leaderboard.BENCHMARKS]
    return env, out, errs, specs


@app.local_entrypoint()
def main():
    import math
    # Everything needed for the table comes back from the GPU side: importing
    # bench_leaderboard locally would pull in torch, which is not installed here.
    env, out, errs, specs = bench.remote()
    print(env)
    print(f"{'batch':>6} {'n':>6} | {'cute_early':>11} {'cute_late':>11} {'gluon':>11} |"
          f" {'late/early':>10} {'early/gluon':>11}")
    print("-" * 78)
    for i, (batch, n) in enumerate(specs):
        e, l, g = out["cute_early"][i], out["cute_late"][i], out["gluon"][i]
        print(f"{batch:>6} {n:>6} | {e:>11.3f} {l:>11.3f} {g:>11.3f} |"
              f" {l / e:>10.2f}x {e / g:>10.2f}x")
    geo = {k: math.exp(sum(math.log(v) for v in ts) / len(ts)) for k, ts in out.items()}
    print("-" * 78)
    print(f"{'GEOMEAN':>13} | {geo['cute_early']:>11.3f} {geo['cute_late']:>11.3f}"
          f" {geo['gluon']:>11.3f} | {geo['cute_late'] / geo['cute_early']:>10.2f}x"
          f" {geo['cute_early'] / geo['gluon']:>10.2f}x")
    print("recon err n=2048:", {k: f"{v:.3e}" for k, v in errs.items()})
