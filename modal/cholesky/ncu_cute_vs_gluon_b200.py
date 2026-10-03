"""
Side-by-side Nsight Compute profiles of the CuTe DSL SYRK and the Gluon SYRK
at batch=1 / n=8192, to attribute the ~1.3x geomean gap.

Usage: modal run ncu_cute_vs_gluon_b200.py
"""
import os
import time

import modal

LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(LOCAL_DIR, "outputs", "traces")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .run_commands(
        "apt-get update -qq",
        # MUST pin: the unversioned metapackage predates Blackwell.
        "apt-get install -y nsight-compute-2026.3.0",
        "apt-get install -y build-essential",
        # triton PINNED for the same reason as cutlass-dsl, and this one has already
        # broken once: triton 3.8.0 renamed get_tmem_reg_layout ->
        # _compute_tmem_reg_layout, so cholesky_gluon_tcgen05_blocked.py fails to
        # import and the Gluon baseline silently drops out of the comparison.
        "pip install torch triton==3.7.1 numpy",
        # PINNED. Modal caches image layers by the definition hash, so an
        # unpinned "nvidia-cutlass-dsl" would keep whatever version was
        # latest when the layer was FIRST built, silently, forever -- and
        # two of this port's bugs were version-sensitive DSL semantics
        # (internal_type=TFloat32 rounding inside the TMA; cute.copy and
        # cute.gemm electing per-warp). Pin it so the number is a fact in
        # the source and the recorded measurements stay reproducible.
        "pip install nvidia-cutlass-dsl==4.7.1",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib",
        "CUTE_DSL_ARCH": "sm_100a",
        "CUTE_DSL_LINEINFO": "1",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(LOCAL_DIR, "/root/python_standalone")
)

app = modal.App("cholesky-ncu-cute-vs-gluon")

# after the r2s/stmatrix store change: did the shared-memory wavefronts move?

RUNNER = r'''
import os
# Line info must be compiled INTO the kernel or ncu can only ever show SASS,
# whatever flags it is given. CUTE_DSL_LINEINFO is the CuTe DSL equivalent of
# TRITON_DISABLE_LINE_INFO=0; both are set so the two sides are matched.
os.environ["CUTE_DSL_LINEINFO"] = "1"
os.environ["TRITON_DISABLE_LINE_INFO"] = "0"
import shutil
shutil.rmtree(os.path.expanduser("~/.triton/cache"), ignore_errors=True)
import sys
import torch
import cutlass
print("VERSIONS: torch", torch.__version__, "cutlass-dsl", cutlass.__version__,
      "gpu", torch.cuda.get_device_name(0), flush=True)
sys.path.insert(0, "/root/python_standalone")
import importlib
mod = importlib.import_module(sys.argv[1])
import bench_leaderboard

batch, n = int(sys.argv[2]), int(sys.argv[3])
A = bench_leaderboard.generate_input(batch, n, 2, 48192)
mod.custom_kernel(A)
torch.cuda.synchronize()

torch.cuda.profiler.start()
L = mod.custom_kernel(A)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print("max_abs_err", (L @ L.transpose(-1, -2) - A).abs().amax().item())
'''

TARGETS = [
    ("cute_syrk", "cholesky_cute_tcgen05_blocked", "syrk", 0, 2),
    ("gluon_syrk", "cholesky_gluon_tcgen05_blocked", "_syrk_kernel_tcgen05", 0, 2),
    ("cute_panel", "cholesky_cute_tcgen05_blocked", "panel", 0, 2),
    ("gluon_panel", "cholesky_gluon_tcgen05_blocked", "_panel_trsm_kernel", 0, 2),
]

WANT = [
    "Duration", "Compute (SM) Throughput", "Memory Throughput",
    "Achieved Occupancy", "Theoretical Occupancy", "Block Limit Shared Mem",
    "Block Limit Registers", "Block Limit Warps", "Registers Per Thread",
    "Static Shared Memory Per Block", "Dynamic Shared Memory Per Block",
    "Waves Per SM", "Achieved Active Warps Per SM", "Executed Ipc Active",
    "Stall", "L2 Hit Rate", "DRAM Throughput", "Grid Size", "Block Size",
]


@app.function(gpu="B200", image=image, timeout=5400)
def ncu_all(batch: int, n: int, only: str = ""):
    import glob
    import shutil
    import subprocess
    import sys

    ncu = shutil.which("ncu") or next(
        iter(glob.glob("/opt/nvidia/nsight-compute/*/ncu")), None)
    assert ncu, "ncu not found"
    ver = subprocess.run([ncu, "--version"], capture_output=True,
                         text=True).stdout.strip()

    with open("/root/runner.py", "w") as f:
        f.write(RUNNER)

    results = {"ncu_version": ver, "targets": {}}
    for label, module, kregex, skip, count in TARGETS:
        if only and label not in only.split(","):
            continue
        out = f"/tmp/{label}"
        cmd = [
            ncu, "-o", out, "-f", "--target-processes", "all",
            "--profile-from-start", "off", "--set", "full",
            "--import-source", "yes",
            "--kernel-name", f"regex:{kregex}",
            "--launch-skip", str(skip), "--launch-count", str(count),
            sys.executable, "/root/runner.py", module, str(batch), str(n),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=4500)
        # ncu 2026.3 writes .ncu-repz; 2026.2.x and earlier wrote .ncu-rep.
        rep = next((out + e for e in (".ncu-repz", ".ncu-rep")
                    if os.path.exists(out + e)), out + ".ncu-rep")
        if not os.path.exists(rep):
            results["targets"][label] = {
                "ok": False, "rc": r.returncode,
                "stdout": r.stdout[-3000:], "stderr": r.stderr[-3000:],
                "tmp": sorted(glob.glob(f"/tmp/{label}*"))}
            continue
        det = subprocess.run([ncu, "--import", rep, "--page", "details"],
                             capture_output=True, text=True).stdout
        src = subprocess.run([ncu, "--import", rep, "--page", "source"],
                             capture_output=True, text=True).stdout
        # plain --page source silently defaults to sass-only even when line
        # info IS in the cubin; cuda,sass is what interleaves the Python line
        # above the SASS it produced.
        src_py = subprocess.run(
            [ncu, "--import", rep, "--page", "source",
             "--print-source", "cuda,sass"],
            capture_output=True, text=True).stdout
        csv = subprocess.run([ncu, "--import", rep, "--page", "raw", "--csv"],
                             capture_output=True, text=True).stdout
        with open(rep, "rb") as f:
            blob = f.read()
        results["targets"][label] = {"ok": True, "details": det, "source": src,
                                     "source_pycorr": src_py, "csv": csv,
                                     "report": blob, "rep_ext": os.path.splitext(rep)[1],
                                     "stdout": r.stdout[-800:]}
    return results


@app.local_entrypoint()
def main(batch: int = 1, n: int = 8192, only: str = ""):
    r = ncu_all.remote(batch, n, only)
    print("ncu:", r["ncu_version"].replace("\n", " "))
    os.makedirs(TRACE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for label, t in r["targets"].items():
        print(f"\n{'='*30} {label} {'='*30}")
        if not t["ok"]:
            print("FAILED rc=", t["rc"], "files:", t.get("tmp"))
            print("--- stdout"); print(t["stdout"][-2500:])
            print("--- stderr"); print(t["stderr"][-2500:]); continue
        base = os.path.join(TRACE_DIR, f"ncu_{label}_b{batch}n{n}__{stamp}")
        with open(base + t["rep_ext"], "wb") as f:
            f.write(t["report"])
        for ext, key in ((".details.txt", "details"), (".source.txt", "source"),
                         ("__source_pycorr.txt", "source_pycorr"),
                         (".raw.csv", "csv")):
            with open(base + ext, "w") as f:
                f.write(t[key])
        print("saved ->", base + t["rep_ext"], "(+ .details.txt .source.txt .raw.csv)")
        for line in t["details"].splitlines():
            if any(w in line for w in WANT):
                print("  ", line.strip())
        print("  ", t["stdout"].strip().splitlines()[-1:])
