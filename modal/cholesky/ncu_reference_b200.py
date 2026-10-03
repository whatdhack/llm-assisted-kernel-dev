"""
Collects an Nsight Compute (.ncu-rep) profile of the cuSOLVER reference
implementation (starter.py, torch.linalg.cholesky_ex) on a real B200 via
Modal, for comparison against the ncu_cholesky_b200.py profiles of the
gluon kernels.

Kernel names were identified via profile_cholesky_b200.py --module starter:
the panel kernel (templated on getrf_wo_pivot_params_ in its demangled
kineto label, but NCU's kernel-name filter sees only the untemplated outer
function name "kernel", dominant) and syherk_kernel_ldgsts (cuBLAS SYRK
variants).

Usage:
    modal run ncu_reference_b200.py
    modal run ncu_reference_b200.py --batch 1 --n 8192

Report is written to
python_standalone/outputs/traces/starter__ncu_full_b<batch>n<n>__<stamp>.ncu-rep
"""
import os
import time

import modal

import os as _os
# Kernel sources live one level up, in the cholesky_py problem directory.
LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
TRACE_DIR = os.path.join(LOCAL_DIR, "outputs", "traces")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-base-ubuntu24.04", add_python="3.12")
    .run_commands(
        # the nvidia/cuda base image already has the CUDA apt repo configured;
        # adding cuda-keyring on top of it conflicts on Signed-By.
        "apt-get update -qq",
        # MUST pin a version: the unversioned `nsight-compute` metapackage in
        # this repo resolves to 2022.4.1, which predates Blackwell and fails
        # with "Profiling is not supported on device 0" on sm_100.
        "apt-get install -y nsight-compute-2026.2.1",
        # triton JIT-builds a C driver shim at runtime, so it needs a compiler
        "apt-get install -y build-essential",
        "pip install torch triton",
    )
    .add_local_dir(LOCAL_DIR, "/root/python_standalone")
)

app = modal.App("cholesky-ncu-reference-b200", image=image)

# Runs inside ncu. Warmup first, then profiling is switched on for exactly one
# custom_kernel call.
RUNNER = r'''
import sys
import torch
sys.path.insert(0, "/root/python_standalone")
import importlib
mod = importlib.import_module(sys.argv[1])
import bench_leaderboard

batch, n = int(sys.argv[2]), int(sys.argv[3])
A = bench_leaderboard.generate_input(batch, n, 2, 48192)
mod.custom_kernel(A)          # warmup: JIT compile everything
torch.cuda.synchronize()

torch.cuda.profiler.start()
L = mod.custom_kernel(A)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print("max_abs_err", (L @ L.transpose(-1, -2) - A).abs().amax().item())
'''


@app.function(gpu="B200", timeout=3600)
def ncu_profile(module: str, batch: int, n: int, launches: int):
    import glob
    import shutil
    import subprocess
    import sys

    ncu = shutil.which("ncu") or next(iter(glob.glob("/opt/nvidia/nsight-compute/*/ncu")), None)
    assert ncu, "ncu not found"
    ver = subprocess.run([ncu, "--version"], capture_output=True, text=True).stdout.strip()

    with open("/root/runner.py", "w") as f:
        f.write(RUNNER)

    out = "/tmp/report"
    cmd = [
        ncu, "-o", out, "-f",
        "--target-processes", "all",
        "--profile-from-start", "off",
        "--set", "full",
        "--import-source", "yes",
        "--kernel-name", "regex:(^kernel$|syherk_kernel_ldgsts)",
        "--launch-count", str(launches),
        sys.executable, "/root/runner.py", module, str(batch), str(n),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3300)

    rep = out + ".ncu-rep"
    if not os.path.exists(rep):
        return {"ok": False, "ncu_version": ver, "cmd": " ".join(cmd),
                "stdout": r.stdout[-6000:], "stderr": r.stderr[-6000:], "rc": r.returncode}

    # text summary so the findings are readable without the GUI
    det = subprocess.run([ncu, "--import", rep, "--page", "details"],
                         capture_output=True, text=True).stdout
    csv = subprocess.run([ncu, "--import", rep, "--page", "raw", "--csv"],
                         capture_output=True, text=True).stdout

    with open(rep, "rb") as f:
        blob = f.read()
    return {"ok": True, "ncu_version": ver, "rc": r.returncode,
            "stdout": r.stdout[-3000:], "stderr": r.stderr[-3000:],
            "details": det, "csv": csv, "report": blob}


@app.local_entrypoint()
def main(module: str = "starter",
         batch: int = 1, n: int = 8192, launches: int = 8, tag: str = ""):
    r = ncu_profile.remote(module, batch, n, launches)
    print("ncu:", r["ncu_version"].replace("\n", " "))
    if not r["ok"]:
        print("FAILED rc=", r["rc"])
        print("CMD:", r.get("cmd"))
        print("STDOUT:\n", r["stdout"])
        print("STDERR:\n", r["stderr"])
        return

    os.makedirs(TRACE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(TRACE_DIR, f"{module}__ncu_full{tag}_b{batch}n{n}__{stamp}.ncu-rep")
    with open(path, "wb") as f:
        f.write(r["report"])
    with open(path.replace(".ncu-rep", "__details.txt"), "w") as f:
        f.write(r["details"])
    with open(path.replace(".ncu-rep", "__raw.csv"), "w") as f:
        f.write(r["csv"])
    print(r["stdout"][-1500:])
    print("report  ->", path)
    print("details ->", path.replace(".ncu-rep", "__details.txt"))
    print("raw csv ->", path.replace(".ncu-rep", "__raw.csv"))
