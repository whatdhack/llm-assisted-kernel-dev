"""Side-by-side Nsight Compute profiles of _blocked and _kloop on a B200.

Answers the standing open item ("the profile has moved twice since the last
.ncu-rep capture -- RE-PROFILE before choosing the next target") for both
kernels at once, at batch=1/n=8192.

WHY FOUR TARGETED INVOCATIONS RATHER THAN TWO. The two implementations do
not issue the same launch sequence, so "the first N launches" of each are
not comparable:

  _blocked  P S P S P S ...        every S is the FULL trailing update
  _kloop    P S P S ... P S_BIG    the first 15 S are NARROW phase-1 strip
                                   updates; the phase-2 rank-1024 update of
                                   the whole trailing block is syrk index 15

Profiling launch 0 of each would therefore put _blocked's full trailing SYRK
against _kloop's narrow strip SYRK and call it a comparison. So the SYRK
capture skips to index 15 for _kloop and 0 for _blocked, which puts the two
big trailing updates side by side. Panels need no skip -- both start at
bk=0 with the same BLOCK_I=32 grid, so launch 0 is directly comparable.

Every artifact (.ncu-rep, details page, source page, raw csv) is saved
locally so the numbers can be re-parsed without paying for another B200.

Usage: modal run ncu_compare_blocked_kloop_b200.py
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
        # MUST pin: the unversioned `nsight-compute` metapackage resolves to
        # 2022.4.1, which predates Blackwell and fails with "Profiling is not
        # supported on device 0" on sm_100.
        "apt-get install -y nsight-compute-2026.2.1",
        "apt-get install -y build-essential",   # triton JITs a C driver shim
        "pip install torch triton",
    )
    .add_local_dir(LOCAL_DIR, "/root/python_standalone")
)

app = modal.App("cholesky-ncu-syrk-only", image=image)

RUNNER = r'''
import sys
import torch
sys.path.insert(0, "/root/python_standalone")
import importlib
mod = importlib.import_module(sys.argv[1])
import bench_leaderboard

batch, n = int(sys.argv[2]), int(sys.argv[3])
A = bench_leaderboard.generate_input(batch, n, 2, 48192)
mod.custom_kernel(A)          # warmup: JIT everything before profiling starts
torch.cuda.synchronize()

torch.cuda.profiler.start()
L = mod.custom_kernel(A)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print("max_abs_err", (L @ L.transpose(-1, -2) - A).abs().amax().item())
'''

# (label, module, kernel regex, launch-skip, launch-count)
TARGETS = [
    ("blocked_syrk", "cholesky_gluon_tcgen05_blocked", "_syrk_kernel_tcgen05", 0, 3),
]


@app.function(gpu="B200", timeout=7200, max_containers=2)
def ncu_all(batch: int, n: int):
    import glob
    import shutil
    import subprocess
    import sys

    ncu = shutil.which("ncu") or next(iter(glob.glob("/opt/nvidia/nsight-compute/*/ncu")), None)
    assert ncu, "ncu not found"
    ver = subprocess.run([ncu, "--version"], capture_output=True, text=True).stdout.strip()

    with open("/root/runner.py", "w") as f:
        f.write(RUNNER)

    results = {"ncu_version": ver, "targets": {}}
    for label, module, kregex, skip, count in TARGETS:
        out = f"/tmp/{label}"
        cmd = [
            ncu, "-o", out, "-f",
            "--target-processes", "all",
            "--profile-from-start", "off",
            "--set", "full",
            "--import-source", "yes",
            "--kernel-name", f"regex:{kregex}",
            "--launch-skip", str(skip),
            "--launch-count", str(count),
            sys.executable, "/root/runner.py", module, str(batch), str(n),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3300)
        rep = out + ".ncu-rep"
        if not os.path.exists(rep):
            results["targets"][label] = {
                "ok": False, "rc": r.returncode, "cmd": " ".join(cmd),
                "stdout": r.stdout[-4000:], "stderr": r.stderr[-4000:]}
            continue
        det = subprocess.run([ncu, "--import", rep, "--page", "details"],
                             capture_output=True, text=True).stdout
        src = subprocess.run([ncu, "--import", rep, "--page", "source"],
                             capture_output=True, text=True).stdout
        csv = subprocess.run([ncu, "--import", rep, "--page", "raw", "--csv"],
                             capture_output=True, text=True).stdout
        with open(rep, "rb") as f:
            blob = f.read()
        results["targets"][label] = {"ok": True, "rc": r.returncode, "details": det,
                                     "source": src, "csv": csv, "report": blob,
                                     "stdout": r.stdout[-1500:]}
    return results


@app.local_entrypoint()
def main(batch: int = 1, n: int = 8192):
    r = ncu_all.remote(batch, n)
    print("ncu:", r["ncu_version"].replace("\n", " "))
    os.makedirs(TRACE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    saved = {}
    for label, t in r["targets"].items():
        if not t["ok"]:
            print(f"\n=== {label}: FAILED rc={t['rc']} ===")
            print("CMD:", t.get("cmd"))
            print("STDERR:\n", t["stderr"][-3000:])
            continue
        base = os.path.join(TRACE_DIR, f"cmp_{label}_b{batch}n{n}__{stamp}")
        with open(base + ".ncu-rep", "wb") as f:
            f.write(t["report"])
        for ext, key in ((".details.txt", "details"), (".source.txt", "source"),
                         (".raw.csv", "csv")):
            with open(base + ext, "w") as f:
                f.write(t[key])
        saved[label] = base
        print(f"saved {label:14} -> {base}.ncu-rep (+ .details.txt .source.txt .raw.csv)")

    print("\nAll artifacts are on disk; re-parse them locally without another B200 run.")
    if saved:
        print("\nparse with: python parse_ncu_compare.py " + " ".join(
            f"{k}={v}.raw.csv" for k, v in saved.items()))
