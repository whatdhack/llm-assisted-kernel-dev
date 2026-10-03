"""
Per-source-line time attribution for _syrk_tile_tcgen05, via ncu PC sampling.

ncu samples the warp scheduler's program counter at a fixed interval and
records, per PC, whether a warp was issuing or which stall reason it was in.
Aggregated per source line that gives "% of sampled cycles spent at this
statement" -- the closest thing to a per-statement profile on a GPU.

Triton emits .loc line info, so the samples map back to the .py source
(_syrk_tile_tcgen05 is a @gluon.jit helper inlined into the kernel).

Usage: modal run ncu_source_syrk_b200.py
"""
import os
import tempfile

import modal

import os as _os
# Kernel sources live one level up, in the cholesky_py problem directory.
LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-base-ubuntu24.04", add_python="3.12")
    .run_commands(
        "apt-get update -qq",
        "apt-get install -y nsight-compute-2026.2.1",
        "apt-get install -y build-essential",
        "pip install torch triton",
    )
    .add_local_dir(LOCAL_DIR, "/root/python_standalone")
)

app = modal.App("cholesky-ncu-source-b200", image=image)

RUNNER = r'''
import sys, torch
sys.path.insert(0, "/root/python_standalone")
import importlib
mod = importlib.import_module("cholesky_gluon_tcgen05_blocked")
import bench_leaderboard
A = bench_leaderboard.generate_input(1, 8192, 2, 48192)
mod.custom_kernel(A)
torch.cuda.synchronize()
torch.cuda.profiler.start()
mod.custom_kernel(A)
torch.cuda.synchronize()
torch.cuda.profiler.stop()
'''


@app.function(gpu="B200", timeout=3600)
def source_profile():
    import glob
    import shutil
    import subprocess
    import sys

    ncu = shutil.which("ncu") or next(iter(glob.glob("/opt/nvidia/nsight-compute/*/ncu")), None)
    open("/root/runner.py", "w").write(RUNNER)

    out = "/tmp/src"
    cmd = [
        ncu, "-o", out, "-f", "--target-processes", "all",
        "--profile-from-start", "off",
        "--set", "full", "--import-source", "yes",
        "--kernel-name", "regex:_syrk_kernel_tcgen05",
        "--launch-count", "1",
        sys.executable, "/root/runner.py",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)
    rep = out + ".ncu-rep"
    if not os.path.exists(rep):
        return {"ok": False, "stdout": r.stdout[-4000:], "stderr": r.stderr[-4000:]}

    res = {"ok": True}
    # what page names does this ncu accept?
    h = subprocess.run([ncu, "--help"], capture_output=True, text=True).stdout
    res["page_help"] = "\n".join(l for l in h.splitlines() if "--page" in l or "source" in l.lower())[:1200]

    # the source page as csv: one row per source line with its metrics
    for page in ("source", "details"):
        p = subprocess.run([ncu, "--import", rep, "--page", page, "--csv"],
                           capture_output=True, text=True)
        res[f"csv_{page}_rc"] = p.returncode
        res[f"csv_{page}"] = p.stdout[:400000]
        res[f"csv_{page}_err"] = p.stderr[:800]
    return res


@app.local_entrypoint()
def main():
    import csv
    import io
    r = source_profile.remote()
    if not r["ok"]:
        print("PROFILE FAILED")
        print(r["stdout"][-2500:])
        print(r["stderr"][-2500:])
        return

    csv_txt = r.get("csv_source", "")
    scratch = os.environ.get("SCRATCH_DIR", os.path.join(tempfile.gettempdir(), "ncu_scratch"))
    os.makedirs(scratch, exist_ok=True)
    with open(f"{scratch}/syrk_source.csv", "w") as f:
        f.write(csv_txt)
    with open(f"{scratch}/syrk_details.csv", "w") as f:
        f.write(r.get("csv_details", ""))
    print("rc:", r.get("csv_source_rc"), " bytes:", len(csv_txt))
    print("saved ->", f"{scratch}/syrk_source.csv")
    print()
    print("--- first 12 raw lines ---")
    for i, line in enumerate(csv_txt.splitlines()[:12]):
        print(f"{i}: {line[:200]}")
