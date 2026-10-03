"""
Extract the Source page (per-SASS-instruction PC sampling) from an EXISTING
.ncu-rep under outputs/traces/, so analysis is tied to a saved report rather
than a throwaway one inside a container.

`ncu --import` is pure post-processing, so this runs CPU-only -- no GPU.

Usage:
    modal run ncu_import_source.py
    modal run ncu_import_source.py --report <basename under outputs/traces>
"""
import os

import modal

import os as _os
import tempfile as _tempfile
# Kernel sources live one level up, in the cholesky_py problem directory.
LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
SCRATCH = _os.environ.get("SCRATCH_DIR",
                          _os.path.join(_tempfile.gettempdir(), "ncu_scratch"))

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-base-ubuntu24.04", add_python="3.12")
    .run_commands(
        "apt-get update -qq",
        "apt-get install -y nsight-compute-2026.2.1",
    )
    .add_local_dir(LOCAL_DIR, "/root/python_standalone")
)

app = modal.App("cholesky-ncu-import-source", image=image)

DEFAULT = "cholesky_gluon_tcgen05_blocked__ncu_full_splitbar_lazydesc_b1n8192__20260815-132402.ncu-rep"


@app.function(timeout=1800)          # no gpu=: --import is CPU-only
def extract(report: str):
    import glob
    import shutil
    import subprocess

    ncu = shutil.which("ncu") or next(iter(glob.glob("/opt/nvidia/nsight-compute/*/ncu")), None)
    rep = f"/root/python_standalone/outputs/traces/{report}"
    if not os.path.exists(rep):
        return {"ok": False, "msg": f"not found: {report}",
                "available": sorted(os.path.basename(p) for p in
                                    glob.glob("/root/python_standalone/outputs/traces/*.ncu-rep"))}

    p = subprocess.run([ncu, "--import", rep, "--page", "source", "--csv"],
                       capture_output=True, text=True)
    return {"ok": True, "size": os.path.getsize(rep), "rc": p.returncode,
            "csv": p.stdout, "err": p.stderr[:600]}


@app.local_entrypoint()
def main(report: str = DEFAULT):
    r = extract.remote(report)
    if not r["ok"]:
        print(r["msg"])
        for a in r["available"]:
            print("  ", a)
        return
    os.makedirs(SCRATCH, exist_ok=True)
    dst = os.path.join(SCRATCH, report.replace(".ncu-rep", "__source.csv"))
    with open(dst, "w") as f:
        f.write(r["csv"])
    print(f"report : {report}")
    print(f"size   : {r['size']:,} bytes    csv rc {r['rc']}, {len(r['csv']):,} bytes")
    if r["err"]:
        print("stderr :", r["err"][:300])
    print(f"saved  : {dst}")
