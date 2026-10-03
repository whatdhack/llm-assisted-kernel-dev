"""Export the SASS source page from an existing .ncu-rep for opcode counting."""
import modal, os
import os as _os
# Kernel sources live one level up, in the cholesky_py problem directory.
LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
TRACE_DIR = os.path.join(LOCAL_DIR, "outputs", "traces")
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-base-ubuntu24.04", add_python="3.12")
         .run_commands("apt-get update -qq", "apt-get install -y nsight-compute-2026.2.1",
                       "apt-get install -y build-essential", "pip install torch triton")
         .add_local_dir(LOCAL_DIR, "/root/python_standalone"))
app = modal.App("ncu-source-export", image=image)

@app.function(gpu="B200", timeout=1800)
def export(name: str):
    import glob, shutil, subprocess
    ncu = shutil.which("ncu") or next(iter(glob.glob("/opt/nvidia/nsight-compute/*/ncu")), None)
    rep = f"/root/python_standalone/outputs/traces/{name}"
    return subprocess.run([ncu, "--import", rep, "--page", "source"],
                          capture_output=True, text=True).stdout

@app.local_entrypoint()
def main(name: str):
    out = export.remote(name)
    p = os.path.join(TRACE_DIR, name.replace(".ncu-rep", ".source.txt"))
    open(p, "w").write(out)
    print(f"{len(out)} bytes -> {p}")
