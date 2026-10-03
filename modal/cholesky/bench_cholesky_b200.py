"""
Runs the real bench_leaderboard.py suite (15 official GPU MODE cases) for
both starter.py (torch.linalg.cholesky_ex / cuSOLVER) and
cholesky_triton_blocked.py on an actual B200 via Modal, for direct
comparison against local GB10 numbers and the GPU MODE leaderboard's own
reported benchmark.

Usage: modal run bench_cholesky_b200.py
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

app = modal.App("cholesky-bench-b200", image=image)


def run_bench(module_name):
    import io
    import sys

    sys.path.insert(0, "/root/python_standalone")
    import bench_leaderboard

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        bench_leaderboard.main(module_name)
    finally:
        sys.stdout = old_stdout

    return buf.getvalue()


def run_all_benches():
    import torch
    import triton

    info = {
        "gpu_name": torch.cuda.get_device_name(0),
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
    }
    return {
        **info,
        "starter": run_bench("starter"),
        "cholesky_triton_blocked": run_bench("cholesky_triton_blocked"),
    }


@app.cls(gpu="B200", scaledown_window=60)
class BenchRunner:
    @modal.method()
    def run(self):
        return run_all_benches()


@app.local_entrypoint()
def main():
    runner = BenchRunner()
    result = runner.run.remote()
    print("=" * 70)
    print("gpu_name:", result["gpu_name"])
    print("triton_version:", result["triton_version"])
    print("torch_version:", result["torch_version"])
    print("=" * 70)
    print("--- starter.py (torch.linalg.cholesky_ex / cuSOLVER) ---")
    print(result["starter"])
    print("--- cholesky_triton_blocked.py ---")
    print(result["cholesky_triton_blocked"])
