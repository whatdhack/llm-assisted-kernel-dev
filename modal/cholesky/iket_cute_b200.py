"""
IKET (In-Kernel Event Tracing) timeline of the CuTe DSL blocked Cholesky on a
B200: the iket ranges in cholesky_cute_tcgen05_blocked.py (panel_L11/stage/L21;
SYRK tmem_alloc, tma_issue, wait_L, split, mma, t2r_load, wait_A, epilogue,
tma_store, tmem_free) recorded per warp, per launch, back to back.

Only the first --steps bk steps are run by default (panel + SYRK each), IN SITU,
because a full n=8192 factorization is 511 launches of up to 8192 blocks and the
trace would be far too large for Perfetto. --steps 0 runs the whole
custom_kernel instead.

IKET cannot run alongside ncu or nsys, so this is a separate script from
ncu_cute_vs_gluon_b200.py. It is CuTe-only: Gluon kernels have no iket events.

Usage:
    modal run iket_cute_b200.py                       # b=1 n=8192, 4 bk steps
    modal run iket_cute_b200.py --steps 2 --postprocess perfetto

Open the saved *.pftrace at https://ui.perfetto.dev/
"""
import io
import os
import tarfile
import time

import modal

LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(LOCAL_DIR, "outputs", "traces")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .run_commands(
        "apt-get update -qq",
        "apt-get install -y build-essential",
        # Same versions as the ncu captures (torch resolved to 2.13.0 there
        # unpinned; pinned here so this image cannot drift from them).
        "pip install torch==2.13.0 triton==3.7.1 numpy",
        # run-iket and cutlass.cute.experimental.iket both ship in 4.7.1.
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

app = modal.App("cholesky-iket-cute")

RUNNER = r'''
import sys
import torch
import cutlass
sys.path.insert(0, "/root/python_standalone")
import cholesky_cute_tcgen05_blocked as mod
import bench_leaderboard
from cutlass.cute.runtime import from_dlpack

print("VERSIONS: torch", torch.__version__, "cutlass-dsl", cutlass.__version__,
      "gpu", torch.cuda.get_device_name(0), flush=True)
batch, n, steps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
A0 = bench_leaderboard.generate_input(batch, n, 2, 48192)

if steps <= 0:
    L = mod.custom_kernel(A0)
    torch.cuda.synchronize()
    print("max_abs_err", (L @ L.transpose(-1, -2) - A0).abs().amax().item())
else:
    # The same bk loop as custom_kernel, stopped after `steps` panel+SYRK pairs.
    A = A0.to(torch.float32).contiguous().clone()
    L = torch.zeros_like(A)
    tA = from_dlpack(A.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    tL = from_dlpack(L.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    panel, syrk = mod._get_compiled(tA, tL)
    NB, BM, BN = mod.NB, mod.BLOCK_M, mod.BLOCK_N
    k = min(steps * NB, n)
    for bk in range(0, k, NB):
        panel(tA, tL, bk, n, batch)
        trailing = n - bk - NB
        if trailing > 0:
            syrk(tL, tA, bk, bk + NB, (trailing + BM - 1) // BM,
                 (trailing + BN - 1) // BN, batch)
    torch.cuda.synchronize()
    ref = torch.linalg.cholesky(A0.double()).float()
    print("first", k, "columns vs cholesky: max_abs_diff",
          (L[..., :k] - ref[..., :k]).abs().amax().item(), flush=True)
'''


@app.function(gpu="B200", image=image, timeout=3600)
def iket_run(batch: int, n: int, steps: int, postprocess: str):
    import subprocess
    import sys

    with open("/root/runner.py", "w") as f:
        f.write(RUNNER)
    help_rc = subprocess.run(["run-iket", "--help"], capture_output=True).returncode
    assert help_rc == 0, "run-iket not found on PATH"

    out = "/tmp/iket_out"
    cmd = ["run-iket", "--output-dir", out, "--clobber",
           "profile", "--postprocess", postprocess, "--",
           sys.executable, "/root/runner.py", str(batch), str(n), str(steps)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3300)
    elapsed = time.time() - t0

    files = []
    blob = None
    if os.path.isdir(out):
        for root, _, names in os.walk(out):
            for name in names:
                p = os.path.join(root, name)
                files.append((os.path.relpath(p, out), os.path.getsize(p)))
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(out, arcname=".")
        blob = buf.getvalue()
    return {"rc": r.returncode, "elapsed_s": elapsed, "cmd": " ".join(cmd),
            "stdout": r.stdout[-6000:], "stderr": r.stderr[-6000:],
            "files": sorted(files), "tar": blob}


@app.local_entrypoint()
def main(batch: int = 1, n: int = 8192, steps: int = 4, postprocess: str = "all"):
    r = iket_run.remote(batch, n, steps, postprocess)
    print("cmd:", r["cmd"])
    print(f"rc={r['rc']}  elapsed={r['elapsed_s']:.0f}s")
    print("--- stdout (tail)"); print(r["stdout"][-3000:])
    if r["stderr"].strip():
        print("--- stderr (tail)"); print(r["stderr"][-3000:])
    if r["tar"] is None:
        print("NO OUTPUT DIRECTORY"); return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(TRACE_DIR, f"iket_cute_b{batch}n{n}_s{steps}__{stamp}")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(r["tar"]), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    print("saved ->", dest)
    for rel, size in r["files"]:
        print(f"  {size:>14,}  {rel}")
