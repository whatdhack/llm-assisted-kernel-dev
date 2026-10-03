import modal

app = modal.App("test-ncu-profiling")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-base-ubuntu22.04", add_python="3.11"
    )
    .run_commands(
        "apt-get update -qq && apt-get install -y wget",
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update -qq",
        "apt-get install -y nsight-compute-2025.1.1",
        "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128",
    )
)


@app.function(gpu="B200", image=image, timeout=180)
def test_ncu():
    import subprocess, shutil, os, glob

    # Find ncu binary (nsight-compute installs under /opt/nvidia/nsight-compute/)
    candidates = glob.glob("/opt/nvidia/nsight-compute/*/ncu")
    ncu_path = shutil.which("ncu") or (candidates[0] if candidates else None)
    print(f"ncu path: {ncu_path}")
    print(f"all candidates: {candidates}")
    if not ncu_path:
        print("FAIL: ncu not found anywhere")
        return

    env = {**os.environ, "PATH": os.path.dirname(ncu_path) + ":" + os.environ.get("PATH", "")}

    r = subprocess.run([ncu_path, "--version"], capture_output=True, text=True, env=env)
    print("ncu --version:", r.stdout.strip())

    # Profile a trivial torch matmul kernel
    r2 = subprocess.run(
        [
            ncu_path, "--target-processes", "all", "--set", "basic",
            "python3", "-c",
            "import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x; torch.cuda.synchronize(); print('kernel done')",
        ],
        capture_output=True, text=True, timeout=120, env=env,
    )
    print("\n--- ncu profile run ---")
    print("stdout:", r2.stdout[:3000])
    print("stderr:", r2.stderr[:2000])
    print("returncode:", r2.returncode)
    if r2.returncode == 0:
        print("\nRESULT: NCU PROFILING WORKS on Modal")
    else:
        print("\nRESULT: NCU PROFILING FAILED")


@app.local_entrypoint()
def main():
    test_ncu.remote()
