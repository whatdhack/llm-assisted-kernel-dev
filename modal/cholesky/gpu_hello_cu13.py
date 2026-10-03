import modal
import os

# 1. Image with CUDA 13.0, CUTLASS DSL, and Python 3.12 (required for DSL 4.x)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04", 
        add_python="3.12"
    )
    .pip_install("nvidia-cutlass", "nvidia-cutlass-dsl", "torch")
    .env({"CUDA_HOME": "/usr/local/cuda", "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
)

app = modal.App("cutlass-dsl-tensor-add", image=image)

def run_add_logic(size: int = 1024):
    import torch
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    
    # Define the GPU Kernel using CuTe DSL
    @cute.kernel
    def add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
        # Get thread and block indices
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        
        # Simple 1D indexing for this hello-world example
        idx = bidx * bdim + tidx
        
        if idx < gA.shape[0]:
            # Perform element-wise addition
            gC[idx] = gA[idx] + gB[idx]

    # Host-side JIT function to launch the kernel
    @cute.jit
    def host_add(tA, tB, tC):
        threads_per_block = 256
        blocks = (tA.shape[0] + threads_per_block - 1) // threads_per_block
        
        add_kernel(tA, tB, tC).launch(
            grid=(blocks, 1, 1), 
            block=(threads_per_block, 1, 1)
        )

    # Prepare PyTorch tensors and convert to CuTe tensors via DLPack
    a = torch.ones(size, device="cuda")
    b = torch.ones(size, device="cuda") * 2
    c = torch.zeros(size, device="cuda")
    
    # from_dlpack enables zero-copy integration
    dA, dB, dC = from_dlpack(a), from_dlpack(b), from_dlpack(c)
    
    # Compile and Run
    compiled_fn = cute.compile(host_add, dA, dB, dC)
    compiled_fn(dA, dB, dC)
    
    return c.cpu().numpy()[:10] # Return first 10 results for verification

@app.cls(gpu="B200", scaledown_window=300)
class CutlassAddition:
    @modal.method()
    def run_add(self, size: int = 1024):
        return run_add_logic(size)

@app.local_entrypoint()
def main():
    if os.environ.get("MODAL_LOCAL_RUN"):
        print("Running locally...")
        result = run_add_logic(2048)
        print(f"Local Result: {result}")
    else:
        print("Running on Modal...")
        runner = CutlassAddition()
        result = runner.run_add.remote(2048)
        print(f"Modal Result: {result}")

if __name__ == "__main__":
    # If run directly as a script without modal run, or if forced local
    if os.environ.get("MODAL_LOCAL_RUN") or not os.environ.get("MODAL_TOKEN_ID"):
        print("Direct execution - Running locally...")
        result = run_add_logic(2048)
        print(f"Local Result: {result}")
    else:
        main()
