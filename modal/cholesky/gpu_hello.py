import modal
import torch

# Define the Modal App and specify the base image with torch installed
# You can choose a specific GPU type like "T4", "A10G", or "H100"
GPU_CONFIG = modal.App(
    "example-gpu-hello-world",
    image=modal.Image.debian_slim().pip_install("torch")
)

@GPU_CONFIG.function(gpu="T4") # Attach a T4 GPU to this function
def torch_cuda():
    """A simple function that runs on a GPU and prints device info."""
    print("Hello, world from the GPU!")
    # Check if CUDA is available and print the device properties
    if torch.cuda.is_available():
        print(f"CUDA is available. Device name: {torch.cuda.get_device_properties('cuda:0')}")
    else:
        print("CUDA is not available.")
    return "Function finished successfully."

@GPU_CONFIG.local_entrypoint()
def main():
    # This entrypoint will run the torch_cuda function remotely on Modal's cloud GPU
    result = torch_cuda.remote()
    print(f"Remote function output: {result}")

