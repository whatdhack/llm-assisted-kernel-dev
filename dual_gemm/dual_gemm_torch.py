import torch
import os
import modal
import dataclasses
import math
import time
from task import input_t, output_t
from utils import make_match_reference

# 1. Define Modal Image with PyTorch Nightly and dependencies
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "torch",
        pre=True,
        index_url="https://download.pytorch.org/whl/nightly/cu130"
    )
    .pip_install("numpy")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib",
        "LD_PRELOAD": "/usr/local/cuda/lib64/libcublas.so:/usr/local/cuda/lib64/libcublasLt.so"
    })
    .add_local_file("task.py", remote_path="/root/task.py")
    .add_local_file("utils.py", remote_path="/root/utils.py")
)

app = modal.App("dual-gemm-torch", image=image)

# Scaling factor vector size
sf_vec_size = 16

@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float

def calculate_stats(durations: list[float]):
    runs = len(durations)
    if runs == 0:
        return None
    total = sum(durations)
    best = min(durations)
    worst = max(durations)

    avg = total / runs
    if runs > 1:
        variance = sum(map(lambda x: (x - avg) ** 2, durations))
        std = math.sqrt(variance / (runs - 1))
        err = std / math.sqrt(runs)
    else:
        std = 0
        err = 0

    return Stats(
        runs=runs, mean=avg, std=std, err=err, best=float(best), worst=float(worst)
    )

# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b

# Helper function to convert scale factor tensor to blocked format
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape

    # Please ensure rows and cols are multiples of 128 and 4 respectively
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    # Use reshape to handle non-contiguous input_matrix
    blocks = padded.reshape(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def ref_kernel(
    data: input_t,
    combine_weights: bool = True,
    split_k: int = 1
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled dual GEMM with silu activation,
    C = silu(A @ B1) * (A @ B2).
    Supports combining weights and Split-K.
    """
    a_ref, b1_ref, b2_ref, sfa_ref, sfb1_ref, sfb2_ref, _, _, _, c_ref = data

    # Get dimensions from MxNxL layout
    m, n, l = c_ref.shape
    k_half = a_ref.shape[1]

    sf_k = sfa_ref.shape[1]

    k_half_per_split = k_half // split_k
    sf_k_per_split = sf_k // split_k

    if combine_weights:
        # Combine B1 and B2 along the N dimension: (2*n, k_half, l)
        b_combined = torch.cat([b1_ref.view(torch.int8), b2_ref.view(torch.int8)], dim=0).view(torch.float4_e2m1fn_x2)
        # Combine scale factors along the N dimension: (2*n, sf_k, l)
        sfb_combined = torch.cat([sfb1_ref, sfb2_ref], dim=0)
        ref_combined_acc = torch.zeros((m, 2 * n, l), dtype=torch.float32, device="cuda")
    else:
        ref1_acc = torch.zeros((m, n, l), dtype=torch.float32, device="cuda")
        ref2_acc = torch.zeros((m, n, l), dtype=torch.float32, device="cuda")

    for l_idx in range(l):
        for sk in range(split_k):
            k_start = sk * k_half_per_split
            k_end = (sk + 1) * k_half_per_split
            sf_k_start = sk * sf_k_per_split
            sf_k_end = (sk + 1) * sf_k_per_split

            # Slice A and SFA. Use int8 view to avoid "copy_ not implemented" error for float4_e2m1fn_x2
            a_chunk = a_ref.view(torch.int8)[:, k_start:k_end, l_idx].contiguous().view(torch.float4_e2m1fn_x2)
            sfa_chunk = to_blocked(sfa_ref[:, sf_k_start:sf_k_end, l_idx]).cuda()

            if combine_weights:
                # Slice combined B and SFB
                b_chunk = b_combined.view(torch.int8)[:, k_start:k_end, l_idx].contiguous().view(torch.float4_e2m1fn_x2)
                sfb_chunk = to_blocked(sfb_combined[:, sf_k_start:sf_k_end, l_idx]).cuda()

                # (m, k_chunk) @ (2*n, k_chunk).T -> (m, 2*n)
                res_combined = torch._scaled_mm(
                    a_chunk,
                    b_chunk.transpose(0, 1),
                    sfa_chunk,
                    sfb_chunk,
                    bias=None,
                    out_dtype=torch.float32,
                )
                ref_combined_acc[:, :, l_idx] += res_combined
            else:
                # Process B1 and B2 separately
                b1_chunk = b1_ref.view(torch.int8)[:, k_start:k_end, l_idx].contiguous().view(torch.float4_e2m1fn_x2)
                sfb1_chunk = to_blocked(sfb1_ref[:, sf_k_start:sf_k_end, l_idx]).cuda()

                res1 = torch._scaled_mm(
                    a_chunk,
                    b1_chunk.transpose(0, 1),
                    sfa_chunk,
                    sfb1_chunk,
                    bias=None,
                    out_dtype=torch.float32,
                )
                ref1_acc[:, :, l_idx] += res1

                b2_chunk = b2_ref.view(torch.int8)[:, k_start:k_end, l_idx].contiguous().view(torch.float4_e2m1fn_x2)
                sfb2_chunk = to_blocked(sfb2_ref[:, sf_k_start:sf_k_end, l_idx]).cuda()

                res2 = torch._scaled_mm(
                    a_chunk,
                    b2_chunk.transpose(0, 1),
                    sfa_chunk,
                    sfb2_chunk,
                    bias=None,
                    out_dtype=torch.float32,
                )
                ref2_acc[:, :, l_idx] += res2

    if combine_weights:
        ref1 = ref_combined_acc[:, :n, :]
        ref2 = ref_combined_acc[:, n:, :]
    else:
        ref1 = ref1_acc
        ref2 = ref2_acc

    # Do silu on the first GEMM result and multiply with the second GEMM result
    c_ref = (torch.nn.functional.silu(ref1) * ref2).to(torch.float16)
    return c_ref


def generate_input(
    m: int,
    n: int,
    k: int,
    l: int,
    seed: int,
):
    """
    Generate input tensors for NVFP4 block-scaled dual GEMM with silu activation,
    C = silu(A @ B1) * (A @ B2).
    """
    torch.manual_seed(seed)

    # Generate uint8 tensor, then convert to float4e2m1fn_x2 data type
    a_ref = torch.randint(
        -6, 6, (l, m, k // 2), dtype=torch.int8, device="cuda"
    ).permute(1, 2, 0)
    b1_ref = torch.randint(
        -6, 6, (l, n, k // 2), dtype=torch.int8, device="cuda"
    ).permute(1, 2, 0)
    b2_ref = torch.randint(
        -6, 6, (l, n, k // 2), dtype=torch.int8, device="cuda"
    ).permute(1, 2, 0)
    a_ref = a_ref.view(torch.float4_e2m1fn_x2)
    b1_ref = b1_ref.view(torch.float4_e2m1fn_x2)
    b2_ref = b2_ref.view(torch.float4_e2m1fn_x2)

    # Create float16 output tensor
    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(
        1, 2, 0
    )

    def create_scale_factor_tensors(l, mn, sf_k):
        ref_shape = (l, mn, sf_k)
        ref_permute_order = (1, 2, 0)
        ref_f8_random_int = torch.randint(-3, 3, ref_shape, dtype=torch.int8, device='cuda')
        ref_f8_torch_tensor = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
        ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(*ref_permute_order)

        atom_m = (32, 4)
        atom_k = 4
        mma_shape = (
            l,
            ceil_div(mn, atom_m[0] * atom_m[1]),
            ceil_div(sf_k, atom_k),
            atom_m[0],
            atom_m[1],
            atom_k,
        )

        mma_permute_order = (3, 4, 1, 5, 2, 0)
        rand_int_tensor = torch.randint(-3, 3, mma_shape, dtype=torch.int8, device='cuda')
        reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
        reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(*mma_permute_order)

        i_idx = torch.arange(mn, device='cuda')
        j_idx = torch.arange(sf_k, device='cuda')
        b_idx = torch.arange(l, device='cuda')

        i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing='ij')

        mm = i_grid // (atom_m[0] * atom_m[1])
        mm32 = i_grid % atom_m[0]
        mm4 = (i_grid % 128) // atom_m[0]
        kk = j_grid // atom_k
        kk4 = j_grid % atom_k

        reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_torch_tensor_permuted[i_grid, j_grid, b_grid]

        return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor

    sf_k = ceil_div(k, sf_vec_size)
    sfa_ref_cpu, sfa_ref_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb1_ref_cpu, sfb1_ref_permuted = create_scale_factor_tensors(l, n, sf_k)
    sfb2_ref_cpu, sfb2_ref_permuted = create_scale_factor_tensors(l, n, sf_k)

    return (a_ref, b1_ref, b2_ref, sfa_ref_cpu.to("cuda"), sfb1_ref_cpu.to("cuda"), sfb2_ref_cpu.to("cuda"), sfa_ref_permuted, sfb1_ref_permuted, sfb2_ref_permuted, c_ref)


def custom_kernel(
    data: input_t,
    combine_weights: bool = True,
    split_k: int = 1
) -> output_t:
    """
    Custom kernel that calls the ref_kernel with specified arguments.
    """
    return ref_kernel(data, combine_weights=combine_weights, split_k=split_k)

def _clone_data(data):
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    elif isinstance(data, list):
        return [_clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data

def benchmark_one_config(
    m, n, k, l, seed, combine_weights, split_k,
    max_repeats=200, max_time_ns=1e10
):
    from utils import clear_l2_cache, make_match_reference
    NUM_ITERATIONS_PER_BENCHMARK = 50
    
    # 1. Generate multiple sets of input data to average over different random values
    data_list = []
    current_seed = seed
    for _ in range(NUM_ITERATIONS_PER_BENCHMARK):
        data_list.append(generate_input(m, n, k, l, current_seed))
        current_seed += 42
    
    # Keep a copy for potential rechecks
    check_copy = _clone_data(data_list)
    
    # 2. Obligatory correctness check
    check_implementation = make_match_reference(ref_kernel, rtol=1e-03, atol=1e-03)
    outputs = []
    try:
        for data in data_list:
            output = custom_kernel(_clone_data(data), combine_weights=combine_weights, split_k=split_k)
            outputs.append(output)
    except Exception as E:
        print(f"  [Error] Kernel execution failed: {E}")
        return None
        
    for ref_in, cust_out in zip(data_list, outputs):
        good, message = check_implementation(ref_in, cust_out)
        if not good:
            print(f"  [Error] Correctness check failed: {message}")
            return None

    # 3. Timing runs
    durations = []
    bm_start_time = time.perf_counter_ns()
    
    # Warmup
    for data in data_list[:3]: # simple warmup
        custom_kernel(data, combine_weights=combine_weights, split_k=split_k)
    torch.cuda.synchronize()

    for i in range(max_repeats):
        clear_l2_cache()
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for data in data_list:
            custom_kernel(data, combine_weights=combine_weights, split_k=split_k)
        end_event.record()
        
        torch.cuda.synchronize()
        # Avg duration per iteration in nanoseconds
        duration = (start_event.elapsed_time(end_event) / NUM_ITERATIONS_PER_BENCHMARK) * 1e6
        durations.append(duration)

        total_bm_duration = time.perf_counter_ns() - bm_start_time
        if i > 1 and total_bm_duration > 1e8:  # at least 2 runs, and 100ms total
            stats = calculate_stats(durations)
            # stop if error < 0.1%, or exceeded per-kernel time limit, or 2min wallclock
            if (stats.err / stats.mean < 0.001 or total_bm_duration > 120e9):
                break

    return calculate_stats(durations)

def run_dual_gemm_benchmark():
    test_specs = [
        {"m": 256, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
        {"m": 512, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
        {"m": 256, "n": 3072, "k": 4096, "l": 1, "seed": 1111},
        {"m": 512, "n": 3072, "k": 7168, "l": 1, "seed": 1111},
    ]

    all_results = []

    for spec in test_specs:
        m, n, k, l, seed = spec["m"], spec["n"], spec["k"], spec["l"], spec["seed"]
        print(f"\n--- Testing spec: M={m}, N={n}, K={k}, L={l} ---")
        
        # Explicit warmup for the spec
        print(f"  Performing warmup for M={m}, N={n}, K={k}...")
        benchmark_one_config(m, n, k, l, seed, True, 1, max_repeats=1)
        torch.cuda.synchronize()
        
        configs = [
            (True, 1),
            (False, 1),
            (True, 2),
            (False, 4)
        ]

        for ck, sk in configs:
            print(f"  Config: combine_weights={ck}, split_k={sk}")
            stats = benchmark_one_config(m, n, k, l, seed, ck, sk)
            if stats:
                tflops = (2 * m * n * k) / (stats.mean * 1e-9) / 1e12
                print(f"    Mean: {stats.mean/1e3:.2f} us, TFLOPS: {tflops:.2f}")
                all_results.append({
                    "spec": f"M={m},N={n},K={k}",
                    "ck": ck,
                    "sk": sk,
                    "tflops": tflops,
                    "mean_us": stats.mean / 1e3
                })

    print("\n" + "="*80)
    print(f"{'Spec':<20} | {'Combine':<8} | {'Split-K':<8} | {'TFLOPS':<10} | {'Mean (us)':<10}")
    print("-"*80)
    
    tflops_values = []
    time_values_us = []
    best_per_spec = {}

    for res in all_results:
        print(f"{res['spec']:<20} | {str(res['ck']):<8} | {res['sk']:<8} | {res['tflops']:<10.2f} | {res['mean_us']:<10.2f}")
        tflops_values.append(res['tflops'])
        time_values_us.append(res['mean_us'])
        
        spec_name = res['spec']
        if spec_name not in best_per_spec or res['tflops'] > best_per_spec[spec_name]['tflops']:
            best_per_spec[spec_name] = res

    if tflops_values:
        log_sum_tflops = sum(math.log(x) for x in tflops_values)
        geomean_tflops = math.exp(log_sum_tflops / len(tflops_values))
        
        log_sum_time = sum(math.log(x) for x in time_values_us)
        geomean_time = math.exp(log_sum_time / len(time_values_us))
        
        print("-"*80)
        print(f"{'GEOMETRIC MEAN':<42} | {geomean_tflops:<10.2f} | {geomean_time:<10.2f}")
    
    print("\n" + "="*80)
    print(f"{'Best Configs per Spec':^80}")
    print("-"*80)
    print(f"{'Spec':<20} | {'Combine':<8} | {'Split-K':<8} | {'TFLOPS':<10} | {'Mean (us)':<10}")
    print("-"*80)
    for spec, best in best_per_spec.items():
        print(f"{spec:<20} | {str(best['ck']):<8} | {best['sk']:<8} | {best['tflops']:<10.2f} | {best['mean_us']:<10.2f}")
    print("="*80)

@app.cls(gpu="B200", scaledown_window=300)
class DualGemmTorch:
    @modal.method()
    def run_benchmark(self):
        run_dual_gemm_benchmark()

@app.local_entrypoint()
def main():
    print("Running on Modal...")
    runner = DualGemmTorch()
    runner.run_benchmark.remote()

if __name__ == "__main__":
    print("Running locally (default)...")
    run_dual_gemm_benchmark()
