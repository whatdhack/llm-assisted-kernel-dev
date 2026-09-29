# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import dataclasses
import math
import time
import os

# Support local compilation testing for B200 (sm_100a)
# Set COMPILE_LOCALLY_FOR_B200=1 to compile for sm_100a on local hardware (e.g., sm_121)
# This catches compilation errors without Modal costs
if os.environ.get("COMPILE_LOCALLY_FOR_B200"):
    print("🔧 Local compilation mode: Compiling for sm_100a (B200 architecture)")
    os.environ["CUTE_DSL_ARCH"] = "sm_100a"
elif not os.environ.get("CUTE_DSL_ARCH"):
    # Default to sm_100a for B200
    os.environ["CUTE_DSL_ARCH"] = "sm_100a"

from typing import Type, Tuple, Union

import cuda.bindings.driver as cuda
import torch
import modal

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack

"""
This example provides an experimental implementation of the SM100 batched dual dense blockscaled GEMM kernel with TMA prefetch support, please note that the APIs and implementation details related to this kernel may change in future releases.

A high-performance persistent batched dual dense blockscaled GEMM example for the NVIDIA Blackwell SM100 architecture
using CUTE DSL with TMA prefetch support.

- Matrix A is MxKxL, L is batch dimension, A can only be row-major("K") for NVF4 input type
- Matrix B1 is NxKxL, L is batch dimension, B1 can only be row-major("K") for NVF4 input type
- Matrix B2 is NxKxL, L is batch dimension, B2 can only be row-major("K") for NVF4 input type
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk, which has M×ceil_div(K, sf_vec_size)×L elements respectively
- Matrix SFB1 layout is filled internally according to B1 shape and BlockScaledBasicChunk, which has N×ceil_div(K, sf_vec_size)×L elements respectively
- Matrix SFB2 layout is filled internally according to B2 shape and BlockScaledBasicChunk, which has N×ceil_div(K, sf_vec_size)×L elements respectively

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2cta mma instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma
    - Support TMA prefetch for improved memory latency hiding
    - Fused SiLU activation and element-wise multiplication in epilogue

TMA Prefetch Configuration:
    The ``--prefetch_dist`` parameter controls TMA prefetch behavior:
    - Default (not specified): Uses num_ab_stage as prefetch distance for optimal pipeline utilization
    - 0: Disables TMA prefetch entirely
    - >0: Uses the specified value as explicit prefetch distance

    TMA prefetch issues prefetch hints before the actual TMA load operations to hide memory latency.
    Both initial prefetch (before mainloop) and rolling prefetch (inside mainloop) use the same
    prefetch distance for unified control.

This GEMM works as follows:
1. DMA warp: Load A, B1, B2 matrices and scale factors from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp:
    - Load scale factor A, B1, B2 from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp instruction.
    - Perform two matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction (A x B1 and A x B2).
3. EPILOGUE warp:
    - Load both completed accumulators from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Apply fused SiLU activation and elementwise multiplication: C = SiLU(Acc1) * Acc2
    - Type convert C matrix to output type.
    - Optionally store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations,
      or directly store C matrix from registers (RMEM) to global memory (GMEM) without TMA operations.

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below Sm100BlockScaledPersistentDenseGemmKernel class documentation
* A/B1/B2 tensors must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 64/128/192/256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B1/B2/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.
"""

# 1. Define Modal Image with PyTorch Nightly and dependencies
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "torch",
        pre=True,
        index_url="https://download.pytorch.org/whl/nightly/cu130"
    )
    .pip_install("nvidia-cutlass", "nvidia-cutlass-dsl", "numpy")
    .run_commands(
        "apt-get update",
        "apt-get install -y cuda-sanitizer-13-0 cuda-nsight-systems-13-0"  # Install compute-sanitizer and nsys
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib",
        "LD_PRELOAD": "/usr/local/cuda/lib64/libcublas.so:/usr/local/cuda/lib64/libcublasLt.so",
        "CUTE_DSL_ARCH": "sm_100a",
        # "CUDA_LAUNCH_BLOCKING": "1"  # Report CUDA errors synchronously at exact location
    })
    .add_local_file("reference-kernels/problems/nvidia/nvfp4_dual_gemm/task.py", remote_path="/root/task.py")
    .add_local_file("reference-kernels/problems/nvidia/utils.py", remote_path="/root/utils.py")
    .add_local_file("reference-kernels/problems/nvidia/nvfp4_dual_gemm/reference.py", remote_path="/root/reference.py")
    .add_local_file("reference-kernels/problems/nvidia/nvfp4_dual_gemm/dual_gemm_opt_cute.py", remote_path="/root/dual_gemm_opt_cute.py")
)

app = modal.App("dual-gemm-cute-opt", image=image)

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

NUM_ITERATIONS_PER_BENCHMARK = 50

_compiled_kernel_cache = None

@cute.jit
def gemm_entry(
    a_ptr: cute.Pointer,
    b1_ptr: cute.Pointer,
    b2_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb1_ptr: cute.Pointer,
    sfb2_ptr: cute.Pointer,
    c: cute.Pointer, # Changed to match make_ptr return type
    m: int,
    n: int,
    k: int,
    l: int,
):
    """
    JIT Entry point for the dual GEMM kernel.
    Accepts raw pointers for inputs and CuTe Tensor for output (to carry TMA metadata).
    """
    # Instantiate the kernel class
    # Note: These parameters must match what we compile with
    gemm_kernel = Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size=16,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
    )


    
    # A (M, K, L) - K-major (RowMajor if M,K?)
    # K-major usually means stride-1 on K.
    # Layout((M,K,L), (K,1,M*K)) -> (m,k,l) -> m*K + k*1. This is RowMajor.
    a_stride = (k, 1, m * k)
    a_layout = cute.make_layout((m, k, l), stride=a_stride)
    a_tensor = cute.make_tensor(a_ptr, a_layout)
    
    # B (N, K, L) - K-major
    b_stride = (k, 1, n * k)
    b_layout = cute.make_layout((n, k, l), stride=b_stride)
    b1_tensor = cute.make_tensor(b1_ptr, b_layout)
    b2_tensor = cute.make_tensor(b2_ptr, b_layout)
    
    # C (M, N, L) - N-major (RowMajor)
    # Re-create tensor from pointer since we pass raw pointer now
    c_stride = (n, 1, m * n)
    c_layout = cute.make_layout((m, n, l), stride=c_stride)
    c_tensor = cute.make_tensor(c, c_layout)
    
    # SFA (M, K, L) -> BlockScaled
    # The kernel re-creates SFA layout using `tile_atom_to_shape_SF`.
    # But we need to pass a tensor with *some* layout to `__call__` so it can use `.iterator`.
    # The kernel's `__call__` does:
    #   sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
    #   sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
    # So we just need to pass a dummy layout that points to the data correctly?
    # Actually SFA passed in is packed.
    # Let's treat it as flat raw pointer since the kernel re-makes the tensor.
    sfa_tensor = cute.make_tensor(sfa_ptr, cute.make_layout((1,)))
    sfb1_tensor = cute.make_tensor(sfb1_ptr, cute.make_layout((1,)))
    sfb2_tensor = cute.make_tensor(sfb2_ptr, cute.make_layout((1,)))
    


    # Calculate grid size based on device properties to ensure full occupancy
    # We want to launch enough clusters to fill all SMs on the GPU.
    # Persistent Tile Scheduler will distribute work among these persistent clusters.
    try:
        device_props = torch.cuda.get_device_properties(0)
        sm_count = device_props.multi_processor_count
        # Each cluster uses cluster_shape_mn[0] * cluster_shape_mn[1] SMs
        # For (1, 1) cluster, it's 1 SM per cluster.
        # We launch enough clusters to cover all SMs.
        cluster_size = 1 * 1 # Hardcoded here since gemm_kernel below uses (1,1)
        # Note: If we change cluster_shape_mn below, we must update this too.
        max_active_clusters = sm_count // cluster_size
        # print(f"DEBUG: Dynamic Grid Calculation: SMs={sm_count}, ClusterSize={cluster_size}, MaxClusters={max_active_clusters}")
    except Exception as e:
        # Fallback if torch.cuda not ready or error
        print(f"WARNING: Could not get device properties: {e}. Defaulting to 132 SMs (B200).")
        max_active_clusters = 132

    # Run kernel
    gemm_kernel(
        a_tensor, b1_tensor, b2_tensor,
        sfa_tensor, sfb1_tensor, sfb2_tensor,
        c_tensor,
        max_active_clusters, 
    )

def compile_kernel():
    global _compiled_kernel_cache
    if _compiled_kernel_cache is not None:
        return _compiled_kernel_cache
        
    print("Compiling kernel...")
    
    # Create dummy pointers for compilation
    from cutlass.cute.runtime import make_ptr
    
    # Types must match standard configuration
    # A/B: Float4E2M1FN
    # SF: Float8E4M3FN
    # C: Float16
    
    # Create aligned tensors using cutlass_torch to match Base file logic
    # Use dummy shapes for compilation (symbols will handle dynamic sizes)
    dummy_m, dummy_n, dummy_k, dummy_l = 128, 128, 128, 1
    
    # Check default c_major in Base file is "n" (Row Major)
    c_major = "n"
    
    # Create C tensor dummy
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    
    a_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    sf_ptr = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    # c_ptr replaced above
    # All pointers + 4 ints + xtream
    
    # We compile with specific argument types
    # Note: compilation runs offline (conceptually), we don't need real data
    
    func = cute.compile(
        gemm_entry,
        a_ptr, b_ptr, b_ptr,
        sf_ptr, sf_ptr, sf_ptr,
        c_ptr,
        0, 0, 0, 0, # dummy ints for m, n, k, l
    )
    
    _compiled_kernel_cache = func
    print("Compilation complete.")
    return func

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

class Sm100BlockScaledPersistentDenseGemmKernel:
    """This class implements batched matrix multiplication (C = SiLU(A x SFA x B1 x SFB1) * (A x SFA x B2 x SFB2)) 
    with support for various data types and architectural features specific to Blackwell GPUs 
    with persistent tile scheduling and warp specialization.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensor must have the same data type
        - i.e., Float8E4M3FN for A and Float8E5M2 for B is not supported

    :note: Supported combinations of A/B data types, SF data typs and SF vector size:
        - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - MXF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU + sf_vec_size: 32
        - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN + sf_vec_size: 16

    :note: Supported accumulator data types:
        - Float32

    :note: Supported C data types:
        - Float32
        - Float16/BFloat16
        - Float8E4M3FN/Float8E5M2
    :note: Constraints:
        - MMA tiler M must be 128 or 256 (use_2cta_instrs)
        - MMA tiler N must be 64/128/192/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Also, Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors

    Example:
        >>> gemm = Sm100BlockScaledPersistentDenseGemmKernel(
        ...     sf_vec_size=16,
        ...     mma_tiler_mn=(256, 128),
        ...     cluster_shape_mn=(2, 1)
        ... )
        >>> gemm(a_tensor, b1_tensor, b2_tensor, sfa_tensor, sfb1_tensor, sfb2_tensor, c_tensor, max_active_clusters, xtream)
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        prefetch_dist: Union[int, None] = None,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel with TMA prefetch support.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data types for MMA accumulator, always set to Float32
            - sf_vec_size: Scalefactor A/B vector size.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3. TMA Prefetch:
            - prefetch_dist: Prefetch distance for TMA operations.
              None = use num_ab_stage (default), 0 = disable prefetch, >0 = explicit distance.

        :param sf_vec_size: Scalefactor vector size.
        :type sf_vec_size: int
        :param mma_tiler_mn: Tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: Tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: Tuple[int, int]
        :param prefetch_dist: Prefetch distance for TMA operations (None=auto, 0=disable, >0=explicit).
        :type prefetch_dist: Union[int, None]
        """
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)

        # Prefetch configuration: None=auto (num_ab_stage), 0=disable, >0=explicit distance
        self.prefetch_dist_param = prefetch_dist

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # Set specialized warp ids
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        # Set barrier id for epilogue sync and tmem ptr sync
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B/SFA/SFB
        - Computing epilogue subtile
        - Setting up A/B/SFA/SFB/C stage counts in shared memory
        - Computing A/B/SFA/SFB/C shared memory layout
        """
        # Compute mma instruction shapes
        # (MMA_Tile_Shape_M, MMA_Tile_Shape_N, MMA_Inst_Shape_K)
        self.mma_inst_shape_mn = (
            self.mma_tiler[0],
            self.mma_tiler[1],
        )
        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b1_mcast = self.num_mcast_ctas_b > 1
        self.is_b2_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        # Note: In dual GEMM, we have B1, B2, SFB1, SFB2.
        # This will affect the shared memory capacity and stage calculation.
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
        )
        print(f"  [DEBUG] num_ab_stage: {self.num_ab_stage}, num_acc_stage: {self.num_acc_stage}, num_c_stage: {self.num_c_stage}, occupancy: {self.occupancy}")

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b1_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.b2_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb1_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb2_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Overlap and double buffer accumulator when num_acc_stage == 1 for cta_tile_n = 256 case
        # Overlap and double buffer accumulator when num_acc_stage == 1 for cta_tile_n = 256 case
        # Note: Disabled for Dual GEMM to avoid complex address logic causing illegal memory access
        # Disabled to prevent TMEM Overflow:
        # With overlap: TOTAL = 720 cols. Capacity = 512 cols.
        self.overlapping_accum = False # self.num_acc_stage == 1

        # Compute number of TMEM columns for SFA/SFB/Accumulator
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        
        # Dual GEMM: 1 SFA, 2 SFB
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + 2 * self.num_sfb_tmem_cols
        
        # Dual GEMM: 2 Accumulators
        self.num_accumulator_tmem_cols = 2 * (self.cta_tile_shape_mnk[1] * self.num_acc_stage if not self.overlapping_accum else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols)

        # Only when overlapping_accum is enabled, we need to release accumulator buffer early in epilogue
        self.iter_acc_early_release_in_epilogue = self.num_sf_tmem_cols // self.epi_tile_n

        # Set prefetch distance for both initial and rolling prefetch (unified control)
        # None = use num_ab_stage (default), 0 = disable prefetch, >0 = explicit distance
        if self.prefetch_dist_param is None:
            self.prefetch_dist = self.num_ab_stage
        else:
            self.prefetch_dist = self.prefetch_dist_param
        
        # Check if prefetch is enabled (prefetch_dist > 0)
        self.prefetch_enabled = self.prefetch_dist > 0

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,
        b1_tensor: cute.Tensor,
        b2_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb1_tensor: cute.Tensor,
        sfb2_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        epilogue_op: cutlass.Constexpr = lambda x: x
        * (1.0 / (1.0 + cute.math.exp(-x, fastmath=True))),
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a_tensor: Input tensor A
        :type a_tensor: cute.Tensor
        :param b1_tensor: Input tensor B1
        :type b1_tensor: cute.Tensor
        :param b2_tensor: Input tensor B2
        :type b2_tensor: cute.Tensor
        :param sfa_tensor: Scale factor tensor A
        :type sfa_tensor: cute.Tensor
        :param sfb1_tensor: Scale factor tensor B1
        :type sfb1_tensor: cute.Tensor
        :param sfb2_tensor: Scale factor tensor B2
        :type sfb2_tensor: cute.Tensor
        :param c_tensor: Output tensor C
        :type c_tensor: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param xtream: CUDA xtream for asynchronous execution
        :type xtream: cuda.CUxtream
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b1_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b1_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        # print(f"DEBUG: GMEM SFA Layout: {sfa_layout}")
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b1_tensor.shape, self.sf_vec_size)
        sfb1_tensor = cute.make_tensor(sfb1_tensor.iterator, sfb_layout)
        sfb2_tensor = cute.make_tensor(sfb2_tensor.iterator, sfb_layout)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B1 and B2
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b1_smem_layout_staged, (None, None, None, 0))
        tma_atom_b1, tma_tensor_b1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b1_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b2_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB1 and SFB2
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(self.sfb1_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb1, tma_tensor_sfb1 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb1_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb2, tma_tensor_sfb2 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb2_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Handle special 192 N-dimension case if needed (keeping same logic as base)
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            # This logic would be duplicated for sfb2 if we used sfb2 in the same way
            # For now let's assume standard behavior or apply to both if needed.
            # Base script only applies to tma_tensor_sfb.
            pass

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        
        # Dual GEMM: A + 2*B + SFA + 2*SFB
        self.num_tma_load_bytes = (
            a_copy_size + 2 * b_copy_size + sfa_copy_size + 2 * sfb_copy_size
        ) * atom_thr_size

        # Setup TMA store for C

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size

        self.tile_sched_params, grid = self._compute_grid(
            c_tensor,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB1: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b1_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB2: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b2_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB1: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb1_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB2: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb2_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b1,
            tma_tensor_b1,
            tma_atom_b2,
            tma_tensor_b2,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb1,
            tma_tensor_sfb1,
            tma_atom_sfb2,
            tma_tensor_sfb2,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b1_smem_layout_staged,
            self.b2_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb1_smem_layout_staged,
            self.sfb2_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
        )


    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b1: cute.CopyAtom,
        mB_nkl1: cute.Tensor,
        tma_atom_b2: cute.CopyAtom,
        mB_nkl2: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb1: cute.CopyAtom,
        mSFB_nkl1: cute.Tensor,
        tma_atom_sfb2: cute.CopyAtom,
        mSFB_nkl2: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b1_smem_layout_staged: cute.ComposedLayout,
        b2_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb1_smem_layout_staged: cute.Layout,
        sfb2_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched dual GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        
        # Debug: Print warp entry
        tidx = cute.arch.thread_idx()[0]
        # if tid x % 32 == 0:
        #     cute.printf("[WARP %d] Entered kernel\\n", warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b1)
            cpasync.prefetch_descriptor(tma_atom_b2)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb1)
            cpasync.prefetch_descriptor(tma_atom_sfb2)
            cpasync.prefetch_descriptor(tma_atom_c)

        # Debug checkpoint
        # if tidx % 32 == 0:
        #     cute.printf("[CHECKPOINT A] Warp %d before use_2cta_instrs\\n", warp_idx)
        
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        
        # Debug: Print value right after assignment
        # if tidx % 32 == 0:
        #     cute.printf("[CHECKPOINT A2] Warp %d: use_2cta_instrs=%d\\n", warp_idx, use_2cta_instrs)

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        # Debug checkpoint
        # if tidx % 32 == 0:
        #     cute.printf("[CHECKPOINT B] Warp %d before smem alloc\n", warp_idx)
        
        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Debug checkpoint
        # if tidx % 32 == 0:
        #     cute.printf("[CHECKPOINT C] Warp %d before ab_pipeline init\n", warp_idx)
        
        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Debug checkpoint
        # if tidx % 32 == 0:
        #     cute.printf("[CHECKPOINT D] Warp %d before acc_pipeline init\n", warp_idx)
        
        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        #
        # Tensor memory dealloc barrier init
        #
        # Debug: Before TmemAllocator
        # if tidx % 32 == 0:
        #     cute.printf("[BEFORE TMEM ALLOC] Warp %d: allocator_warp_id=%d, is_two_cta=%d, tmem_buf=%p\\n", 
        #                warp_idx, self.epilog_warp_id[0], use_2cta_instrs, storage.tmem_holding_buf)
        
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        
        # Debug: After TmemAllocator
        # if tidx % 32 == 0:
        #     cute.printf("[AFTER TMEM ALLOC] Warp %d survived TmemAllocator\\n", warp_idx)

        #
        # Cluster arrive after barrier init
        #
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        
        # Debug: After pipeline_init_arrive
        # if tidx % 32 == 0:
        #     cute.printf("[AFTER ARRIVE] Warp %d survived pipeline_init_arrive\\n", warp_idx)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB1 = storage.sB1.get_tensor(
            b1_smem_layout_staged.outer, swizzle=b1_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB2 = storage.sB2.get_tensor(
            b2_smem_layout_staged.outer, swizzle=b2_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB1 = storage.sSFB1.get_tensor(sfb1_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB2 = storage.sSFB2.get_tensor(sfb2_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b1_full_mcast_mask = None
        b2_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb1_full_mcast_mask = None
        sfb2_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b1_mcast or self.is_b2_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b1_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            b2_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb1_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )
            sfb2_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )


        #
        #
        # Global tensor partition
        #
        # (bM, bK, RestM, RestK, RestL)
        gA = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB1 = cute.local_tile(
            mB_nkl1, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB2 = cute.local_tile(
            mB_nkl2, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB1 = cute.local_tile(
            mSFB_nkl1,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB2 = cute.local_tile(
            mSFB_nkl2,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA, mode=[3])

        #
        #
        # Partition for TiledMMA
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB1 = thr_mma.partition_B(gB1)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB2 = thr_mma.partition_B(gB2)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB1 = thr_mma_sfb.partition_B(gSFB1)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB2 = thr_mma_sfb.partition_B(gSFB2)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        #
        # Partition for TMA load
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )

        # TMA load B1/B2 partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB1, tBgB1 = cpasync.tma_partition(
            tma_atom_b1,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB1, 0, 3),
            cute.group_modes(tCgB1, 0, 3),
        )
        tBsB2, tBgB2 = cpasync.tma_partition(
            tma_atom_b2,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB2, 0, 3),
            cute.group_modes(tCgB2, 0, 3),
        )

        #  TMA load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load SFB1/2 partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB1, tBgSFB1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb1,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB1, 0, 3),
            cute.group_modes(tCgSFB1, 0, 3),
        )
        tBsSFB1 = cute.filter_zeros(tBsSFB1)
        tBgSFB1 = cute.filter_zeros(tBgSFB1)
        tBsSFB2, tBgSFB2 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb2,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB2, 0, 3),
            cute.group_modes(tCgSFB2, 0, 3),
        )
        tBsSFB2 = cute.filter_zeros(tBsSFB2)
        tBgSFB2 = cute.filter_zeros(tBgSFB2)

        #
        #
        # Fragment setup
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB1 = tiled_mma.make_fragment_B(sB1)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB2 = tiled_mma.make_fragment_B(sB2)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        if cutlass.const_expr(self.overlapping_accum):
            num_acc_stage_overlapped = 2
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, num_acc_stage_overlapped)
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = cute.make_tensor(
                tCtAcc_fake.iterator,
                cute.make_layout(
                    tCtAcc_fake.shape,
                    stride=(
                        tCtAcc_fake.stride[0],
                        tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        (256 - self.num_sf_tmem_cols) * tCtAcc_fake.stride[0][1],
                    ),
                ),
            )
        else:
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_fake = tiled_mma.make_fragment_C(
                cute.append(acc_shape, self.num_acc_stage)
            )

        # Debug: Before pipeline_init_wait
        # if tidx % 32 == 0:
        #     cute.printf("[BEFORE WAIT] Warp %d about to call pipeline_init_wait\\n", warp_idx)

        #
        #
        # Wait for cluster sync
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        
        # Debug: After pipeline_init_wait
        # if tidx % 32 == 0:
        #     cute.printf("[AFTER WAIT] Warp %d survived pipeline_init_wait\\n", warp_idx)

        #
        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            # Debug: TMA warp entry
            # if tidx % 32 == 0:
            #     cute.printf("[TMA START] Warp %d tidx %d: TMA warp entered\n", warp_idx, tidx)
            
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB1_slice = tBgB1[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB2_slice = tBgB2[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]

                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                # ((atom_v, rest_v), RestK)
                tBgSFB1_slice = tBgSFB1[(None, slice_n, None, mma_tile_coord_mnl[2])]
                # ((atom_v, rest_v), RestK)
                tBgSFB2_slice = tBgSFB2[(None, slice_n, None, mma_tile_coord_mnl[2])]

                #
                # Prefetch: Initial batch of prefetches to prime the pipeline
                #
                if self.prefetch_enabled:
                    for pf_k_tile in cutlass.range(
                        0, min(self.prefetch_dist, k_tile_cnt), unroll=1
                    ):
                        if is_leader_cta:
                            cute.prefetch(tma_atom_a, tAgA_slice[(None, pf_k_tile)])
                            cute.prefetch(tma_atom_b1, tBgB1_slice[(None, pf_k_tile)])
                            cute.prefetch(tma_atom_b2, tBgB2_slice[(None, pf_k_tile)])
                            cute.prefetch(tma_atom_sfa, tAgSFA_slice[(None, pf_k_tile)])
                            cute.prefetch(tma_atom_sfb1, tBgSFB1_slice[(None, pf_k_tile)])
                            cute.prefetch(tma_atom_sfb2, tBgSFB2_slice[(None, pf_k_tile)])

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )

                #
                # Tma load loop
                #
                # Debug: Before TMA loop
                # if tidx % 32 == 0:
                #     cute.printf("[TMA LOOP START] Warp %d tidx %d: Starting TMA load loop\\n", warp_idx, tidx)
                
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )
                    bar = ab_pipeline.producer_get_barrier(ab_producer_state)

                    if is_leader_cta:
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, ab_producer_state.count)],
                            tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=bar,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b1,
                            tBgB1_slice[(None, ab_producer_state.count)],
                            tBsB1[(None, ab_producer_state.index)],
                            tma_bar_ptr=bar,
                            mcast_mask=b1_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b2,
                            tBgB2_slice[(None, ab_producer_state.count)],
                            tBsB2[(None, ab_producer_state.index)],
                            tma_bar_ptr=bar,
                            mcast_mask=b2_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, ab_producer_state.count)],
                            tAsSFA[(None, ab_producer_state.index)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb1,
                            tBgSFB1_slice[(None, ab_producer_state.count)],
                            tBsSFB1[(None, ab_producer_state.index)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb1_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb2,
                            tBgSFB2_slice[(None, ab_producer_state.count)],
                            tBsSFB2[(None, ab_producer_state.index)],
                            tma_bar_ptr=bar,
                            mcast_mask=sfb2_full_mcast_mask,
                        )

                    # Prefetch: Rolling prefetch for next tiles
                    if self.prefetch_enabled:
                        if k_tile < k_tile_cnt - self.prefetch_dist:
                            future_k_tile = ab_producer_state.count + self.prefetch_dist
                            if is_leader_cta:
                                cute.prefetch(
                                    tma_atom_a, tAgA_slice[(None, future_k_tile)]
                                )
                                cute.prefetch(
                                    tma_atom_b1, tBgB1_slice[(None, future_k_tile)]
                                )
                                cute.prefetch(
                                    tma_atom_b2, tBgB2_slice[(None, future_k_tile)]
                                )
                                cute.prefetch(
                                    tma_atom_sfa, tAgSFA_slice[(None, future_k_tile)]
                                )
                                cute.prefetch(
                                    tma_atom_sfb1, tBgSFB1_slice[(None, future_k_tile)]
                                )
                                cute.prefetch(
                                    tma_atom_sfb2, tBgSFB2_slice[(None, future_k_tile)]
                                )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)
            
            # Debug: TMA warp exit
            # if tidx % 32 == 0:
            #     cute.printf("[TMA END] Warp %d tidx %d: TMA warp completed\n", warp_idx, tidx)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            # Debug checkpoint 1
            # if tidx % 32 == 0:
            #     cute.printf("[CHECKPOINT 1] Warp %d tidx %d: MMA warp entered\n", warp_idx, tidx)

            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()
            # Debug checkpoint 2
            # if tidx % 32 == 0:
            #     cute.printf("[CHECKPOINT 2] Warp %d tidx %d: TMEM allocation complete\n", warp_idx, tidx)

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor  
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # Debug checkpoint 3
            # if tidx % 32 == 0:
            #     cute.printf("[CHECKPOINT 3] Warp %d tidx %d: Retrieved TMEM pointer\n", warp_idx, tidx)

            # ACC1 and ACC2
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc1_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            # # Debug checkpoint 5
            # if tidx % 32 == 0:
            #     cute.printf("[CHECKPOINT 5] Warp %d tidx %d: Acc1 tensor created\n", warp_idx, tidx)
            acc_tmem_ptr1 = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc1_base),
                dtype=self.acc_dtype,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc2_base = cute.make_tensor(acc_tmem_ptr1, tCtAcc_fake.layout)
            
            # # Debug: After Acc2 creation
            # if tidx % 32 == 0:
            #     cute.printf("[CHECKPOINT 6] Warp %d tidx %d: Acc2 tensor created\\n", warp_idx, tidx)

            # # Debug: Print TMEM offsets
            # offset_acc1 = tcgen05.find_tmem_tensor_col_offset(tCtAcc1_base)
            # offset_acc2 = tcgen05.find_tmem_tensor_col_offset(tCtAcc2_base)
            # if warp_idx == self.mma_warp_id and tidx == 0:
            #     cute.printf("[TMEM DEBUG] Acc1 offset = %d cols\n", offset_acc1)
            #     cute.printf("[TMEM DEBUG] Acc2 offset = %d cols\n", offset_acc2)
            
            # # Debug: After debug printing
            # if tidx % 32 == 0:
            #     cute.printf("[CHECKPOINT 7] Warp %d tidx %d: After offset debug\n", warp_idx, tidx)

            # SFA/SFB1/SFB2
            # SFA starts right after Acc2
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr1 + tcgen05.find_tmem_tensor_col_offset(tCtAcc2_base),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            
            # # Debug: After SFA creation
            # if tidx % 32 == 0:
            #     cute.printf("[CHECKPOINT 8] Warp %d tidx %d: SFA tensor created\n", warp_idx, tidx)

            # # Debug: Print SFA offset
            # offset_sfa = tcgen05.find_tmem_tensor_col_offset(tCtSFA)
            # if warp_idx == self.mma_warp_id and tidx == 0:
            #     cute.printf("[TMEM DEBUG] SFA offset = %d cols\\n", offset_sfa)

            # Debug TMEM Offsets
            if warp_idx == self.mma_warp_id and tidx == 0:
                off1 = tcgen05.find_tmem_tensor_col_offset(tCtAcc1_base)
                off2 = tcgen05.find_tmem_tensor_col_offset(tCtAcc2_base)
                off_sfa = tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                cute.printf("TMEM DEBUG: Acc1 Size=%d, Acc2 Size=%d, SFA Size=%d\n", off1, off2, off_sfa)
                cute.printf("TMEM DEBUG: Acc1 Ptr=%p, Acc2 Ptr=%p, SFA Ptr=%p\n", acc_tmem_ptr, acc_tmem_ptr1, sfa_tmem_ptr)

            # Get SFB1 tmem ptr
            sfb_tmem_ptr1 = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc1_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc2_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb1_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB1 = cute.make_tensor(sfb_tmem_ptr1, tCtSFB_layout)

            # Debug: Print SFB1 offset
            offset_sfb1 = tcgen05.find_tmem_tensor_col_offset(tCtSFB1)
            if warp_idx == self.mma_warp_id and tidx == 0:
                cute.printf("[TMEM DEBUG] SFB1 offset = %d cols\n", offset_sfb1)
            
            # Get SFB2 tmem ptr
            sfb_tmem_ptr2 = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc1_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc2_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFB1),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB2 = cute.make_tensor(sfb_tmem_ptr2, tCtSFB_layout)

            # Debug: Print SFB2 offset and total


            #
            # Partition for S2T copy of SFA/SFB
            #
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB1_compact_s2t,
                tCtSFB1_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB1, tCtSFB1)
            (
                _,
                tCsSFB2_compact_s2t,
                tCtSFB2_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB2, tCtSFB2)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc1 = tCtAcc1_base[(None, None, None, acc_stage_index)]
                # (MMA, MMA_M, MMA_N)
                tCtAcc2 = tCtAcc2_base[(None, None, None, acc_stage_index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB1_mma = tCtSFB1
                tCtSFB2_mma = tCtSFB2
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    offset = (
                        cutlass.Int32(2)
                        if mma_tile_coord_mnl[1] % 2 == 1
                        else cutlass.Int32(0)
                    )
                    shifted_ptr1 = cute.recast_ptr(
                        sfb_tmem_ptr1 + offset, dtype=self.sf_dtype
                    )
                    tCtSFB1_mma = cute.make_tensor(shifted_ptr1, tCtSFB_layout)
                    shifted_ptr2 = cute.recast_ptr(
                        sfb_tmem_ptr2 + offset, dtype=self.sf_dtype
                    )
                    tCtSFB2_mma = cute.make_tensor(shifted_ptr2, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr1 = cute.recast_ptr(
                        sfb_tmem_ptr1 + offset, dtype=self.sf_dtype
                    )
                    tCtSFB1_mma = cute.make_tensor(shifted_ptr1, tCtSFB_layout)
                    shifted_ptr2 = cute.recast_ptr(
                        sfb_tmem_ptr2 + offset, dtype=self.sf_dtype
                    )
                    tCtSFB2_mma = cute.make_tensor(shifted_ptr2, tCtSFB_layout)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        #  Copy SFA/SFB from smem to tmem
                        s2t_coord = (None, None, None, None, ab_consumer_state.index)

                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[s2t_coord],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB1_compact_s2t[s2t_coord],
                            tCtSFB1_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB2_compact_s2t[s2t_coord],
                            tCtSFB2_compact_s2t,
                        )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )

                            # Set SFA/SFB tensor to tiled_mma
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA,
                                tCtSFA[sf_kblock_coord].iterator,
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB1_mma[sf_kblock_coord].iterator,
                            )
                            
                            # Debug: Print A, B, SFA, SFB before GEMM
                            # Reverted due to "load & store swizzled memory is not supported yet"
                            
                            cute.gemm(
                                tiled_mma,
                                tCtAcc1,
                                tCrA[kblock_coord],
                                tCrB1[kblock_coord],
                                tCtAcc1,
                            )

                            tiled_mma.set(
                                tcgen05.Field.SFB,
                                tCtSFB2_mma[sf_kblock_coord].iterator,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc2,
                                tCrA[kblock_coord],
                                tCrB2[kblock_coord],
                                tCtAcc2,
                            )

                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(
                                ab_consumer_state
                            )

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            # Debug: Epilogue warp checkpoint A
            # if warp_idx == 0 and tidx == 0:
            #     cute.printf("[EPILOGUE A] Warp %d: Entering epilogue warp\n", warp_idx)
            
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)
            # Debug: Epilogue warp checkpoint B
            # if warp_idx == 0 and tidx == 0:
            #     cute.printf("[EPILOGUE B] Warp %d: TMEM allocated (%d cols)\n", warp_idx, self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()
            # Debug: Epilogue warp checkpoint C
            # if warp_idx == 0 and tidx == 0:
            #     cute.printf("[EPILOGUE C] Warp %d: TMEM wait complete\n", warp_idx)

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            # ACC1 and ACC2
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc1_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            acc_tmem_ptr1 = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc1_base),
                dtype=self.acc_dtype,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc2_base = cute.make_tensor(acc_tmem_ptr1, tCtAcc_fake.layout)

            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc1_base,
                tTR_rAcc1,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc1_base, tCgC, epi_tile, use_2cta_instrs
            )
            (
                _,
                tTR_tAcc2_base,
                tTR_rAcc2,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc2_base, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc1.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC, epi_tile, sC
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                # Get accumulator stage index
                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_consumer_state.phase
                    reverse_subtile = cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
                else:
                    acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                tTR_tAcc1 = tTR_tAcc1_base[
                    (None, None, None, None, None, acc_stage_index)
                ]
                tTR_tAcc2 = tTR_tAcc2_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc1 = cute.group_modes(tTR_tAcc1, 3, cute.rank(tTR_tAcc1))
                tTR_tAcc2 = cute.group_modes(tTR_tAcc2, 3, cute.rank(tTR_tAcc2))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc1.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in cutlass.range(subtile_cnt):
                    real_subtile_idx = subtile_idx
                    if cutlass.const_expr(self.overlapping_accum):
                        if reverse_subtile:
                            real_subtile_idx = self.cta_tile_shape_mnk[1] // self.epi_tile_n - 1 - subtile_idx
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc1_mn = tTR_tAcc1[(None, None, None, real_subtile_idx)]
                    tTR_tAcc2_mn = tTR_tAcc2[(None, None, None, real_subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc1_mn, tTR_rAcc1)
                    cute.copy(tiled_copy_t2r, tTR_tAcc2_mn, tTR_rAcc2)

                    #
                    # Async arrive accumulator buffer empty earlier when overlapping_accum is enabled
                    #
                    if cutlass.const_expr(self.overlapping_accum):
                        if subtile_idx == self.iter_acc_early_release_in_epilogue:
                            # Fence for TMEM load
                            cute.arch.fence_view_async_tmem_load()
                            with cute.arch.elect_one():
                                acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()

                    #
                    # Convert to C type and apply fused SiLU(Acc1) * Acc2
                    #
                    acc_vec1 = tiled_copy_r2s.retile(tTR_rAcc1).load()
                    acc_vec2 = tiled_copy_r2s.retile(tTR_rAcc2).load()

                    # Apply SiLU to acc1: acc1 / (1.0 + exp(-acc1))
                    silu_vec1 = acc_vec1 / (1.0 + cute.math.exp(-acc_vec1, fastmath=True))
                    # silu_vec1 = acc_vec1
                    
                    # Multiply: SiLU(acc1) * Acc2
                    result_vec = silu_vec1 * acc_vec2
                    
                    result_vec_c = result_vec.to(self.c_dtype)
                    tRS_rC.store(result_vec_c)



                    #
                    # Store C to shared memory
                    #
                    c_buffer = (num_prev_subtiles + real_subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    


                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    self.epilog_sync_barrier.arrive_and_wait()
                    


                    #
                    # TMA store C to global memory
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, real_subtile_idx)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                if cutlass.const_expr(not self.overlapping_accum):
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy for TMEM to register
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The register array for C
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor for C
        :type sC: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tma_atom_c, bSG_sC, bSG_gC) where:
            - tma_atom_c: The TMA copy atom
            - bSG_sC: The partitioned shared memory tensor C
            - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of Scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Scale factor vector size.
        :type sf_vec_size: int
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # ACC stages
        # Force 1 stage for dual GEMM to avoid TMEM overflow (512 cols max)
        # 2 stages (N=128) would require ~560 cols (512 for acc + 48 for SF)
        num_acc_stage = 1

        # Default C stages
        num_c_stage = 2

        # Calculate smem layout and size for one stage of A, B, SFA, SFB and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )
        # print(f"DEBUG: SFA Layout: {sfa_smem_layout_staged_one}")
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        # Dual GEMM: 1 A, 2 B, 1 SFA, 2 SFB
        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + 2 * cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + 2 * cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B/SFA/SFB stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B/SFA/SFB stage
        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B/SFA/SFB stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes and sf_vec_size are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes and sf_vec_size are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Check valid ab_dtype
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """
        Check if the mma tiler and cluster shape are valid

        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]

        :return: True if the mma tiler and cluster shape are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # Skip invalid mma tile shape
        if mma_tiler_mn[0] not in [128, 256]:
            is_valid = False
        if mma_tiler_mn[1] not in [64, 128, 192, 256]:
            is_valid = False
        # Skip illegal cluster shape
        if cluster_shape_mn[0] % (2 if mma_tiler_mn[0] == 256 else 1) != 0:
            is_valid = False
        # Skip invalid cluster shape
        is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            # Special cluster shape check for scale factor multicasts.
            # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not is_power_of_2(cluster_shape_mn[0])
            or not is_power_of_2(cluster_shape_mn[1])
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        l: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            is_valid = False
        return is_valid

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        l: int,
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor tensor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param l: The number of columns in the C tensor
        :type l: int
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        can_implement = True
        # Skip unsupported types
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # Skip unsupported layouts
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment(
            m, n, k, l, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement


# Global dtypes for compatibility with reference implementation style
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.Float16

def custom_kernel(data):
    """
    Execute the block-scaled dual GEMM kernel with silu activation,
    C = silu(A @ B1) * (A @ B2).

    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.

    Args:
        data: Tuple of (a, b1, b2, sfa_cpu, sfb1_cpu, sfb2_cpu, c) PyTorch tensors
            a: [m, k, l] - Input matrix in float4e2m1fn
            b1: [n, k, l] - Input matrix in float4e2m1fn
            b2: [n, k, l] - Input matrix in float4e2m1fn
            sfa_cpu: [m, k, l] - Scale factors in float8_e4m3fn, used by reference implementation
            sfb1_cpu: [n, k, l] - Scale factors in float8_e4m3fn, used by reference implementation
            sfb2_cpu: [n, k, l] - Scale factors in float8_e4m3fn, used by reference implementation
            sfa_permuted: [32, 4, rest_m, 4, rest_k, l] - Scale factors in float8_e4m3fn
            sfb1_permuted: [32, 4, rest_n, 4, rest_k, l] - Scale factors in float8_e4m3fn
            sfb2_permuted: [32, 4, rest_n, 4, rest_k, l] - Scale factors in float8_e4m3fn
            c: [m, n, l] - Output vector in float16

    Returns:
        Output tensor c with computed results
    """
    a, b1, b2, _, _, _, sfa_permuted, sfb1_permuted, sfb2_permuted, c = data

    # Ensure kernel is compiled (will use cached version if available)
    # To avoid the compilation overhead, we compile the kernel once and cache it.
    compiled_func = compile_kernel()

    # Get dimensions from MxKxL layout
    _, k, _ = a.shape
    m, n, l = c.shape
    # Torch use e2m1_x2 data type, thus k is halved
    k = k * 2
    
    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    from cutlass.cute.runtime import make_ptr
    
    a_ptr = make_ptr(
        ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b1_ptr = make_ptr(
        ab_dtype, b1.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b2_ptr = make_ptr(
        ab_dtype, b2.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    # Create aligned tensors using cutlass_torch to match Base file logic
    # This ensures the TMA descriptor has correct alignment/layout assertions.
    # We wrap the existing 'c' tensor (passed from main).
    
    # Check default c_major in Base file is "n" (Row Major)
    # Check default c_major in Base file is "n" (Row Major)
    c_major = "n"

    # DEBUG: Verify Host Shape
    # print(f"DEBUG_HOST_SHAPE: c.shape={c.shape}, c.stride={c.stride()}")

    # Ensure c is rank-3 (m, n, l) even if passed as (m, n) for L=1
    if c.ndim == 2:
        c = c.unsqueeze(-1)
    
    # Capture potential shadow tensor
    # Capture potential shadow tensor
    # c_tensor, c_out_shadow = cutlass_torch.cute_tensor_like(
    #    c, c_dtype, is_dynamic_layout=True, assumed_align=16
    # )
    pass # Bypass shadow buffer creation
    
    # If a shadow buffer was created (different object than c), print warning
    # if c_out_shadow.data_ptr() != c.data_ptr():
        # print("WARNING: Shadow Buffer Created for C! Kernel writes to this new buffer.")
        # print(f"Shadow Ptr: {c_out_shadow.data_ptr()}, Original Ptr: {c.data_ptr()}")
    #     pass
        # We need to copy shadow back to c if possible, but shape/dtype mismatch prevents direct copy?
        # Better to fix invocation to pass F16 C_out.

    
    # Apply metadata for TMA (Base file logic)
    # Apply metadata for TMA (Base file logic)
    # c_tensor.mark_compact_shape_dynamic(
    #     mode=1 if c_major == "n" else 0,
    #     stride_order=(2, 0, 1) if c_major == "n" else (2, 1, 0),
    #     divisibility=32, # 32 for FP4 inputs per Base File
    # )
    
    # c_ptr = c_tensor
    c_ptr = make_ptr(
        c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_ptr = make_ptr(
        sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb1_ptr = make_ptr(
        sf_dtype, sfb1_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb2_ptr = make_ptr(
        sf_dtype, sfb2_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    max_active_clusters = 1 # Not used by current kernel but argument present in signature? 
    # Current kernel signature in this file: a, b1, b2, sfa, sfb1, sfb2, c, m, n, k, l, xtream
    # It does NOT take max_active_clusters in the compiled function here.
    # Wait, my compile_kernel wraps gemm_kernel.
    
    # xtream = cutlass_torch.default_xtream()

    compiled_func(
        a_ptr, 
        b1_ptr, 
        b2_ptr, 
        sfa_ptr, 
        sfb1_ptr, 
        sfb2_ptr,
        c_ptr,
        m, n, k, l,
    )
    
    # CRITICAL: Copy shadow buffer back to original C if needed
    # CRITICAL: Copy shadow buffer back to original C if needed
    # if c_out_shadow.data_ptr() != c.data_ptr():
    #      # print("DEBUG: Copying Shadow Buffer back to C_out...")
    #      c.copy_(c_out_shadow)

    return c
    
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


def benchmark_one_config(m, n, k, l, seed, max_repeats=100):
    from reference import ref_kernel as dual_gemm_reference, generate_input
    from utils import clear_l2_cache, verbose_allclose
    
    # 1. Generate multiple sets of input data
    data_list = []
    current_seed = seed
    for _ in range(NUM_ITERATIONS_PER_BENCHMARK):
        data_list.append(generate_input(m, n, k, l, current_seed))
        current_seed += 42
        
    # 2. Correctness check (on a subset)
    try:
        compile_kernel() # Ensure compiled
        
        # We just check the first one
        (A_val, B1_val, B2_val, sA_layout, sB1_layout, sB2_layout, SFA, SFB1, SFB2, C_ref) = data_list[0]
        
        # Run with standard random inputs
        # print("  [Debug] Running with Random Inputs and Random Scale Factors...")
        
        # Init C_out directly as (M, N, L) to match kernel expectation and avoid shadow buffer
        C_out = torch.zeros((m, n, l), dtype=torch.float16, device="cuda")
        
        # Construct new arguments tuple with C_out replacing the last element (C_ref)
        # We reuse the variables unpacked at line 2830
        kernel_args = (A_val, B1_val, B2_val, sA_layout, sB1_layout, sB2_layout, SFA, SFB1, SFB2, C_out)

        # Call custom kernel with new tuple containing C_out
        # Clone data before running custom kernel to preserve input state for reference check
        kernel_args_cloned = _clone_data(kernel_args)
        custom_kernel(kernel_args)
        
        # Ensure completion
        torch.cuda.synchronize()
        
        # Calculate Reference Result using the Reference Kernel
        print("  [Reference] Computing ground truth C_ref using 'ref_kernel'...")
        # Note: ref_kernel expects the same tuple.
        C_ref = dual_gemm_reference(kernel_args_cloned)
        
        # print(f"  [Debug] C_out.shape={C_out.shape}")
        # print(f"  [Debug] C_ref.shape={C_ref.shape}")

        abs_diff = torch.abs(C_out.to(torch.float32) - C_ref.to(torch.float32))
        max_diff = torch.max(abs_diff).item()
        min_diff = torch.min(abs_diff).item()
        num_diffs = (abs_diff > 0.1).sum().item() # Using 0.1 threshold to ignore minor FP discrepancies
        total_elements = abs_diff.numel()

        print(f"  [Stats] C_out: min={C_out.min().item()}, max={C_out.max().item()}, nans={torch.isnan(C_out).sum().item()}, infs={torch.isinf(C_out).sum().item()}")
        print(f"  [Stats] C_ref: min={C_ref.min().item()}, max={C_ref.max().item()}, nans={torch.isnan(C_ref).sum().item()}, infs={torch.isinf(C_ref).sum().item()}")

        # Use verbose_allclose for detailed mismatch info
        mismatches = verbose_allclose(C_out.to(torch.float32), C_ref.to(torch.float32), atol=1e-03, rtol=1e-03)
        diff_info = f"max_diff={max_diff:.6f}, min_diff={min_diff:.6f}, diffs>0.1={num_diffs}"
        if mismatches:
            print(f"  [FAILED] {diff_info} Mismatches found:")
            for m in mismatches:
                print(f"    {m}")
        else:
            print(f"  [PASSED] {diff_info}")
        
        # Diagnostic Check for Identity Validation
        if False: # Always run diagnostics
            print("\n  [Diagnostic] Checking for non-zero off-diagonal elements in C_out...")
            # C_out is (M, N, L). Assume L=1.
            # Diagonal: i == j. Off-diagonal: i != j.
            # Create mask for off-diagonal
            m_idx = torch.arange(m, device='cuda').view(m, 1, 1)
            n_idx = torch.arange(n, device='cuda').view(1, n, 1)
            off_diag_mask = (m_idx != n_idx)
            
            off_diag_elements = C_out[off_diag_mask]
            non_zero_off_diag = off_diag_elements[off_diag_elements.abs() > 1e-4]
            num_non_zero_off_diag = non_zero_off_diag.numel()
            
            print(f"  [Diagnostic] Found {num_non_zero_off_diag} non-zero off-diagonal elements.")
            if num_non_zero_off_diag > 0:
                 print(f"    First 10 non-zero off-diag: {non_zero_off_diag[:10].tolist()}")

            print("\n  [Diagnostic] Checking Diagonal Elements (C_out vs C_ref)...")
            diag_mask = (m_idx == n_idx)
            # Create valid diagonal mask based on min(m, n)
            min_dim = min(m, n)
            
            # Extract diagonals
            # Note: C_out[i, i, 0]
            # Simple manual extraction
            diag_diffs = []
            diag_mismatch_count = 0
            for i in range(min_dim):
                val_out = C_out[i, i, 0].item()
                val_ref = C_ref[i, i, 0].item()
                if abs(val_out - val_ref) > 0.1:
                    diag_mismatch_count += 1
                    if len(diag_diffs) < 10:
                        diag_diffs.append((i, val_out, val_ref))
            
            print(f"  [Diagnostic] Found {diag_mismatch_count} mismatches on diagonal (First {len(diag_diffs)} shown):")
            for i, tout, tref in diag_diffs:
                print(f"    idx[{i},{i}]: out={tout:.4f}, ref={tref:.4f}, diff={abs(tout-tref):.4f}")
        
        # Extended Stats
        c_out_f32 = C_out.to(torch.float32)
        n_nans = torch.isnan(c_out_f32).sum().item()
        n_infs = torch.isinf(c_out_f32).sum().item()
        min_val = torch.min(c_out_f32[torch.isfinite(c_out_f32)]).item() if torch.any(torch.isfinite(c_out_f32)) else "N/A"
        max_val = torch.max(c_out_f32[torch.isfinite(c_out_f32)]).item() if torch.any(torch.isfinite(c_out_f32)) else "N/A"
        
        print(f"  [Stats] C_out: min={min_val}, max={max_val}, nans={n_nans}, infs={n_infs}")
        # C_ref Stats
        c_ref_f32 = C_ref.to(torch.float32)
        n_nans_ref = torch.isnan(c_ref_f32).sum().item()
        n_infs_ref = torch.isinf(c_ref_f32).sum().item()
        min_val_ref = torch.min(c_ref_f32[torch.isfinite(c_ref_f32)]).item() if torch.any(torch.isfinite(c_ref_f32)) else "N/A"
        max_val_ref = torch.max(c_ref_f32[torch.isfinite(c_ref_f32)]).item() if torch.any(torch.isfinite(c_ref_f32)) else "N/A"
        print(f"  [Stats] C_ref: min={min_val_ref}, max={max_val_ref}, nans={n_nans_ref}, infs={n_infs_ref}")
        print(f"  [Check] Max diff: {max_diff:.4f}, Min diff: {min_diff:.4f}, Diff Count (>0.1): {num_diffs}/{total_elements} ({num_diffs/total_elements*100:.2f}%)")
        
        # if max_diff > 2.0 or n_nans > 0 or n_infs > 0: # Heuristic threshold
        #      print(f"  [Error] Large diff or invalid values found: diff={max_diff}, nans={n_nans}, infs={n_infs}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [Error] Execution failed: {e}")
        return None

    # 3. Timing runs
    durations = []
    
    # Warmup (5 iterations)
    print("  [Benchmark] Warming up (5 iters)...")
    for data in data_list[:5]:
        (A_val, B1_val, B2_val, sA_layout, sB1_layout, sB2_layout, SFA, SFB1, SFB2, _) = data
        C_warm = torch.zeros((m, n, l), dtype=torch.float16, device="cuda")
        args = (A_val, B1_val, B2_val, sA_layout, sB1_layout, sB2_layout, SFA, SFB1, SFB2, C_warm)
        custom_kernel(args)
    torch.cuda.synchronize()
    
    # Benchmark (50 iterations)
    print(f"  [Benchmark] Running 50 iterations for timing...")
    for i in range(1): # Loop once over data list, assuming 50 items
        clear_l2_cache()
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for data in data_list:
            (A_val, B1_val, B2_val, sA_layout, sB1_layout, sB2_layout, SFA, SFB1, SFB2, _) = data
            C_bench = torch.zeros((m, n, l), dtype=torch.float16, device="cuda")
            args = (A_val, B1_val, B2_val, sA_layout, sB1_layout, sB2_layout, SFA, SFB1, SFB2, C_bench)
            custom_kernel(args)
        end_event.record()
        
        torch.cuda.synchronize()
        duration = (start_event.elapsed_time(end_event) / NUM_ITERATIONS_PER_BENCHMARK) * 1e6 # ms to ns
        durations.append(duration)
        
        # Stop early if stable
        if i > 5:
            stats = calculate_stats(durations)
            if stats.err / stats.mean < 0.005: # 0.5% error
                break
                
    return calculate_stats(durations)

@app.function(gpu="B200", timeout=300)
def run_dual_gemm_benchmark(
    m, n, k, l, skip_ref_check=True, warmup_iterations=5, iterations=20
):
    # Ignoring warmup_iterations/iterations args in favor of internal benchmark logic constants
    print("Benchmarking Dual GEMM (Optimized with JIT Compilation)")
    
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
        
        stats = benchmark_one_config(m, n, k, l, seed)
        if stats:
             tflops = (2 * m * n * k) / (stats.mean * 1e-9) / 1e12
             print(f"    Mean: {stats.mean/1e3:.2f} us, TFLOPS: {tflops:.2f}")
             all_results.append({
                 "spec": f"M={m},N={n},K={k}",
                 "tflops": tflops,
                 "mean_us": stats.mean / 1e3
             })
             
    print("\n" + "="*80)
    print(f"{'Spec':<20} | {'TFLOPS':<10} | {'Mean (us)':<10}")
    print("-"*80)
    
    tflops_values = []
    time_values_us = []

    for res in all_results:
        print(f"{res['spec']:<20} | {res['tflops']:<10.2f} | {res['mean_us']:<10.2f}")
        tflops_values.append(res['tflops'])
        time_values_us.append(res['mean_us'])

    if tflops_values:
        log_sum_tflops = sum(math.log(x) for x in tflops_values)
        geomean_tflops = math.exp(log_sum_tflops / len(tflops_values))

        log_sum_time = sum(math.log(x) for x in time_values_us)
        geomean_time = math.exp(log_sum_time / len(time_values_us))

        print("-"*80)
        print(f"{'GEOMETRIC MEAN':<20} | {geomean_tflops:<10.2f} | {geomean_time:<10.2f}")
    print("="*80)


@app.function(gpu="B200", timeout=60)
def run_with_compute_sanitizer(
    m, n, k, l, skip_ref_check=True, warmup_iterations=10, iterations=100
):
    """
    Run benchmark with compute-sanitizer to debug memory errors.
    This will pinpoint the exact location of CUDA_ERROR_ILLEGAL_ADDRESS.
    """
    import subprocess
    import sys
    
    print("=" * 80)
    print("RUNNING WITH COMPUTE-SANITIZER - Memory Error Detection Enabled")
    print("=" * 80)
    
    # Create a temporary Python script to run the benchmark
    script_content = f"""
import sys
sys.path.insert(0, '/root')
from dual_gemm_opt_cute import run_dual_gemm_benchmark
run_dual_gemm_benchmark.local({m}, {n}, {k}, {l}, {skip_ref_check}, {warmup_iterations}, {iterations})
"""
    
    with open("/tmp/run_benchmark.py", "w") as f:
        f.write(script_content)
    
    # Run with compute-sanitizer
    cmd = [
        "compute-sanitizer",
        "--tool", "memcheck",
        "--print-limit", "1000",
        sys.executable,
        "/tmp/run_benchmark.py"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 80)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    # Print output
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print("=" * 80)
    
    return result.returncode

@app.function(gpu="B200", timeout=1200)
def run_with_nsys(
    m, n, k, l, skip_ref_check=True, warmup_iterations=0, iterations=10
):
    """
    Run benchmark with Nsight Systems (nsys) to profile kernel timeline.
    """
    import subprocess
    import sys
    
    print("=" * 80)
    print("RUNNING WITH NSIGHT SYSTEMS (nsys) - Timeline Profiling Enabled")
    print("=" * 80)
    
    # Create a temporary Python script
    script_content = f'''
import sys
sys.path.insert(0, "/root")
from dual_gemm_opt_cute import run_dual_gemm_benchmark
# Use more iterations for nsys to capture enough data
run_dual_gemm_benchmark.local({m}, {n}, {k}, {l}, {skip_ref_check}, {warmup_iterations}, {iterations})
'''
    
    with open("/tmp/run_benchmark_nsys.py", "w") as f:
        f.write(script_content)
    
    # Run with nsys
    # -t cuda,nvtx: Trace CUDA and NVTX
    # --stats=true: Print summary statistics to stdout
    # --sample=none --cpuctxsw=none: Disable sampling to avoid internal errors in container
    cmd = [
        "nsys",
        "profile",
        "-t", "cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--stats=true",
        "--force-overwrite", "true",
        "-o", "/root/profile",
        sys.executable,
        "/tmp/run_benchmark_nsys.py"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 80)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    # Print output
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print("=" * 80)
    
    return result.returncode

@app.local_entrypoint()
def debug_main(m: int = 256, n: int = 3072, k: int = 4096, l: int = 1, skip_ref_check: bool = True):
    """Debug entrypoint for running with compute-sanitizer"""
    print("Starting compute-sanitizer debug session...")
    print("Note: Run this with: compute-sanitizer --tool memcheck modal run dual_demm_opt_cute.py::debug_main")
    # Actually this debug_main calls remote function run_with_compute_sanitizer
    run_with_compute_sanitizer.remote(m, n, k, l, skip_ref_check=skip_ref_check)

@app.local_entrypoint()
def profile_main(m: int = 256, n: int = 4096, k: int = 7168, l: int = 1):
    """Profile entrypoint for running with nsys"""
    print(f"Starting Nsight Systems profiling session for M={m}, N={n}, K={k}...")
    run_with_nsys.remote(m, n, k, l, skip_ref_check=True, warmup_iterations=1, iterations=10)

@app.local_entrypoint()
def main(m: int = 256, n: int = 3072, k: int = 4096, l: int = 1, skip_ref_check: bool = True):
    run_dual_gemm_benchmark.remote(m, n, k, l, skip_ref_check=skip_ref_check)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Simple arg parsing for local run
        m = int(sys.argv[1]) if len(sys.argv) > 1 else 256
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3072
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
        l = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        run_dual_gemm_benchmark(m, n, k, l, skip_ref_check=False)
    else:
        # Default local benchmark
        run_dual_gemm_benchmark(256, 3072, 4096, 1, skip_ref_check=False)
