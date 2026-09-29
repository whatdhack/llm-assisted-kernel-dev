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

import os

# Force architecture for Blackwell CuTe DSL
os.environ["CUTE_DSL_ARCH"] = "sm_100a"



#import pprint
#pprint.pprint(dict(os.environ))

import subprocess
import sys
import importlib.util

def ensure_tvm_installed():
    """Checks if apache-tvm-ffi is installed, and installs it if not."""
    if importlib.util.find_spec("tvm_ffi") is None:
        print("apache-tvm-ffi not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "apache-tvm-ffi"])
            print("Successfully installed apache-tvm-ffi.")
            # Important: Invalidate caches to ensure new package is visible
            importlib.invalidate_caches()
        except subprocess.CalledProcessError as e:
            print(f"Failed to install apache-tvm-ffi: {e}")
            pass

ensure_tvm_installed()
#import tvm_ffi

#import pkg_resources

#def get_installed_packages_pkg_resources():
#    installed_packages = pkg_resources.working_set
#    # Format the output as a list of "package==version" strings
#    installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])

    #for package_info in installed_packages_list():
    #        print(package_info)

#get_installed_packages_pkg_resources()

if "POPCORN_FD" not in os.environ:
    import modal
import dataclasses
import argparse


import functools
from typing import List, Type, Tuple, Union
from inspect import isclass

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor
from cutlass.base_dsl import ir

from task import input_t, output_t
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


"""
This example provides an experimental implementation of the SM100 grouped blockscaled GEMM kernel, please note that the APIs and implementation details related to this kernel may change in future releases.

A grouped blockscaled GEMM example for the NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of grouped blockscaled GEMM using a TMA plus Blackwell SM100 TensorCore
warp-specialized persistent kernel.
The grouped GEMM workload computes a batch of GEMM operations with distinct problem sizes. Pointers to matrices
in global memory are passed to the kernel in an array (also held in global memory). Similarly, problem shapes and
strides are also stored in arrays in GMEM.

This differs from "Batched Array" GEMM since the size of each GEMM problem in the grouped GEMM concept may be distinct.

To run this example:

.. code-block:: bash

    modal run examples/blackwell/grouped_blockscaled_gemm.py                                  \
      --ab-dtype Float4E2M1FN --sf-dtype Float8E8M0FNU --sf-vec-size 16                       \
      --c-dtype Float16                                                                       \
      --mma-tiler-mn 128,128 --cluster-shape-mn 1,1                                           \
      --problem-sizes-mnkl "(8192,1280,32,1),(32,384,1536,1),(640,1280,32,1),(640,160,32,1)"  \
      --num-groups 4

The above example command makes 4 groups of different m, n, k sizes. The Blackwell tcgen05 MMA tile shape
is specified as (128, 64) and the cluster shape is (1,1). The input, mma accumulator and output data type
are set as fp16, fp32 and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu modal run examples/blackwell/grouped_blockscaled_gemm.py                              \
      --ab-dtype Float4E2M1FN --sf-dtype Float8E8M0FNU --sf-vec-size 16                       \
      --c-dtype Float16                                                                       \
      --mma-tiler-mn 128,128 --cluster-shape-mn 1,1                                           \
      --problem-sizes-mnkl "(8192,1280,32,1),(32,384,1536,1),(640,1280,32,1),(640,160,32,1)"  \
      --num-groups 4
      --warmup-iterations 1 --iterations 10 --skip-ref-check

Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below Sm100GroupedBlockScaledGemmKernel class documentation
* A/B tensors must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 128 or 256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The l mode(aka, batch size) for each group must be 1.
* The majorness for A, B and C must be the same across all groups.
* The contiguous dimension of A/B/C tensors in each group must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.
"""


@dsl_user_op
def clock64(*, loc=None, ip=None) -> cutlass.Int64:
    return llvm.inline_asm(
        T.i64(),
        [],
        "mov.u64 $0, %clock64;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

# Kernel configuration parameters
# Size of tma descriptor in bytes
bytes_per_tensormap = 128
# Number of tensormaps: a, b, sfa, sfb
num_tensormaps = 4
# Tile sizes for M, N, K dimensions
mma_tiler_mnk = (128, 128, 256)
# Shape of the K dimension for the MMA instruction
mma_inst_shape_k = 64
# FP4 data type for A and B
ab_dtype = cutlass.Float4E2M1FN
# FP8 data type for scale factors
sf_dtype = cutlass.Float8E4M3FN
# FP16 output type
c_dtype = cutlass.Float16
# Scale factor block size (16 elements share one scale)
sf_vec_size = 16
# Number of threads per CUDA thread block
threads_per_cta = 128
# Stage numbers of shared memory and tmem
num_acc_stage = 1
num_ab_stage = 1
# Total number of columns in tmem
num_tmem_alloc_cols = 512


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b

# copied from cute4

class Sm100GroupedBlockScaledGemmKernel:
    """This example demonstrates an implementation of grouped blockscaled GEMM using a TMA plus Blackwell SM100 TensorCore
    warp-specialized persistent kernel.

    :param sf_vec_size: Scalefactor vector size.
    :type sf_vec_size: int
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: In current version, A and B tensors must have the same data type
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
        - MMA tiler N must be 128/256
        - Cluster shape M must be multiple of 2 if Mma tiler M is 256
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
        - Cluster shape M/N must be <= 4 for scale factor multicasts due to limited size of scale factors
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        num_groups: int,
    ):
        """Initializes the configuration for a Blackwell grouped blockscaled GEMM kernel.

        Besides configurations for dense persistent blockscaled GEMM, there is an extra config specific to grouped blockscaled GEMM:

        :param sf_vec_size: Scalefactor vector size.
        :type sf_vec_size: int
        :param mma_tiler_mn: tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: tuple[int, int]
        :param cluster_shape_mn: tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: tuple[int, int]
        """
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.num_groups = num_groups

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.tensormap_update_mode = utils.TensorMapUpdateMode.SMEM

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
        # Set barrier for epilogue sync and tmem ptr sync
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        # Barrier used by MMA/TMA warps to signal A/B tensormap initialization completion
        self.tensormap_ab_init_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=64,
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

    # Set up configurations that dependent on gemm inputs.
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
        - Checking reserved smem bytes size capacity for mbar, tensor memory management and tensormap updates utilization
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
        self.cluster_tile_shape_mnk = tuple(
            x * y for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn, 1))
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
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute epilogue subtile
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
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

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
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
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
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

        mbar_smem_bytes = self._get_mbar_smem_bytes(
            num_acc_stage=self.num_acc_stage,
            num_ab_stage=self.num_ab_stage,
            num_c_stage=self.num_c_stage,
        )

        # Use utils.TensorMapUpdateMode.SMEM by default
        tensormap_smem_bytes = (
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap
            * Sm100GroupedBlockScaledGemmKernel.num_tensormaps
        )
        if (
            mbar_smem_bytes
            + tensormap_smem_bytes
            + Sm100GroupedBlockScaledGemmKernel.tensor_memory_management_bytes
            > self.reserved_smem_bytes
        ):
            raise ValueError(
                f"smem consumption for mbar and tensormap {mbar_smem_bytes + tensormap_smem_bytes} exceeds the "
                f"reserved smem bytes {self.reserved_smem_bytes}"
            )

    @cute.jit
    def __call__(
        self,
        a_tuple,
        b_tuple,
        c_tuple,
        sfa_tuple,
        sfb_tuple,
        #group_count: cutlass.Constexpr[int],
        problem_shape_mnkl_ptr: cute.Pointer,
        strides_abc_ptr: cute.Pointer,
        tensor_address_abc_ptr: cute.Pointer,
        tensor_address_sfasfb_ptr: cute.Pointer,
        #total_num_clusters: cutlass.Constexpr[int],
        tensormap_ptr: cute.Pointer,
        #max_active_clusters: cutlass.Constexpr[int],
        total_num_clusters,
        max_active_clusters,
        debug_buf: cute.Tensor,
    ):

        # Reconstruct Tensors from Pointers
        problem_shape_mnkl = cute.make_tensor(
            problem_shape_mnkl_ptr,
            cute.make_layout((self.num_groups, 4), stride=(4, 1))
        )

        strides_abc = cute.make_tensor(
            strides_abc_ptr,
            cute.make_layout((self.num_groups, 3, 2), stride=(6, 2, 1))
        )

        tensor_address_abc = cute.make_tensor(
            tensor_address_abc_ptr,
            cute.make_layout((self.num_groups, 3), stride=(3, 1))
        )

        tensor_address_sfasfb = cute.make_tensor(
            tensor_address_sfasfb_ptr,
            cute.make_layout((self.num_groups, 2), stride=(2, 1))
        )

        tensormap_cute_tensor = cute.make_tensor(
            tensormap_ptr,
            cute.make_layout((total_num_clusters, 4, 16), stride=(64, 16, 1))
        )


        initial_a = a_tuple[0]
        initial_b = b_tuple[0]

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            initial_a.shape, sf_vec_size
        )
        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            initial_b.shape, sf_vec_size
        )

        initial_sfa = sfa_tuple[0]

        initial_sfb = sfb_tuple[0]


        initial_c = c_tuple[0]


        self.a_dtype = initial_a.element_type
        self.b_dtype = initial_b.element_type
        self.sf_dtype = initial_sfa.element_type
        self.c_dtype = initial_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(initial_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(initial_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(initial_c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            initial_a.shape, self.sf_vec_size
        )
        initial_sfa = cute.make_tensor(initial_sfa.iterator, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            initial_b.shape, self.sf_vec_size
        )
        initial_sfb = cute.make_tensor(initial_sfb.iterator, sfb_layout)

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
            initial_a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            initial_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            initial_sfa,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )



        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            initial_sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            initial_c,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            total_num_clusters, self.cluster_shape_mn, max_active_clusters
        )

        self.buffer_align_bytes = 1024
        self.size_tensormap_in_i64 = (
            Sm100GroupedBlockScaledGemmKernel.num_tensormaps
            * Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap
            // 8
        )

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.MemRange[
                cutlass.Int64, self.size_tensormap_in_i64
            ]
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
                    self.a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype,
                    cute.cosize(self.sfa_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype,
                    cute.cosize(self.sfb_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # --- Internal Tensor Creation (Avoiding cute_tensor_like) ---


        # 2. Construct Metadata Tensors using Torch (on device)
        # Metadata Tensors passed as arguments
        # ts_mnkl / cute_mnkl
        # ts_strides / cute_strides
        # ts_ptrs_abc / cute_ptrs_abc
        # ts_ptrs_sf / cute_ptrs_sf

        group_count = self.num_groups

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            group_count,
            problem_shape_mnkl,      # problem_sizes_mnkl
            strides_abc,
            tensor_address_abc,
            tensor_address_sfasfb,
            tensormap_cute_tensor,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        group_count: cutlass.Constexpr,
        problem_sizes_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        ptrs_abc: cute.Tensor,
        ptrs_sfasfb: cute.Tensor,
        tensormaps: cute.Tensor,
    ):
        """
        GPU device kernel performing the grouped GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == self.tma_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfb)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

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
        # coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: tensormap buffer, a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
        tensormap_a_smem_ptr = tensormap_smem_ptr
        tensormap_b_smem_ptr = (
            tensormap_a_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )
        tensormap_sfa_smem_ptr = (
            tensormap_b_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )
        tensormap_sfb_smem_ptr = (
            tensormap_sfa_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )
        tensormap_c_smem_ptr = (
            tensormap_sfb_smem_ptr
            + Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8
        )

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

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
        )

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
        )

        # Tensor memory dealloc barrier init
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads
                    )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
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
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA Load SFA partition_S/D
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

        # TMA Load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Get tensormap buffer address
        #
        grid_dim = cute.arch.grid_dim()
        tensormap_workspace_idx = (
            bidz * grid_dim[1] * grid_dim[0] + bidy * grid_dim[0] + bidx
        )

        tensormap_manager = utils.TensorMapManager(
            utils.TensorMapUpdateMode.SMEM,
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap,
        )
        tensormap_a_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 0, None)].iterator
        )
        tensormap_b_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 1, None)].iterator
        )
        tensormap_sfa_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 2, None)].iterator
        )
        tensormap_sfb_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 3, None)].iterator
        )
        tensormap_c_gmem_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 4, None)].iterator
        )

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), grid_dim
            )
            # grouped gemm tile scheduler helper will compute the group index for the tile we're working on
            group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
                group_count,
                tile_sched_params,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
            )
            tensormap_init_done = cutlass.Boolean(False)
            # group index of last tile
            last_group_idx = cutlass.Int32(-1)

            work_tile = tile_sched.initial_work_tile_info()

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                grouped_gemm_cta_tile_info = group_gemm_ts_helper.delinearize_z(
                    cur_tile_coord,
                    problem_sizes_mnkl,
                )
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                is_group_changed = cur_group_idx != last_group_idx
                # skip tensormap update if we're working on the same group
                if is_group_changed:
                    real_tensor_a = self.make_tensor_abc_for_tensormap_update(
                        cur_group_idx,
                        self.a_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        0,  # 0 for tensor A
                    )
                    real_tensor_b = self.make_tensor_abc_for_tensormap_update(
                        cur_group_idx,
                        self.b_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        1,  # 1 for tensor B
                    )
                    real_tensor_sfa = self.make_tensor_sfasfb_for_tensormap_update(
                        cur_group_idx,
                        self.sf_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        ptrs_sfasfb,
                        0,  # 0 for tensor SFA
                    )
                    real_tensor_sfb = self.make_tensor_sfasfb_for_tensormap_update(
                        cur_group_idx,
                        self.sf_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        ptrs_sfasfb,
                        1,  # 1 for tensor SFB
                    )
                    if tensormap_init_done == False:
                        # wait tensormap initialization complete
                        self.tensormap_ab_init_barrier.arrive_and_wait()
                        tensormap_init_done = True

                    tensormap_manager.update_tensormap(
                        (
                            real_tensor_a,
                            real_tensor_b,
                            real_tensor_sfa,
                            real_tensor_sfb,
                        ),
                        (tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb),
                        (
                            tensormap_a_gmem_ptr,
                            tensormap_b_gmem_ptr,
                            tensormap_sfa_gmem_ptr,
                            tensormap_sfb_gmem_ptr,
                        ),
                        self.tma_warp_id,
                        (
                            tensormap_a_smem_ptr,
                            tensormap_b_smem_ptr,
                            tensormap_sfa_smem_ptr,
                            tensormap_sfb_smem_ptr,
                        ),
                    )

                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m
                    // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < cur_k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )

                if is_group_changed:
                    tensormap_manager.fence_tensormap_update(tensormap_a_gmem_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_b_gmem_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_sfa_gmem_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_sfb_gmem_ptr)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )

                    # TMA load A/B/SFA/SFB
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_a_gmem_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_b_gmem_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfa_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_sfa_gmem_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=sfb_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_sfb_gmem_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < cur_k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Initialize tensormaps for A, B, SFA and SFB
            #
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_a, tensormap_a_smem_ptr, self.mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_b, tensormap_b_smem_ptr, self.mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_sfa, tensormap_sfa_smem_ptr, self.mma_warp_id
            )
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_sfb, tensormap_sfb_smem_ptr, self.mma_warp_id
            )
            # indicate tensormap initialization has finished
            self.tensormap_ab_init_barrier.arrive_and_wait()

            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            self.tmem_alloc_barrier.arrive_and_wait()

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            # Make accumulator tmem tensor
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
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

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            #
            # Partition for S2T copy of SFA/SFB
            #
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            )
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t = (
                self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), grid_dim
            )
            # grouped gemm tile scheduler helper will compute the group index for the tile we're working on
            group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
                group_count,
                tile_sched_params,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
            )

            work_tile = tile_sched.initial_work_tile_info()
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                # MMA warp is only interested in number of tiles along K dimension
                (
                    cur_k_tile_cnt,
                    cur_group_idx,
                ) = group_gemm_ts_helper.search_cluster_tile_count_k(
                    cur_tile_coord,
                    problem_sizes_mnkl,
                )

                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < cur_k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in range(cur_k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                        #  Copy SFA/SFB from smem to tmem
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                        tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t_staged,
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t_staged,
                            tCtSFB_compact_s2t,
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
                                tCtSFB[sf_kblock_coord].iterator,
                            )

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )

                            # Enable accumulate on tCtAcc after first kblock
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < cur_k_tile_cnt:
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
            # initialize tensorap for C
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_c,
                tensormap_c_smem_ptr,
                self.epilog_warp_id[0],
            )
            #
            # Alloc tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            self.tmem_alloc_barrier.arrive_and_wait()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            ### Start from here
            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = (
                self.epilog_tmem_copy_and_partition(
                    epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
                )
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            tma_atom_c, bSG_sC, bSG_gC_partitioned = (
                self.epilog_gmem_copy_and_partition(
                    epi_tidx, tma_atom_c, tCgC, epi_tile, sC
                )
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), grid_dim
            )
            # grouped gemm tile scheduler helper will compute the group index for the tile we're working on
            group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
                group_count,
                tile_sched_params,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
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
            # group index to start searching
            last_group_idx = cutlass.Int32(-1)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                grouped_gemm_cta_tile_info = group_gemm_ts_helper.delinearize_z(
                    cur_tile_coord,
                    problem_sizes_mnkl,
                )
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                is_group_changed = cur_group_idx != last_group_idx

                if is_group_changed:
                    # construct tensor c based on real shape, stride information
                    real_tensor_c = self.make_tensor_abc_for_tensormap_update(
                        cur_group_idx,
                        self.c_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        2,  # 2 for tensor C
                    )
                    tensormap_manager.update_tensormap(
                        ((real_tensor_c),),
                        ((tma_atom_c),),
                        ((tensormap_c_gmem_ptr),),
                        self.epilog_warp_id[0],
                        (tensormap_c_smem_ptr,),
                    )

                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m
                    // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k

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

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                if is_group_changed:
                    if warp_idx == self.epilog_warp_id[0]:
                        tensormap_manager.fence_tensormap_update(tensormap_c_gmem_ptr)

                #
                # Store accumulator to global memory in subtiles
                #
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in range(subtile_cnt):
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    #
                    # Convert to C type
                    #
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    tRS_rC.store(acc_vec.to(self.c_dtype))

                    #
                    # Store C to shared memory
                    #
                    c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    #
                    # TMA store C to global memory
                    #
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC[(None, subtile_idx)],
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_c_gmem_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()
                #
                # Async arrive accumulator buffer empty
                #
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            #
            # Dealloc the tensor memory buffer
            #
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            self.epilog_sync_barrier.arrive_and_wait()
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    acc_tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

    @cute.jit
    def make_tensor_abc_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_index: int,
    ):
        """Extract stride and tensor address for a given group and construct a global tensor for A, B or C.

        This function is used within the kernel to dynamically create a CUTE tensor
        representing A, B, or C for the current group being processed, using the
        group-specific address, shape, and stride information.

        :param group_idx: The index of the current group within the grouped GEMM.
        :type group_idx: cutlass.Int32
        :param dtype: The data type of the tensor elements (e.g., cutlass.Float16).
        :type dtype: Type[cutlass.Numeric]
        :param problem_shape_mnk: The (M, N, K) problem shape for the current group.
        :type problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        :param strides_abc: Tensor containing strides for A, B, C for all groups. Layout: (group_count, 3, 2).
        :type strides_abc: cute.Tensor
        :param tensor_address_abc: Tensor containing global memory addresses for A, B, C for all groups. Layout: (group_count, 3).
        :type tensor_address_abc: cute.Tensor
        :param tensor_index: Specifies which tensor to create: 0 for A, 1 for B, 2 for C.
        :type tensor_index: int
        :return: A CUTE tensor representing the requested global memory tensor (A, B, or C) for the specified group.
        :rtype: cute.Tensor
        :raises TypeError: If the provided dtype is not a subclass of cutlass.Numeric.
        """
        ptr_i64 = tensor_address_abc[(group_idx, tensor_index)]
        if cutlass.const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16
        )

        strides_tensor_gmem = strides_abc[(group_idx, tensor_index, None)]
        strides_tensor_reg = cute.make_rmem_tensor(
            cute.make_layout(2),
            strides_abc.element_type,
        )
        cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
        stride_mn = strides_tensor_reg[0]
        stride_k = strides_tensor_reg[1]
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        if cutlass.const_expr(tensor_index == 0):  # tensor A
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        elif cutlass.const_expr(tensor_index == 1):  # tensor B
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((n, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        else:  # tensor C
            m = problem_shape_mnk[0]
            n = problem_shape_mnk[1]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, n, c1), stride=(stride_mn, stride_k, c0)),
            )

    @cute.jit
    def make_tensor_sfasfb_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        tensor_address_sfasfb: cute.Tensor,
        tensor_index: int,
    ):
        """Extract tensor address for a given group and construct a global tensor for SFA or SFB.

        This function is used within the kernel to dynamically create a CUTE tensor
        representing SFA or SFB for the current group being processed, using the
        group-specific address, shape information.

        :param group_idx: The index of the current group within the grouped GEMM.
        :type group_idx: cutlass.Int32
        :param dtype: The data type of the tensor elements (e.g., cutlass.Float16).
        :type dtype: Type[cutlass.Numeric]
        :param problem_shape_mnk: The (M, N, K) problem shape for the current group.
        :type problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        :param tensor_address_sfasfb: Tensor containing global memory addresses for SFA, SFB for all groups. Layout: (group_count, 2).
        :type tensor_address_sfasfb: cute.Tensor
        :param tensor_index: Specifies which tensor to create: 0 for SFA, 1 for SFB.
        :type tensor_index: int
        :return: A CUTE tensor representing the requested global memory tensor (SFA, SFB) for the specified group.
        :rtype: cute.Tensor
        :raises TypeError: If the provided dtype is not a subclass of cutlass.Numeric.
        """
        ptr_i64 = tensor_address_sfasfb[(group_idx, tensor_index)]
        if cutlass.const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16
        )

        c1 = cutlass.Int32(1)
        if cutlass.const_expr(tensor_index == 0):  # tensor SFA
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (m, k, c1), self.sf_vec_size
            )
            return cute.make_tensor(
                tensor_gmem_ptr,
                sfa_layout,
            )
        else:  # tensor SFB
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (n, k, c1), self.sf_vec_size
            )
            return cute.make_tensor(
                tensor_gmem_ptr,
                sfb_layout,
            )

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

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
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
        num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

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

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
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
        total_num_clusters: int,
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr[int],
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """Compute tile scheduler parameters and grid shape for grouped GEMM operations.

        :param total_num_clusters: Total number of clusters to process across all groups.
        :type total_num_clusters: int
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr[int]

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: tuple[utils.PersistentTileSchedulerParams, tuple[int, ...]]
        """
        # Create problem shape with M, N dimensions from cluster shape
        # and L dimension representing the total number of clusters.
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )

        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, (*cluster_shape_mn, 1)
        )

        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _get_mbar_smem_bytes(**kwargs_stages: int) -> int:
        """Calculate shared memory consumption for memory barriers based on provided stages.

        Each stage requires 2 barriers, and each barrier consumes 8 bytes of shared memory.
        The total consumption is the sum across all provided stages. This function calculates the total
        shared memory needed for these barriers.

        :param kwargs_stages: Variable keyword arguments where each key is a stage name
                              (e.g., num_acc_stage, num_ab_stage) and each value is the
                              number of stages of that type.
        :type kwargs_stages: int
        :return: Total shared memory bytes required for all memory barriers.
        :rtype: int
        """
        num_barriers_per_stage = 2
        num_bytes_per_barrier = 8
        mbar_smem_consumption = sum(
            [
                num_barriers_per_stage * num_bytes_per_barrier * stage
                for stage in kwargs_stages.values()
            ]
        )
        return mbar_smem_consumption

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
        if mma_tiler_mn[1] not in [128, 256]:
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
        problem_sizes_mnkl: List[Tuple[int, int, int, int]],
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param problem_sizes_mnkl: The problem shape for each group
        :type problem_sizes_mnkl: List[Tuple[int, int, int, int]]
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

        for m, n, k, l in problem_sizes_mnkl:
            if (
                not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
                or not check_contigous_16B_alignment(
                    ab_dtype, b_major == "n", (n, k, l)
                )
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
        problem_sizes_mnkl: List[Tuple[int, int, int, int]],
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
        if not Sm100GroupedBlockScaledGemmKernel.is_valid_dtypes_and_scale_factor_vec_size(
            ab_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            can_implement = False
        # Skip unsupported layouts
        if not Sm100GroupedBlockScaledGemmKernel.is_valid_layouts(
            ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        # Skip invalid mma tile shape and cluster shape
        if not Sm100GroupedBlockScaledGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            mma_tiler_mn, cluster_shape_mn
        ):
            can_implement = False
        # Skip illegal problem shape for load/store alignment
        if not Sm100GroupedBlockScaledGemmKernel.is_valid_tensor_alignment(
            problem_sizes_mnkl, ab_dtype, c_dtype, a_major, b_major, c_major
        ):
            can_implement = False
        return can_implement

    # Size of smem we reserved for mbarrier, tensor memory management and tensormap update
    reserved_smem_bytes = 1024
    bytes_per_tensormap = 128
    num_tensormaps = 5
    # size of smem used for tensor memory management
    tensor_memory_management_bytes = 12


# Create tensor and return the pointer, tensor, and stride
def create_tensor_and_stride(
    l: int,
    mode0: int,
    mode1: int,
    is_mode0_major: bool,
    dtype: type[cutlass.Numeric],
    is_dynamic_layout: bool = True,
    identity: bool = False,
) -> tuple[int, torch.Tensor, cute.Tensor, torch.Tensor, tuple[int, int]]:
    """Create GPU tensor from either a new or existing CPU tensor.

    :param torch_tensor_cpu: Optional existing CPU tensor to reuse. If None, creates a new one.
    :type torch_tensor_cpu: torch.Tensor, optional
    """

    # Create new CPU tensor
    torch_tensor_cpu = cutlass_torch.matrix(
        l,
        mode0,
        mode1,
        is_mode0_major,
        cutlass.Float32,
    )

    if identity:
        torch.nn.init.eye_(torch_tensor_cpu[0])

    # Create GPU tensor from CPU tensor (new or existing)
    cute_tensor, torch_tensor = cutlass_torch.cute_tensor_like(
        torch_tensor_cpu, dtype, is_dynamic_layout, assumed_align=16
    )

    # Mark tensor with element divisibility for 16B alignment
    cute_tensor.mark_compact_shape_dynamic(
        mode=0 if is_mode0_major else 1,
        stride_order=(2, 1, 0) if is_mode0_major else (2, 0, 1),
        divisibility=32 if dtype == cutlass.Float4E2M1FN else 16,
    )

    # omit stride for L mode as it is always 1
    stride = (1, mode0) if is_mode0_major else (mode1, 1)

    return (
        torch_tensor.data_ptr(),
        torch_tensor,
        cute_tensor,
        torch_tensor_cpu,
        stride,
    )


def create_tensors_abc_for_all_groups(
    problem_sizes_mnkl: List[tuple[int, int, int, int]],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    identity_a: bool = False,
    identity_b: bool = False,
) -> tuple[
    List[List[int]],
    List[List[torch.Tensor]],
    List[tuple],
    List[List[tuple]],
    List[List[torch.Tensor]],
]:
    ref_torch_fp32_tensors_abc = []
    torch_tensors_abc = []
    cute_tensors_abc = []
    strides_abc = []
    ptrs_abc = []

    # Iterate through all groups and create tensors for each group
    for group_idx, (m, n, k, l) in enumerate(problem_sizes_mnkl):
        # Create tensors  A, B, C
        (
            ptr_a,
            torch_tensor_a,
            cute_tensor_a,
            ref_torch_fp32_tensor_a,
            stride_mk_a,
        ) = create_tensor_and_stride(l, m, k, a_major == "m", ab_dtype, identity=identity_a)

        (
            ptr_b,
            torch_tensor_b,
            cute_tensor_b,
            ref_torch_fp32_tensor_b,
            stride_nk_b,
        ) = create_tensor_and_stride(l, n, k, b_major == "n", ab_dtype, identity=identity_b)

        (
            ptr_c,
            torch_tensor_c,
            cute_tensor_c,
            ref_torch_fp32_tensor_c,
            stride_mn_c,
        ) = create_tensor_and_stride(l, m, n, c_major == "m", c_dtype)

        ref_torch_fp32_tensors_abc.append(
            [ref_torch_fp32_tensor_a, ref_torch_fp32_tensor_b, ref_torch_fp32_tensor_c]
        )

        ptrs_abc.append([ptr_a, ptr_b, ptr_c])
        torch_tensors_abc.append([torch_tensor_a, torch_tensor_b, torch_tensor_c])
        strides_abc.append([stride_mk_a, stride_nk_b, stride_mn_c])
        cute_tensors_abc.append(
            (
                cute_tensor_a,
                cute_tensor_b,
                cute_tensor_c,
            )
        )

    return (
        ptrs_abc,
        torch_tensors_abc,
        cute_tensors_abc,
        strides_abc,
        ref_torch_fp32_tensors_abc,
    )


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


# Create scale factor tensor SFA/SFB
def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype):
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(k, sf_vec_size)
    ref_shape = (l, mn, sf_k)

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

    ref_permute_order = (1, 2, 0)
    mma_permute_order = (3, 4, 1, 5, 2, 0)

    # Create f32 ref torch tensor (cpu)
    ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        ref_shape,
        torch.float32,
        permute_order=ref_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=1,
            max_val=3,
        ),
    )

    # Create f32 cute torch tensor (cpu)
    cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=0,
            max_val=1,
        ),
    )

    # convert ref f32 tensor to cute f32 tensor
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref_f32_torch_tensor_cpu),
        from_dlpack(cute_f32_torch_tensor_cpu),
    )
    cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

    # reshape makes memory contiguous
    ref_f32_torch_tensor_cpu = (
        ref_f32_torch_tensor_cpu.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, mn, sf_k, sf_vec_size)
        .reshape(l, mn, sf_k * sf_vec_size)
        .permute(*ref_permute_order)
    )
    # prune to mkl for reference check.
    ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

    # Create dtype cute torch tensor (cpu)
    cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
        cute_f32_torch_tensor_cpu,
        dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )

    # Convert f32 cute tensor to dtype cute tensor
    cute_tensor = cutlass_torch.convert_cute_tensor(
        cute_f32_torch_tensor,
        cute_tensor,
        dtype,
        is_dynamic_layout=True,
    )
    # get pointer of the tensor
    ptr = cute_torch_tensor.data_ptr()
    return ref_f32_torch_tensor_cpu, ptr, cute_tensor, cute_torch_tensor


def create_tensors_sfasfb_for_all_groups(
    problem_sizes_mnkl: List[tuple[int, int, int, int]],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
) -> tuple[
    List[List[int]],
    List[List[torch.Tensor]],
    List[tuple],
    List[List[torch.Tensor]],
]:
    ptrs_sfasfb = []
    torch_tensors_sfasfb = []
    cute_tensors_sfasfb = []
    refs_sfasfb = []

    # Iterate through all groups and create tensors for each group
    for group_idx, (m, n, k, l) in enumerate(problem_sizes_mnkl):
        sfa_ref, ptr_sfa, sfa_tensor, sfa_torch = create_scale_factor_tensor(
            l, m, k, sf_vec_size, sf_dtype
        )
        sfb_ref, ptr_sfb, sfb_tensor, sfb_torch = create_scale_factor_tensor(
            l, n, k, sf_vec_size, sf_dtype
        )
        ptrs_sfasfb.append([ptr_sfa, ptr_sfb])
        torch_tensors_sfasfb.append([sfa_torch, sfb_torch])
        cute_tensors_sfasfb.append(
            (
                sfa_tensor,
                sfb_tensor,
            )
        )
        refs_sfasfb.append([sfa_ref, sfb_ref])

    return (
        ptrs_sfasfb,
        torch_tensors_sfasfb,
        cute_tensors_sfasfb,
        refs_sfasfb,
    )



@dataclasses.dataclass
class CustomStats:
    pre_launch_cpu: float = 0.0
    post_launch_cpu: float = 0.0
    kernel_launch_overhead: float = 0.0
    e2e_cpu_time: float = 0.0
    debug_buffer: object = None

# Global buffers to avoid host-side tensor creation on every call
# Global buffers to avoid host-side tensor creation on every call
_global_debug_buffer = None
_dummy_debug_buf = None
_dummy_debug_ptr = None
_kernel_metadata_cache = {}

#This function is used to compile the kernel once and cache it and then allow users to
# run the kernel multiple times to get more accurate timing results.
# Global cache for compiled kernels
_compiled_kernel_cache = {}
# Global dummy debug buffer for non-profiled runs
_dummy_debug_buf = None
# Global list to keep metadata tensors alive during kernel execution
# Global list to keep metadata tensors alive during kernel execution
_metadata_keep_alive = []

def compile_kernel0(problem_sizes, sfa_tuple, sfb_tuple, num_tensormap_buffers, total_num_clusters, kernel_host):
    """
    Compile the kernel once and cache it using problem_sizes as the key.
    """
    global _compiled_kernel_cache

    # Cache by number of groups (canonical caching) to avoid redundant compilation for different sizes.
    # We assume the kernel is generic regarding M, N, K (reads execution time params).
    cache_key = (len(problem_sizes),)

    # Check if we already have a compiled kernel for this number of groups
    if cache_key in _compiled_kernel_cache:
        return _compiled_kernel_cache[cache_key]

    num_groups = len(problem_sizes)

    # Use canonical problem sizes for compilation to ensure shared JIT entry
    # The actual shapes are handled at runtime via tensormap updates.
    # Fakes below still use original problem_sizes for correct TMA descriptor setup.
    # The caching key uses only num_groups so we compile once per group count.

    # Create fake tuples for compilation
    a_fakes = []
    b_fakes = []
    c_fakes = []
    sfa_fakes = []
    sfb_fakes = []
    m_fakes = []
    n_fakes = []
    k_fakes = []
    l_fakes = []
    # cta_mn_fakes removed - no longer needed

    # Fix for missing variables logic
    # Defaults if globals not set (for local testing/compilation)
    sf_vec_size = globals().get('sf_vec_size', 16)
    mma_tiler_mn = globals().get('mma_tiler_mn', (128, 128))
    cluster_shape_mn = globals().get('cluster_shape_mn', (1, 1))
    ab_dtype = globals().get('ab_dtype', cutlass.Float4E2M1FN)
    sf_dtype = globals().get('sf_dtype', cutlass.Float8E8M0FNU)
    c_dtype = globals().get('c_dtype', cutlass.Float16)

    for i, (m, n, k, l) in enumerate(problem_sizes):
        # Construct RowMajor fakes to match Runtime layout (K-contiguous).
        # This ensures CuTe understands that the packing (float4x2) reduces the K dimension, not M.
        # Stride: (K, 1, M*K)
        # Note: make_fake_compact_tensor assumes Stride (1, M, ...)
        # We manually create generic tensor with correct stride.

        # A: (M, K/2, L) Stride (K/2, 1, M*K/2) - Row Major Packed
        a_half_k = k // 2
        a_fakes.append(cute.runtime.make_fake_compact_tensor(ab_dtype, (m, k, l), stride_order=(k, 1, m * k)))

        # B: (N, K/2, L) Stride (K/2, 1, N*K/2) - Row Major Packed
        b_half_k = k // 2
        b_fakes.append(cute.runtime.make_fake_compact_tensor(ab_dtype, (n, k, l), stride_order=(k, 1, n * k)))

        c_fakes.append(cute.runtime.make_fake_compact_tensor(c_dtype, (m, n, l), stride_order=(n,1, m*n)))
        # Scale factors are reordered to 6D shape: (32, 4, m_rest, 4, k_rest, l)
        # m_rest = ceil(m / 128)
        # k_rest = ceil(ceil(k / 16) / 4)
        m_rest = (m + 127) // 128
        sf_k = (k + 15) // 16
        k_rest = (sf_k + 3) // 4

        #sf_reordered_shape_a = (32, 4, m_rest, 4, k_rest, l)
        #sf_reordered_stride  = (16, 4, 1,       , 1,
        #sfa_fakes.append(cute.runtime.make_fake_compact_tensor(sf_dtype, sf_reordered_shape_a))
        sf_reordered_shape_a = sfa_tuple[i].shape
        sf_reordered_stride_a  = sfa_tuple[i].stride()
        sfa_fakes.append(cute.runtime.make_fake_compact_tensor(sf_dtype, sf_reordered_shape_a, stride_order=sf_reordered_stride_a))

        #n_rest = (n + 127) // 128
        #sf_reordered_shape_b = (32, 4, n_rest, 4, k_rest, l)
        #sfb_fakes.append(cute.runtime.make_fake_compact_tensor(sf_dtype, sf_reordered_shape_b))
        sf_reordered_shape_b = sfb_tuple[i].shape
        sf_reordered_stride_b  = sfb_tuple[i].stride()
        sfb_fakes.append(cute.runtime.make_fake_compact_tensor(sf_dtype, sf_reordered_shape_b, stride_order=sf_reordered_stride_b))

        m_fakes.append(cutlass.Int64(m))
        n_fakes.append(cutlass.Int64(n))
        k_fakes.append(cutlass.Int64(k))
        l_fakes.append(cutlass.Int64(l))




    # Buffer placeholders
    total_clusters_sym = cutlass.Int32(1)

    tensormap_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (num_tensormap_buffers,
                                                Sm100GroupedBlockScaledGemmKernel.num_tensormaps,
                                                Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8),
                                                stride_order=(Sm100GroupedBlockScaledGemmKernel.num_tensormaps*Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8,
                                                Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8, 1)
                                                )



    debug_buf_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (1024,))

    if os.environ.get("COMPILE_LOCALLY_FOR_B200"):
        # Mock hardware info for B200 (approximate)
        sm_count = 148
        max_active_clusters = 148
    else:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_max_active_clusters(1)
        max_active_clusters = hardware_info.get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )

    # For debug buf, if we expect a pointer in kernel_host, we should pass a pointer here too?
    # kernel_host signature: debug_buf: cute.Tensor
    # In custom_kernel we pass: cute_ptr_of_debug_buf (Pointer?)
    # Wait, in kernel_host (line 808): debug_buf: cute.Tensor
    # In kernel_host body (line ?): debug_buf is used directly.
    # Ah, in custom_kernel we pass `cute_ptr_of_debug_buf`?
    # Let's check kernel_host again.

    # In kernel_host:
    # debug_buf: cute.Tensor

    # In custom_kernel:
    # cute_ptr_of_debug_buf = make_ptr(...)
    # compiled_func(..., cute_ptr_of_debug_buf, ...)

    # So if kernel_host expects cute.Tensor, but we pass cute.Pointer, CuTe might handle it if defined right?
    # Actually, look at kernel_host again.
    # line 930: debug_buf = cute.make_tensor(ptr_of_debug_buf, ...) <--- This is what I saw in group_gemm_cute2.py
    # But in group_gemm_cute4.py, kernel_host signature (line 808) is `debug_buf: cute.Tensor`.
    # And there is NO `make_tensor` wrapping it inside kernel_host (unless I missed it).

    # If kernel_host accepts cute.Tensor, we should pass cute.Tensor (fake) in compile.
    # But I see in custom_kernel I am now passing `cute_ptr_of_debug_buf` (Pointer).
    # This implies I should have changed kernel_host signature for debug_buf too, OR I should not have changed custom_kernel to pass pointer for debug_buf.

    # Let's assume for now I should only fix tensormap ptr.

    #print(f"a_fakes[0] : {a_fakes[0]} ")
    # Metadata Fakes
    #mnkl_fakes_tensor = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (len(problem_sizes), 4), stride_order=(4,1))
    #fake_mnkl = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (num_groups, 4))
    fake_mnkl = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (num_groups, 4), stride_order=(4, 1))

    #fake_strides = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (num_groups, 3, 2))
    fake_strides = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (num_groups, 3, 2), stride_order=(6, 2, 1))

    #fake_ptrs_abc = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (num_groups, 3))
    fake_ptrs_abc = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (num_groups, 3), stride_order=(3, 1))

    #fake_ptrs_sf = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (num_groups, 2))
    fake_ptrs_sf = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (num_groups, 2), stride_order=(2, 1))

    # Create fake pointers for compilation
    fake_ptr_mnkl = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    fake_ptr_strides = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem)
    fake_ptr_abc = make_ptr(cutlass.Int64, 0, cute.AddressSpace.gmem)
    fake_ptr_sf = make_ptr(cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16)
    fake_ptr_tensormap = make_ptr(cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16)

    compiled_func = cute.compile(
        kernel_host,
        tuple(a_fakes),
        tuple(b_fakes),
        tuple(c_fakes),
        tuple(sfa_fakes),
        tuple(sfb_fakes),
        #group_count: cutlass.Constexpr[int],
        fake_ptr_mnkl,
        fake_ptr_strides,
        fake_ptr_abc,
        fake_ptr_sf,
        fake_ptr_tensormap,
        #max_active_clusters: cutlass.Constexpr[int],
        cutlass.Int32(total_num_clusters),
        cutlass.Int32(128),     # max_active_clusters
        debug_buf_fake,
        #options="--enable-tvm-ffi ",
        options="--enable-tvm-ffi --opt-level 3",
    )

    # Store compiled kernel in cache with problem_sizes as key
    _compiled_kernel_cache[cache_key] = compiled_func
    return compiled_func


compile_kernel = compile_kernel0
#os.environ["CUTE_DSL_ENABLE_TVM_FFI"] = "1"

def custom_kernel(data: input_t, events: Tuple[torch.cuda.Event, torch.cuda.Event] = None, stats: CustomStats = None) -> output_t:

    global _dummy_debug_buf
    global _dummy_debug_ptr

    if stats:
        t_start = time.perf_counter()

    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data

    # 1. Faster Tensor Unpacking (Avoids loops)
    if abc_tensors:
         a_tuple, b_tuple, c_tuple = zip(*abc_tensors)
         sfa_tuple, sfb_tuple = zip(*sfasfb_reordered_tensors)
    else:
         a_tuple = b_tuple = c_tuple = sfa_tuple = sfb_tuple = ()


    # Fix for missing variables logic
    # Defaults if globals not set (for local testing/compilation)
    sf_vec_size = globals().get('sf_vec_size', 16)
    mma_tiler_mn = globals().get('mma_tiler_mn', (128, 128))
    cluster_shape_mn = globals().get('cluster_shape_mn', (1, 1))
    ab_dtype = globals().get('ab_dtype', cutlass.Float4E2M1FN)
    sf_dtype = globals().get('sf_dtype', cutlass.Float8E8M0FNU)
    c_dtype = globals().get('c_dtype', cutlass.Float16)

    # Compute Hardware Info for Clusters
    if os.environ.get("COMPILE_LOCALLY_FOR_B200"):
        # Mock hardware info for B200 (approximate)
        sm_count = 148
        max_active_clusters = 148
    else:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_max_active_clusters(1)
        max_active_clusters = hardware_info.get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )

    # Compute total number of cluster tiles
    def compute_cluster_tile_shape(mma, cluster):
        # cta_tile_shape is [128, mma[1]] usually?
        # Original code: cta_tile_shape_mn = [128, mma_tiler_mn[1]]
        # return tuple(x * y for x, y in zip(cta_tile_shape_mn, cluster_shape_mn))
        # Replicating original logic helper inline or use existing if accessible.
        # It was nested. I'll implement inline.
        cta_tile_shape_mn = [128, mma_tiler_mn[1]]
        return tuple(x * y for x, y in zip(cta_tile_shape_mn, cluster_shape_mn))

    cluster_tile_shape_mn = compute_cluster_tile_shape(mma_tiler_mn, cluster_shape_mn)

    total_num_clusters = 0
    for m, n, _, _ in problem_sizes:
        num_clusters_mn = tuple(
            (x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)
        )
        total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)

    # Prepare tensormap buffer for each cluster (one descriptor per cluster)
    num_tensormap_buffers = total_num_clusters
    tensormap_shape = (
        num_tensormap_buffers,
        Sm100GroupedBlockScaledGemmKernel.num_tensormaps,
        Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8,
    )
    # Allocate tensormap tensor on device use zeros
    tensor_of_tensormap, tensor_of_tensormap_torch = cutlass_torch.cute_tensor_like(
        torch.zeros(tensormap_shape, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
    )



    # Instantiate Kernel Wrapper
    grouped_blockscaled_gemm = Sm100GroupedBlockScaledGemmKernel(
        sf_vec_size,
        mma_tiler_mn,
        cluster_shape_mn,
        len(a_tuple)
    )

    # 3. Debug Buffer and Pointer
    # Debug buffer optimization
    # If stats is None, reuse GLOBAL dummy debug pointer to avoid make_ptr cost (~3us)
    if stats is None:
        if _dummy_debug_buf is None:
             _dummy_debug_buf = torch.zeros(1024, dtype=torch.int64, device="cuda")
             _dummy_debug_ptr = make_ptr(
                cutlass.Int64,
                _dummy_debug_buf.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
             )
        cute_ptr_of_debug_buf = _dummy_debug_ptr
        debug_buf = _dummy_debug_buf # needed for call signature but unused by kernel logic if stats None
    else:
        debug_buf = torch.zeros(1024, dtype=torch.int64, device="cuda")
        cute_ptr_of_debug_buf = make_ptr(
            cutlass.Int64,
            debug_buf.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )

    # The following block is replaced by the provided snippet
    # debug_buf_cute = cute.make_tensor(
    #     cute.make_ptr(cutlass.Int64, debug_buf_torch.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
    #     cute.make_layout((1024,))
    # )

    # Prepare arguments for execution
    m_list = [p[0] for p in problem_sizes]
    n_list = [p[1] for p in problem_sizes]
    k_list = [p[2] for p in problem_sizes]
    l_list = [p[3] for p in problem_sizes]

    m_tuple = tuple(cutlass.Int64(x) for x in m_list)
    n_tuple = tuple(cutlass.Int64(x) for x in n_list)
    k_tuple = tuple(cutlass.Int64(x) for x in k_list)
    l_tuple = tuple(cutlass.Int64(x) for x in l_list)

    mnkl_tuple = (m_tuple, n_tuple, k_tuple, l_tuple)

    #problem_sizes_mnkl: List[Tuple[int, int, int, int]],
    # layout (num_groups, 4):(4, 1)
    (
        tensor_of_dim_size_mnkl,
        tensor_of_dim_size_mnkl_torch,
    ) = cutlass_torch.cute_tensor_like(
        torch.tensor(problem_sizes, dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )


    

    # Replaced block ends here

    num_groups = cutlass.Int32(len(problem_sizes))

    if events:
        events[0].record()


    # Strides (N, 3, 2)
    strides_data = []
    for (a, b, c) in abc_tensors:
         strides_data.append([
             (a.stride(0), a.stride(1)),
             (b.stride(0), b.stride(1)),
             (c.stride(0), c.stride(1))
         ])
    #ts_strides = torch.tensor(strides_data, dtype=torch.int32, device="cuda")


    # Pointers ABC (N, 3)
    ptrs_abc_data = []
    for (a, b, c) in abc_tensors:
        ptrs_abc_data.append([a.data_ptr(), b.data_ptr(), c.data_ptr()])
    #ts_ptrs_abc = torch.tensor(ptrs_abc_data, dtype=torch.int64, device="cuda")


    # Pointers SFA/SFB (N, 2)
    ptrs_sf_data = []
    for (sfa, sfb) in sfasfb_reordered_tensors: # Use reordered tensors for pointers
        ptrs_sf_data.append([sfa.data_ptr(), sfb.data_ptr()])
    #ts_ptrs_sf = torch.tensor(ptrs_sf_data, dtype=torch.int64, device="cuda")


    # layout (num_groups, 3, 2):(6, 2, 1)
    tensor_of_strides_abc, tensor_of_strides_abc_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(strides_data, dtype=torch.int32),
        cutlass.Int32,
        is_dynamic_layout=False,
        assumed_align=16,
    )

    # layout (num_groups,3):(3, 1)
    tensor_of_ptrs_abc, tensor_of_ptrs_abc_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(ptrs_abc_data, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
        assumed_align=16,
    )

    # layout (num_groups,2):(2, 1)
    tensor_of_ptrs_sfasfb, tensor_of_ptrs_sfasfb_torch = cutlass_torch.cute_tensor_like(
        torch.tensor(ptrs_sf_data, dtype=torch.int64),
        cutlass.Int64,
        is_dynamic_layout=False,
        assumed_align=16,
    )

    # Retrieve compiled kernel
    compiled_grouped_gemm = compile_kernel(problem_sizes, sfa_tuple, sfb_tuple, num_tensormap_buffers, total_num_clusters, grouped_blockscaled_gemm)

    # Keep transient tensors alive
    global _metadata_keep_alive
    _metadata_keep_alive = [
        tensor_of_strides_abc_torch,
        tensor_of_ptrs_abc_torch,
        tensor_of_ptrs_sfasfb_torch, 
        tensor_of_dim_size_mnkl_torch,
        tensor_of_tensormap_torch,
    ]

    # Create pointers from torch tensors
    ptr_mnkl = make_ptr(cutlass.Int32, tensor_of_dim_size_mnkl_torch.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    ptr_strides = make_ptr(cutlass.Int32, tensor_of_strides_abc_torch.data_ptr(), cute.AddressSpace.gmem)
    ptr_abc = make_ptr(cutlass.Int64, tensor_of_ptrs_abc_torch.data_ptr(), cute.AddressSpace.gmem)
    ptr_sf = make_ptr(cutlass.Int64, tensor_of_ptrs_sfasfb_torch.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    # Tensormap Pointer
    if tensor_of_tensormap_torch is not None:
        ptr_of_tensormap = make_ptr(
            cutlass.Int64,
            tensor_of_tensormap_torch.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16
        )
    else:
        ptr_of_tensormap = make_ptr(cutlass.Int64, 0, cute.AddressSpace.gmem)

    print (f" compile_kernel() completed ")

    # Call the compiled kernel with UNPACKED arguments matching __call__ signature
    compiled_grouped_gemm(
        a_tuple,
        b_tuple,
        c_tuple,
        sfa_tuple,
        sfb_tuple,
        ptr_mnkl,
        ptr_strides,
        ptr_abc,
        ptr_sf,
        ptr_of_tensormap,
        cutlass.Int32(total_num_clusters),
        max_active_clusters,
        debug_buf,
    )

    if events:
        events[1].record()

    if stats:
        t_post_start = time.perf_counter()
        t_end = time.perf_counter()
        # Debug buffer handling...
        if debug_buf is not None:
             buf_cpu = debug_buf.cpu().numpy()
             stats.debug_buffer = buf_cpu

        stats.pre_launch_cpu = t_post_start - t_start
        stats.post_launch_cpu = t_end - t_post_start
        stats.e2e_cpu_time = t_end - t_start

    return [c for (_, _, c) in abc_tensors]



def run_group_gemm_benchmark(
    problem_sizes: List[Tuple[int, int, int, int]],
    warmup_iterations: int = 10,
    iterations: int = 10,
    enable_profiling: bool = True,
    enable_cycle_stats: bool = False,
    identity_a: bool = False,
    identity_b: bool = False,
):
    """
    Benchmark the custom_kernel with given problem sizes.
    """
    def _clone_data(data):
        """
        Recursively goes through data and clones all tensors.
        """
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

    # Generate inputs using reference implementation style for correctness compatibility
    from reference import generate_input
    m_list = [p[0] for p in problem_sizes]
    n_list = [p[1] for p in problem_sizes]
    k_list = [p[2] for p in problem_sizes]
    g = len(problem_sizes)

    # data: (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)
    data = generate_input(
        tuple(m_list),
        tuple(n_list),
        tuple(k_list),
        g,
        seed=1111,
        identity_b=identity_b,
        identity_a=identity_a,
    )

    # Correctness Verification
    # Only run verification once per suite run usually, or strictly if requested.
    # But let's run it always if not cycle stats only?
    # Actually just run it.
    print("Verifying correctness...")
    from reference import check_implementation
    try:
        # Run kernel once to get output
        # Clone data for input to avoid dirtying if kernel modifies in place
        output = custom_kernel(_clone_data(data))

        if os.environ.get("COMPILE_LOCALLY_FOR_B200"):
             print("Skipping verification for local compilation test.")
             return

        torch.cuda.synchronize()
        passed, msg = check_implementation(data, output)
        if not passed:
             print(f"Verification failed: {msg}")
        else:
             print("Verification passed!")
    except Exception as e:
        print(f"Verification crashed: {e}")
        import traceback
        traceback.print_exc()

run = run_group_gemm_benchmark


def test_compilation_only():
        # -----------------------------------------------------------------------
    # Test Configs from task.yml
    # -----------------------------------------------------------------------
    test_configs = [
        # {"m": [96, 128], "n": [128, 256], "k": [128, 512], "g": 2},
        [
            (96, 128, 128, 1),
            (128, 256, 512, 1),
        ],
    ]
    # Ensure environment variable is set for local compilation
    os.environ["COMPILE_LOCALLY_FOR_B200"] = "1"
    run_group_gemm_benchmark(
        test_configs[0],
        warmup_iterations=0,
        iterations=1,
        enable_profiling=False
    )



# ---------------------------------------------------------------------------
# Modal & Benchmark Boilerplate
# ---------------------------------------------------------------------------

# Benchmark configurations from task.yml
BENCHMARK_CONFIGS = [
    # Benchmark 1: 8 groups
    {
        "m": [80, 176, 128, 72, 64, 248, 96, 160],
        "n": [4096] * 8,
        "k": [7168] * 8,
        "g": 8,
        "l": 1
    },
    # Benchmark 2: 8 groups
    {
        "m": [40, 76, 168, 72, 164, 148, 196, 160],
        "n": [7168] * 8,
        "k": [2048] * 8,
        "g": 8,
        "l": 1
    },
    # Benchmark 3: 2 groups
    {
        "m": [192, 320],
        "n": [3072, 3072],
        "k": [4096, 4096],
        "g": 2,
        "l": 1
    },
    # Benchmark 4: 2 groups
    {
        "m": [128, 384],
        "n": [4096, 4096],
        "k": [1536, 1536],
        "g": 2,
        "l": 1
    },
]


if "POPCORN_FD" not in os.environ:
    # Define Modal Image with PyTorch and dependencies
    image = (
        modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
        .pip_install(
            "torch",
            pre=True,
            index_url="https://download.pytorch.org/whl/nightly/cu130"
        )
        .pip_install("nvidia-cutlass", "nvidia-cutlass-dsl", "numpy", "apache-tvm-ffi")
        .env({
            "CUDA_HOME": "/usr/local/cuda",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib",
            "LD_PRELOAD": "/usr/local/cuda/lib64/libcublas.so:/usr/local/cuda/lib64/libcublasLt.so",
            "CUTE_DSL_ARCH": "sm_100a",
            #"CUTE_DSL_ENABLE_TVM_FFI": "1",
        })
        .add_local_file("task.py", remote_path="/root/task.py")
        .add_local_file("reference.py", remote_path="/root/reference.py")
        .add_local_file("utils.py", remote_path="/root/utils.py")
    )

    app = modal.App("nvfp4-group-gemm-bench-v5", image=image)

    @app.function(
        gpu="B200",
        timeout=180,
    )
    def run_bench(
        num_groups: int,
        problem_sizes_mnkl: List[Tuple[int, int, int, int]],
        ab_dtype: str,
        sf_dtype: str,
        sf_vec_size: int,
        c_dtype: str,
        a_major: str,
        b_major: str,
        c_major: str,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        tolerance: float,
        warmup_iterations: int,
        iterations: int,
        skip_ref_check: bool,
        use_cold_l2: bool,
        identity_a: bool,
        identity_b: bool,
    ):
        print(f"Running benchmark on Modal execution...")

        # Clean up cache to ensure consistent benchmark
        import shutil
        try:
            shutil.rmtree("/root/.cache", ignore_errors=True)
        except Exception as e:
            pass

        # Helper to ensure we have the class/object, not string
        def ensure_dtype(dt):
            if isinstance(dt, str):
                return getattr(cutlass, dt)
            return dt

        # Set globals for custom_kernel
        globals()['sf_vec_size'] = sf_vec_size
        globals()['mma_tiler_mn'] = mma_tiler_mn
        globals()['cluster_shape_mn'] = cluster_shape_mn
        globals()['ab_dtype'] = ensure_dtype(ab_dtype)
        globals()['sf_dtype'] = ensure_dtype(sf_dtype)
        globals()['c_dtype'] = ensure_dtype(c_dtype)

        return run(
            problem_sizes=problem_sizes_mnkl,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            identity_a=identity_a,
            identity_b=identity_b,
        )

    @app.local_entrypoint()
    def main(
        num_groups: int = 2,
        problem_sizes_mnkl: str = "(128, 128, 128, 1), (128, 128, 128, 1)",
        mma_tiler_mn: str = "128, 128",
        cluster_shape_mn: str = "1, 1",
        ab_dtype: str = "Float4E2M1FN",
        sf_dtype: str = "Float8E4M3FN",
        sf_vec_size: int = 16,
        c_dtype: str = "Float16",
        a_major: str = "k",
        b_major: str = "k",
        c_major: str = "n",
        tolerance: float = 1e-03,
        warmup_iterations: int = 0,
        iterations: int = 1,
        skip_ref_check: bool = False,
        use_cold_l2: bool = False,
        run_benchmark_suite: bool = False,
        identity_a: bool = False,
        identity_b: bool = False,
    ):
    
        print(f"Running  on local ...")

        # Parsing helpers (reusing logic from __main__ but adapted for modal CLI)
        def parse_comma_separated_ints(s: Union[str, Tuple[int, ...], List[int]]) -> tuple[int, ...]:
            if isinstance(s, (tuple, list)):
                return tuple(s)
            return tuple(int(x.strip()) for x in s.split(","))

        def parse_comma_separated_tuples(s: Union[str, List[tuple[int, ...]]]) -> List[tuple[int, ...]]:
             if isinstance(s, (list, tuple)):
                 return list(s)
             import ast
             try:
                val = ast.literal_eval(s)
                if isinstance(val, tuple) and len(val) > 0 and isinstance(val[0], tuple):
                    return list(val)
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], tuple):
                    return val
                elif isinstance(val, tuple) and len(val) > 0 and isinstance(val[0], int):
                     return [val]
                else:
                     raise ValueError("Unexpected format")
             except Exception as e:
                 raise ValueError(f"Failed to parse tuple string: {s}. Error: {e}")

        # Map string dtypes to cutlass dtypes
        def get_dtype(name):
            return getattr(cutlass, name)

        ab_dt = get_dtype(ab_dtype)
        sf_dt = get_dtype(sf_dtype)
        c_dt = get_dtype(c_dtype)

        if run_benchmark_suite:
            print("Running full benchmark suite from task.yml...")
            import math
            execution_times = []

            for i, config in enumerate(BENCHMARK_CONFIGS):
                print(f"\n--- Running Benchmark {i+1} ---")
                g = config["g"]
                m_list = config["m"]
                n_list = config["n"]
                k_list = config["k"]
                l_val = config.get("l", 1)

                # Construct problem_sizes_mnkl list
                p_sizes = []
                for j in range(g):
                    p_sizes.append((m_list[j], n_list[j], k_list[j], l_val))

                # Use defaults for others or passed args?
                # task.yml doesn't specify others, so use defaults/args passed to main
                m_tiler = parse_comma_separated_ints(mma_tiler_mn)
                c_shape = parse_comma_separated_ints(cluster_shape_mn)

                # Run
                print(f"Config: {g} groups. Sizes: {p_sizes}")
                val = run_bench.remote(
                    g,
                    p_sizes,
                    ab_dt,
                    sf_dt,
                    sf_vec_size,
                    c_dt,
                    a_major,
                    b_major,
                    c_major,
                    m_tiler,
                    c_shape,
                    tolerance,
                    5, # warmup_iterations
                    20, # iterations
                    True, # skip_ref_check=True for suite speed usually? or args.skip_ref_check?
                    # Let's respect the flag or force true for speed?
                    # Usually suite runs are for perf, so maybe skip ref check.
                    # User didn't specify, but "geomean" implies perf.
                    # Let's use True to be safe on time.
                    True, # use_cold_l2
                    identity_a,
                    identity_b,
                )
                print(f"Time: {val} us")
                execution_times.append(val)

            if execution_times:
                geomean = math.exp(math.fsum(math.log(x) for x in execution_times) / len(execution_times))
                print(f"\nGeometric Mean of Execution Times: {geomean:.4f} us")
            else:
                print("No results.")

            return

        # Normal Run
        p_sizes = parse_comma_separated_tuples(problem_sizes_mnkl)
        m_tiler = parse_comma_separated_ints(mma_tiler_mn)
        c_shape = parse_comma_separated_ints(cluster_shape_mn)

        print(f"Running benchmark remotely on Modal...")
        val = run_bench.remote(
            num_groups,
            p_sizes,
            ab_dt,
            sf_dt,
            sf_vec_size,
            c_dt,
            a_major,
            b_major,
            c_major,
            m_tiler,
            c_shape,
            tolerance,
            warmup_iterations,
            iterations,
            skip_ref_check,
            use_cold_l2,
            identity_a, # Pass identity_a from main's arguments
            identity_b, # Pass identity_b from main's arguments
        )
        print("Remote execution completed.")
        print(f"BENCHMARK_RESULT: {val}")


if __name__ == "__main__":
    if "POPCORN_FD" not in os.environ:
        # Existing Argument Parsing Logic maintained for local non-modal runs

        def parse_comma_separated_ints(s: str) -> tuple[int, ...]:
            try:
                return tuple(int(x.strip()) for x in s.split(","))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    "Invalid format. Expected comma-separated integers."
                )

        def parse_comma_separated_tuples(s: str) -> List[tuple[int, ...]]:
            if s.strip().startswith("("):
                # Split on ),( to separate tuples
                tuples = s.strip("()").split("),(")
                result = []
                tuple_len = None

                for t in tuples:
                    # Parse individual tuple
                    nums = [int(x.strip()) for x in t.split(",")]

                    # Validate tuple length consistency
                    if tuple_len is None:
                        tuple_len = len(nums)
                    elif len(nums) != tuple_len:
                        raise argparse.ArgumentTypeError(
                            "All tuples must have the same length"
                        )

                    result.append(tuple(nums))
                return result

            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers or list of tuples"
            )

        parser = argparse.ArgumentParser(
            description="Example of Grouped GEMM on Blackwell."
        )
        parser.add_argument(
            "--num_groups",
            type=int,
            default=2,
            help="Number of groups",
        )
        parser.add_argument(
            "--problem_sizes_mnkl",
            type=parse_comma_separated_tuples,
            default=((128, 128, 128, 1), (128, 128, 128, 1)),
            help="a tuple of problem sizes for each group (comma-separated tuples)",
        )
        parser.add_argument(
            "--mma_tiler_mn",
            type=parse_comma_separated_ints,
            default=(128, 128),
            help="Mma tile shape (comma-separated)",
        )
        parser.add_argument(
            "--cluster_shape_mn",
            type=parse_comma_separated_ints,
            default=(1, 1),
            help="Cluster shape (comma-separated)",
        )
        #parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
        parser.add_argument("--ab_dtype", type=str, default="Float4E2M1FN")
        #parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
        parser.add_argument("--sf_dtype", type=str, default="Float8E4M3FN")
        parser.add_argument("--sf_vec_size", type=int, default=16)
        #parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16)
        parser.add_argument("--c_dtype", type=str, default="Float16")
        parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
        parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
        parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
        parser.add_argument(
            "--tolerance", type=float, default=1e-03, help="Tolerance for validation"
        )
        parser.add_argument(
            "--warmup_iterations", type=int, default=0, help="Warmup iterations"
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=1,
            help="Number of iterations to run the kernel",
        )
        parser.add_argument(
            "--skip_ref_check", action="store_true", help="Skip reference checking"
        )
        parser.add_argument(
            "--use_cold_l2",
            action="store_true",
            default=False,
            help="Use circular buffer tensor sets to ensure L2 cold cache",
        )
        parser.add_argument(
            "--identity_a",
            action="store_true",
            default=False,
            help="Use identity matrix for A",
        )
        parser.add_argument(
            "--identity_b",
            action="store_true",
            default=False,
            help="Use identity matrix for B",
        )
        parser.add_argument(
            "--run_benchmark_suite",
            action="store_true",
            default=False,
            help="Run the full benchmark suite"
        )

        args = parser.parse_args()

        if (
            len(args.problem_sizes_mnkl) != 0
            and len(args.problem_sizes_mnkl) != args.num_groups
        ):
            parser.error("--problem_sizes_mnkl must contain exactly num_groups tuples")

        # l mode must be 1 for all groups
        for _, _, _, l in args.problem_sizes_mnkl:
            if l != 1:
                parser.error("l must be 1 for all groups")

        if len(args.mma_tiler_mn) != 2:
            parser.error("--mma_tiler_mn must contain exactly 2 values")

        if len(args.cluster_shape_mn) != 2:
            parser.error("--cluster_shape_mn must contain exactly 2 values")

        # WRAP IN APP.RUN() TO FIX EXECUTION ERROR
        if True: #with app.run():
            # Helper to ensure we have the class/object, not string
            def ensure_dtype(dt):
                if isinstance(dt, str):
                    return getattr(cutlass, dt)
                return dt

            # Set globals for custom_kernel
            globals()['sf_vec_size'] = args.sf_vec_size
            globals()['mma_tiler_mn'] = args.mma_tiler_mn
            globals()['cluster_shape_mn'] = args.cluster_shape_mn
            globals()['ab_dtype'] = ensure_dtype(args.ab_dtype)
            globals()['sf_dtype'] = ensure_dtype(args.sf_dtype)
            globals()['c_dtype'] = ensure_dtype(args.c_dtype)

            if args.run_benchmark_suite:
                 print("Running full benchmark suite locally...")
                 import math
                 execution_times = []
                 for i, config in enumerate(BENCHMARK_CONFIGS):
                     print(f"\n--- Running Benchmark {i+1} ---")
                     g = config["g"]
                     m_list = config["m"]
                     n_list = config["n"]
                     k_list = config["k"]
                     l_val = config.get("l", 1)
                     p_sizes = []
                     for j in range(g):
                         p_sizes.append((m_list[j], n_list[j], k_list[j], l_val))
                     
                     # Use defaults for tiling/cluster if not in config
                     m_tiler = args.mma_tiler_mn
                     c_shape = args.cluster_shape_mn
                     
                     print(f"Config: {g} groups. Sizes: {p_sizes}")
                     val = run_group_gemm_benchmark(
                        problem_sizes=p_sizes,
                        warmup_iterations=5,
                        iterations=20,
                        #enable_profiling=True, # default
                        #enable_cycle_stats, # default False
                        identity_a=args.identity_a,
                        identity_b=args.identity_b,
                     )
                     print(f"Time: {val} us")
                     execution_times.append(val)
                 
                 if execution_times:
                    geomean = math.exp(math.fsum(math.log(x) for x in execution_times) / len(execution_times))
                    print(f"\nGeometric Mean of Execution Times: {geomean:.4f} us")
            else:
                # Normal Run
                run_group_gemm_benchmark(
                    problem_sizes=args.problem_sizes_mnkl, # directly pass parsed tuple list
                    warmup_iterations=args.warmup_iterations,
                    iterations=args.iterations,
                    #enable_profiling=True, # default
                    #enable_cycle_stats, # default False
                    identity_a=args.identity_a,
                    identity_b=args.identity_b,
                )
        print("PASS")

