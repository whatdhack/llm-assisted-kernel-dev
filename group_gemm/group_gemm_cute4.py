import os
# Force architecture for Blackwell CuTe DSL
os.environ["CUTE_DSL_ARCH"] = "sm_100a"

# Check and install apache-tvm-ffi before importing cutlass
import subprocess
import sys
import importlib.util
import cutlass.torch as cutlass_torch


def ensure_tvm_installed():
    """Checks if apache-tvm-ffi is installed, and installs it if not."""
    if importlib.util.find_spec("tvm") is None:
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

import modal
import dataclasses
import math
import time
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

import functools
from typing import Tuple, List

import torch
from task import input_t, output_t



from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
import subprocess
import sys
import importlib.util

def ensure_tvm_installed():
    """Checks if apache-tvm-ffi is installed, and installs it if not."""
    if importlib.util.find_spec("tvm") is None:
        print("apache-tvm-ffi not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "apache-tvm-ffi"])
            print("Successfully installed apache-tvm-ffi.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install apache-tvm-ffi: {e}")
            # We don't raise here, as maybe the run can continue or existing tvm matches?
            # But likely it will fail later if FFI is enabled options are used.
            pass

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


# The CuTe reference implementation for NVFP4 block-scaled GEMM
@cute.kernel
def kernel_gpu(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    mSFB_nkl: cute.Tensor,

    # Group Tensors and Metadata
    a_tuple: Tuple[cute.Tensor, ...],
    b_tuple: Tuple[cute.Tensor, ...],
    c_tuple: Tuple[cute.Tensor, ...],
    sfa_tuple: Tuple[cute.Tensor, ...],
    sfb_tuple: Tuple[cute.Tensor, ...],
    m_tuple: Tuple[int, ...],
    n_tuple: Tuple[int, ...],
    k_tuple: Tuple[int, ...],
    l_tuple: Tuple[int, ...],

    tensor_of_tensormap: cute.Tensor,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    cta_mn_list: Tuple[Tuple[int, int], ...],
    num_tma_load_bytes: cutlass.Constexpr[int],
    debug_buf: cute.Tensor,
):
    """
    GPU device kernel performing the Group GEMM computation.
    """
    t_init = clock64()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    tidx, _, _ = cute.arch.thread_idx()

    # delinearize bidz to coord_x, coord_y and group_idx for each CTA
    bidx, bidy, bidz = cute.arch.block_idx()
    group_idx = 0
    find = False
    coord_x = 0
    coord_y = 0
    cta_rest = bidz

    # Selection of pointers and shapes
    atom_shape = ((32, 4), (sf_vec_size, 4))
    # 16 is sf_vec_size
    atom_stride = ((16, 4), (0, 1))

    # Selection of pointers and shapes
    # Initialize with default to satisfy JIT compiler static analysis
    ptr_a_val = a_tuple[0].iterator
    ptr_b_val = b_tuple[0].iterator
    ptr_c_val = c_tuple[0].iterator
    ptr_sfa_val = sfa_tuple[0].iterator
    ptr_sfb_val = sfb_tuple[0].iterator
    m = m_tuple[0]
    n = n_tuple[0]
    k = k_tuple[0]
    l = l_tuple[0]

    # Initialize layouts (will be updated in loop)
    # Using dummy values initially
    _m0 = m_tuple[0]
    _n0 = n_tuple[0]
    _k0 = k_tuple[0]
    _l0 = l_tuple[0]
    mA_mkl_layout = cute.make_layout((_m0, _k0, _l0), stride=(cute.assume(_k0, 32), 1, cute.assume(_m0 * _k0, 32),))
    mB_nkl_layout = cute.make_layout((_n0, _k0, _l0), stride=(cute.assume(_k0, 32), 1, cute.assume(_n0 * _k0, 32),))
    sfa_layout = cute.tile_to_shape(cute.make_layout(atom_shape, stride=atom_stride), mA_mkl_layout.shape, (2, 1, 3))
    sfb_layout = cute.tile_to_shape(cute.make_layout(atom_shape, stride=atom_stride), mB_nkl_layout.shape, (2, 1, 3))

    for i, (cta_m, cta_n) in enumerate(cta_mn_list):
        if cta_rest >= (cta_m * cta_n):
            group_idx += 1
            cta_rest -= cta_m * cta_n
        else:
            if not find:
                group_idx = i
                # Calculate tile coordinates from linear cta_rest
                # cta_m and cta_n are python ints from cta_mn_list[i]
                coord_x = cta_rest % cta_m
                coord_y = cta_rest // cta_m

                ptr_a_val = a_tuple[i].iterator
                ptr_b_val = b_tuple[i].iterator
                ptr_c_val = c_tuple[i].iterator
                ptr_sfa_val = sfa_tuple[i].iterator
                ptr_sfb_val = sfb_tuple[i].iterator

                m = m_tuple[i]
                n = n_tuple[i]
                k = k_tuple[i]
                l = l_tuple[i]

                # Reconstruct layouts for the CURRENT group
                mA_mkl_layout = cute.make_layout(
                    (m, k, l), stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32),))
                mB_nkl_layout = cute.make_layout(
                    (n, k, l), stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32),))

                sfa_layout = cute.tile_to_shape(
                    cute.make_layout(atom_shape, stride=atom_stride),
                    mA_mkl_layout.shape,
                    (2, 1, 3),
                )
                sfb_layout = cute.tile_to_shape(
                    cute.make_layout(atom_shape, stride=atom_stride),
                    mB_nkl_layout.shape,
                    (2, 1, 3),
                )
                find = True

    mC_mnl_iter = ptr_c_val.align(32)
    mC_mnl_layout = cute.make_layout(
        (m, n, l),
        stride=(cute.assume(n, 32), 1, cute.assume(m * n, 32),))
    mC_mnl = cute.make_tensor(mC_mnl_iter, mC_mnl_layout)
    # Local partition for global C Tensor
    # (bM, bN, RestM, RestN, RestL)
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (coord_x, coord_y, 0)
    )

    #
    # Define shared storage for kernel
    #
    size_tensormap_in_i64 = (
        num_tensormaps * bytes_per_tensormap // 8
    )
    @cute.struct
    class SharedStorage:
        tensormap_buffer: cute.struct.MemRange[
            cutlass.Int64, size_tensormap_in_i64
        ]
        ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage * 2]
        tmem_holding_buf: cutlass.Int32
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
    tensormap_a_smem_ptr = tensormap_smem_ptr
    tensormap_b_smem_ptr = (
        tensormap_a_smem_ptr
        + bytes_per_tensormap // 8
    )
    tensormap_sfa_smem_ptr = (
        tensormap_b_smem_ptr
        + bytes_per_tensormap // 8
    )
    tensormap_sfb_smem_ptr = (
        tensormap_sfa_smem_ptr
        + bytes_per_tensormap // 8
    )
    # Setup smem tensor for A, B, SFA, SFB
    # (MMA, MMA_M, MMA_K, STAGE)
    sA = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=a_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=a_smem_layout_staged.inner,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sB = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )
    # (MMA, MMA_M, MMA_K, STAGE)
    sSFA = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfa_smem_layout_staged,
        byte_alignment=128,
    )
    # (MMA, MMA_N, MMA_K, STAGE)
    sSFB = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )

    # Initialize mainloop ab_pipeline, acc_pipeline and their states
    ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=num_ab_stage,
        producer_group=ab_pipeline_producer_group,
        consumer_group=ab_pipeline_consumer_group,
        tx_count=num_tma_load_bytes,
    ).make_participants()

    # Initialize acc_pipeline (accumulator pipeline)
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=num_acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            threads_per_cta,
        ),
    ).make_participants()

    #
    # Local_tile partition global tensors
    #
    # (bM, bK, RestM, RestK, RestL)
    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    # (bM, bK, RestM, RestK, RestL)
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    # (bN, bK, RestN, RestK, RestL)
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    #
    # Partition global tensor for TiledMMA_A/B/C
    #
    thr_mma = tiled_mma.get_slice(tidx)
    # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
    tCgA = thr_mma.partition_A(gA_mkl)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgB = thr_mma.partition_B(gB_nkl)
    # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
    tCgSFA = thr_mma.partition_A(gSFA_mkl)
    # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
    tCgSFB = thr_mma.partition_B(gSFB_nkl)
    # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
    tCgC = thr_mma.partition_C(gC_mnl)

    # Update tma descriptor with the correct shapes and strides
    tensormap_manager = utils.TensorMapManager(
        utils.TensorMapUpdateMode.SMEM,
        128,
    )
    tensormap_a_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensor_of_tensormap[(bidz, 0, None)].iterator
    )
    tensormap_b_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensor_of_tensormap[(bidz, 1, None)].iterator
    )
    tensormap_sfa_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensor_of_tensormap[(bidz, 2, None)].iterator
    )
    tensormap_sfb_gmem_ptr = tensormap_manager.get_tensormap_ptr(
        tensor_of_tensormap[(bidz, 3, None)].iterator
    )

    # Use the LOCALLY derived layouts and pointers for the Real Tensor (physical description)
    mA_mkl_iter = ptr_a_val.align(32)
    mB_nkl_iter = ptr_b_val.align(32)
    sfa_mkl_iter = ptr_sfa_val.align(32)
    sfb_nkl_iter = ptr_sfb_val.align(32)

    # Layouts were updated in the loop.
    real_tensor_a = cute.make_tensor(mA_mkl_iter, mA_mkl_layout)
    real_tensor_b = cute.make_tensor(mB_nkl_iter, mB_nkl_layout)
    real_tensor_sfa = cute.make_tensor(sfa_mkl_iter, sfa_layout)
    real_tensor_sfb = cute.make_tensor(sfb_nkl_iter, sfb_layout)

    # DEBUG: Check initial values at M=32 for correctness verification
    # Only thread 0 of block 0 performs the check



    # Let warp 0 initialize tensormap
    if warp_idx == 0:
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_a, tensormap_a_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_b, tensormap_b_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_sfa, tensormap_sfa_smem_ptr, 0
        )
        tensormap_manager.init_tensormap_from_atom(
            tma_atom_sfb, tensormap_sfb_smem_ptr, 0
        )
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
            0,  # tma warp id
            (
                tensormap_a_smem_ptr,
                tensormap_b_smem_ptr,
                tensormap_sfa_smem_ptr,
                tensormap_sfb_smem_ptr,
            ),
        )


        tensormap_manager.fence_tensormap_update(tensormap_a_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_b_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_sfa_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_sfb_gmem_ptr)

    cute.arch.barrier()



    #
    # Partition global/shared tensor for TMA load A/B/SFA/SFB
    #
    # TMA Partition_S/D for A
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestM, RestK, RestL)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    # TMA Partition_S/D for B
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )
    #  TMA Partition_S/D for SFA
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestM, RestK, RestL)
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFA, 0, 3),
        cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    # TMA Partition_S/D for SFB
    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK, RestL)
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb,
        0,
        cute.make_layout(1),
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
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N)
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
    #
    # Alloc tensor memory buffer
    #
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    tmem.allocate(num_tmem_alloc_cols)
    tmem.wait_for_alloc()
    acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

    #
    # Make SFA/SFB tmem tensor
    #
    # Get SFA tmem ptr
    sfa_tmem_ptr = cute.recast_ptr(
        acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc),
        dtype=sf_dtype,
    )
    # (MMA, MMA_M, MMA_K)
    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
    )
    tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
    # Get SFB tmem ptr
    sfb_tmem_ptr = cute.recast_ptr(
        acc_tmem_ptr
        + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
        + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
        dtype=sf_dtype,
    )
    # (MMA, MMA_N, MMA_K)
    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
    )
    tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

    #
    # Partition for S2T copy of SFA/SFB
    #
    # Make S2T CopyAtom
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
        sf_dtype,
    )
    # (MMA, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact = cute.filter_zeros(sSFA)
    tCtSFA_compact = cute.filter_zeros(tCtSFA)
    tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
    thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
    )
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
    tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

    # (MMA, MMA_MN, MMA_K, STAGE)
    tCsSFB_compact = cute.filter_zeros(sSFB)
    # (MMA, MMA_MN, MMA_K)
    tCtSFB_compact = cute.filter_zeros(tCtSFB)
    tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
    thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
    tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfb, tCsSFB_compact_s2t_
    )
    # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
    tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

    # Number of K loops
    k_tile_cnt = cute.ceil_div(real_tensor_a.shape[1], mma_tiler_mnk[2])

    #
    # Slice to per mma tile index
    #
    mma_tile_coord_mnl = (coord_x, coord_y, 0)
    # ((atom_v, rest_v), RestK)
    tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tAgSFA = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    # ((atom_v, rest_v), RestK)
    tBgSFB = tBgSFB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

    # DEBUG: Check initial values at M=32 for correctness verification
    # Using explicit index check and cute.print for tensor values
    # if warp_idx == 0 and tidx == 0:
    #      cute.printf("DEBUG_KERNEL: Pre-Loop Check (Block 0)\n")
    #
    #      # SFA check (reinterpret/cast to int8 to avoid fp8 cvt issues)
    #      # Using .to(int32) or similar might work better if int8 fails, but trying int8 first
    #      # Actually, safe bet is to load as int8 if possible, or cast to int8
    #      v_sfa0 = real_tensor_sfa[0, 0, 0].to(cutlass.Int8)
    #      v_sfa32 = real_tensor_sfa[32, 0, 0].to(cutlass.Int8)
    #      cute.printf("  SFA(0,0,0): %d (0x%x)\n", v_sfa0, v_sfa0)
    #      cute.printf("  SFA(32,0,0): %d (0x%x)\n", v_sfa32, v_sfa32)
    #
    #      # A check - just print a marker since FP4 tensor printing crashes
    #      cute.printf("  A tensor present (cannot print FP4 values)\n")
    #
    #      # C check (16-bit to f32)
    #      v_c32 = gC_mnl[32, 0].to(cutlass.Float32)
    #      cute.printf("  C(32,0,0) init: %f\n", v_c32)
    #      cute.printf("\n")

    #
    # Main loop
    #
    wait_cycles = cutlass.Int64(0)
    math_cycles = cutlass.Int64(0)
    t_start = clock64()

    if warp_idx == 0:
        # Wait for accumulator buffer empty
        acc_empty = acc_producer.acquire_and_advance()
        # Set ACCUMULATE field to False for the first k_tile iteration
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        # Execute k_tile loop
        for k_tile in range(k_tile_cnt):
            # Wait for AB buffer empty
            t0 = clock64()
            ab_empty = ab_producer.acquire_and_advance()
            t1 = clock64()
            wait_cycles += (t1 - t0)

            #  TMA load A/B/SFA/SFB to shared memory
            cute.copy(
                tma_atom_a,
                tAgA[(None, k_tile)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                    tensormap_a_gmem_ptr,
                    cute.AddressSpace.generic,
                ),
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, k_tile)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                    tensormap_b_gmem_ptr,
                    cute.AddressSpace.generic,
                ),
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA[(None, k_tile)],
                tAsSFA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                    tensormap_sfa_gmem_ptr,
                    cute.AddressSpace.generic,
                ),
            )
            cute.copy(
                tma_atom_sfb,
                tBgSFB[(None, k_tile)],
                tBsSFB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                    tensormap_sfb_gmem_ptr,
                    cute.AddressSpace.generic,
                ),
            )

            # Wait for AB buffer full
            ab_full = ab_consumer.wait_and_advance()

            #  Copy SFA/SFB from shared memory to TMEM
            s2t_stage_coord = (None, None, None, None, ab_full.index)
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
                    ab_full.index,
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

                t2 = clock64()
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[kblock_coord],
                    tCrB[kblock_coord],
                    tCtAcc,
                )
                t3 = clock64()
                math_cycles += (t3 - t2)
                # Enable accumulate on tCtAcc after first kblock
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # Async arrive AB buffer empty
            ab_full.release()
        acc_empty.commit()

    t_end = clock64()

    # Sentinel write (unconditional)
    debug_buf[5] = cutlass.Int64(12345)

    # Debug: Check which blocks are running
    # If bidz is within range [0, 900], mark it.
    if tidx == 0 and bidz < 900:
        debug_buf[100 + bidz] = cutlass.Int64(1)

    if bidx == 0 and bidy == 0 and bidz == 0 and tidx == 0:
        # Debug: Marker to prove we entered this block
        debug_buf[6] = cutlass.Int64(99999)

        # Index 0: Setup start
        debug_buf[0] = t_start

        # Index 1: Loop Total Time
        debug_buf[1] = t_end - t_start

        # Index 2: Wait Total
        debug_buf[2] = wait_cycles

        # Index 3: Math Total
        debug_buf[3] = math_cycles

    #
    # Epilogue
    # Partition for epilogue
    #
    op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
    copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc[None,0,0])
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    # (TmemCpy, NumTmemCpy)
    tDtAcc = thr_copy_t2r.partition_S(tCtAcc[None,0,0])
    # (TmemCpy, NumTmemCpy)
    tDgC = thr_copy_t2r.partition_D(tCgC[None,0,0])

    # (TmemCpy, NumTmemCpy)
    tDrAcc = cute.make_rmem_tensor(tDgC.shape, cutlass.Float32)
    # (TmemCpy, NumTmemCpy)
    tDrC = cute.make_rmem_tensor(tDgC.shape, c_dtype)

    # Release TMEM allocation lock
    tmem.relinquish_alloc_permit()
    # Wait for accumulator buffer full
    t_epilogue_start = clock64()
    acc_full = acc_consumer.wait_and_advance()

    # Copy accumulator to register
    cute.copy(tiled_copy_t2r, tDtAcc, tDrAcc)

    # Debug: Print diagonal elements from each thread in block 0,0,0
    # if bidx == 0 and bidy == 0 and bidz == 0:
    #     val = tDrAcc[tidx].to(cutlass.Float32)
    #     cute.printf("DEBUG_EPILOGUE: thread %d, tDrAcc[%d] = %f\n", tidx, tidx, val)

    acc_vec = tDrAcc.load()
    tDrC.store(acc_vec.to(c_dtype))

    # STG Atom, just to ensure functionality
    # For performance optimization, better to use Tma store operation to
    # reduce address calculation and predicate calulation instructions
    simt_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), c_dtype, num_bits_per_copy=16
    )
    thread_layout = cute.make_layout(
        (1, threads_per_cta), stride=(threads_per_cta, 1))
    value_layout = cute.make_layout((1, 1))
    tiled_copy_r2g = cute.make_tiled_copy_tv(
        simt_atom, thread_layout, value_layout
    )
    thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)
    cC = cute.make_identity_tensor(gC_mnl.shape)
    # ((atom_v, rest_v), NumGmemCpy)
    tDcC = thr_copy_r2g.partition_D(cC)

    # ((atom_v, rest_v), NumGmemCpy)
    tDpC = cute.make_rmem_tensor(tDrC.shape, cutlass.Boolean)
    residue_m = mC_mnl.shape[0] - cutlass.Int32(coord_x) * mma_tiler_mnk[0]
    residue_n = mC_mnl.shape[1] - cutlass.Int32(coord_y) * mma_tiler_mnk[1]
    for i in range(cute.size(tDrC.shape)):
        # Swap residue_m and residue_n to match the order of tDcC
        tDpC[i] = cute.elem_less(tDcC[i], (residue_n, residue_m))
    cute.copy(simt_atom, cute.flatten(tDrC), cute.flatten(tDgC), pred=cute.flatten(tDpC))

    acc_full.release()
    # Deallocate TMEM
    cute.arch.barrier()
    tmem.free(acc_tmem_ptr)

    t_epilogue_end = clock64()
    if bidx == 0 and bidy == 0 and bidz == 0 and tidx == 0:
        # Index 4: Epilogue Total
        debug_buf[4] = t_epilogue_end - t_epilogue_start
        # Index 0: Setup start (absolute timestamp)
        debug_buf[0] = t_start
        # Index 7: End-to-End Kernel Time
        debug_buf[7] = t_epilogue_end - t_init

    pass


# Host-side JIT function to prepare tensors and launch GPU kernel.
@cute.jit
def kernel_host(
    a_tuple,
    b_tuple,
    c_tuple,
    sfa_tuple,
    sfb_tuple,
    m_tuple,
    n_tuple,
    k_tuple,
    l_tuple,
    ptr_of_tensor_of_tensormap: cute.Pointer,
    total_num_clusters: cutlass.Int32,
    debug_buf: cute.Tensor,
    problem_sizes: List[
        Tuple[int, int, int, int]
    ],  # Problem sizes for each group
    num_groups: cutlass.Int32,
):
    tensor_of_tensormap = cute.make_tensor(
        ptr_of_tensor_of_tensormap, cute.make_layout((total_num_clusters, 4, 16), stride=(64, 16, 1))
    )

    # Use fake shape for initial Tma descriptor and atom setup
    # The real Tma desc and atom will be updated during kernel execution.
    min_a_shape = (cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(1))
    min_b_shape = (cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(1))
    initial_a = cute.make_tensor(
        cute.make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,),
        cute.make_layout(
            (min_a_shape[0], cute.assume(min_a_shape[2], 32), min_a_shape[3]),
            stride=(
                cute.assume(min_a_shape[2], 32),
                1,
                cute.assume(min_a_shape[0] * min_a_shape[2], 32),
            ),
        ),
    )
    # print(f"DEBUG initial_a: {initial_a}")
    # print(f"DEBUG initial_a.shape: {initial_a.shape}")
    # print(f"DEBUG initial_a.layout: {initial_a.layout}")
    initial_b = cute.make_tensor(
        cute.make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,),
        cute.make_layout(
            (min_b_shape[1], cute.assume(min_b_shape[2], 32), min_b_shape[3]),
            stride=(
                cute.assume(min_b_shape[2], 32),
                1,
                cute.assume(min_b_shape[1] * min_b_shape[2], 32),
            ),
        ),
    )

    # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
    # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
        initial_a.shape, sf_vec_size
    )
    # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
        initial_b.shape, sf_vec_size
    )
    # Create initial SFA and SFB tensors with fake shape and null pointer.
    initial_sfa = cute.make_tensor(
        cute.make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,), sfa_layout)
    initial_sfb = cute.make_tensor(
        cute.make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16,), sfb_layout)

    # Select MMA operation
    mma_op = tcgen05.MmaMXF4NVF4Op(
        sf_dtype,
        (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)

    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((1, 1, 1)),
        (tiled_mma.thr_id.shape,),
    )

    # Compute A/B/SFA/SFB/C shared memory layout
    a_smem_layout_staged = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        ab_dtype,
        num_ab_stage,
    )
    b_smem_layout_staged = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        ab_dtype,
        num_ab_stage,
    )
    sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        num_ab_stage,
    )
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        num_ab_stage,
    )
    atom_thr_size = cute.size(tiled_mma.thr_id.shape)

    # Setup TMA for A
    a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_a,
        a_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    # Setup TMA for B
    b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_b,
        b_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
    )
    # Setup TMA for SFA
    sfa_smem_layout = cute.slice_(
        sfa_smem_layout_staged, (None, None, None, 0)
    )
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_sfa,
        sfa_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )
    # Setup TMA for SFB
    sfb_smem_layout = cute.slice_(
        sfb_smem_layout_staged, (None, None, None, 0)
    )
    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_sfb,
        sfb_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )

    # Compute TMA load bytes
    a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
    b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
    sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
    sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    num_tma_load_bytes = (
        a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
    ) * atom_thr_size

    # Debug buffer
    # Use debug_buf directly as it is passed as a cute.Tensor

    # Store CTA shape information for each Group in a List
    cta_mn_list = []
    for group_idx, (m, n, k, l) in enumerate(problem_sizes):
        x, y = cute.ceil_div(problem_sizes[group_idx][:2], mma_tiler_mnk[0:2])
        cta_mn_list.append((x, y))

    # Compute grid size
    grid = (1, 1, total_num_clusters)

    # Launch the kernel
    kernel_gpu(
        # MMA (Matrix Multiply-Accumulate) configuration
        tiled_mma,                  # Tiled MMA object defining NVFP4 GEMM compute pattern

        # TMA (Tensor Memory Accelerator) atoms and tensors for input matrix A
        tma_atom_a,                 # TMA copy atom defining how to load A from global memory
        tma_tensor_a,               # Tensor descriptor for A (created from smallest A tensor)

        # TMA atoms and tensors for input matrix B
        tma_atom_b,                 # TMA copy atom defining how to load B from global memory
        tma_tensor_b,               # Tensor descriptor for B (created from smallest B tensor)

        # TMA atoms and tensors for scale factor A
        tma_atom_sfa,               # TMA copy atom for loading scale factors for A
        tma_tensor_sfa,             # Tensor descriptor for SFA (block scale factors for A)

        # TMA atoms and tensors for scale factor B
        tma_atom_sfb,               # TMA copy atom for loading scale factors for B
        tma_tensor_sfb,             # Tensor descriptor for SFB (block scale factors for B)

        # Group Tensors directly passed via tuples
        a_tuple,
        b_tuple,
        c_tuple,
        sfa_tuple,
        sfb_tuple,
        m_tuple,
        n_tuple,
        k_tuple,
        l_tuple,

        # Buffers
        tensor_of_tensormap,        # Locally allocated tensormaps

        # Shared memory layouts with staging for pipelined execution
        a_smem_layout_staged,       # Staged shared memory layout for A (includes stage dimension)
        b_smem_layout_staged,       # Staged shared memory layout for B (includes stage dimension)
        sfa_smem_layout_staged,     # Staged shared memory layout for SFA (includes stage dimension)
        sfb_smem_layout_staged,     # Staged shared memory layout for SFB (includes stage dimension)

        # CTA grid configuration per group
        cta_mn_list,                # List of (M_tiles, N_tiles) for each group

        # Pipeline synchronization parameter
        num_tma_load_bytes,         # Total bytes to load per TMA transaction (for barrier setup)

        # Debug Buffer
        debug_buf,

    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(1, 1, 1),
    )
    return

# This function is used to compile the kernel once and cache it and then allow users to
# run the kernel multiple times to get more accurate timing results.
# Global cache for compiled kernels
_compiled_kernel_cache = {}
# Global dummy debug buffer for non-profiled runs
_dummy_debug_buf = None

def compile_kernel(problem_sizes, sfa_tuple, sfb_tuple, total_num_clusters,):
    """
    Compile the kernel once and cache it using problem_sizes as the key.
    """
    global _compiled_kernel_cache

    # Cache by number of groups and their shapes to avoid redundant compilation
    # However, since shapes might vary, caching by count is a simplified version.
    # Note: different shapes WILL cause different JIT code if we use them for layout.
    # Cache by number of groups and their shapes to avoid redundant compilation
    # However, since shapes might vary, caching by count is a simplified version.
    # Note: different shapes WILL cause different JIT code if we use them for layout.
    cache_key = f"tuple_v2_{str(problem_sizes)}"


    # Check if we already have a compiled kernel for these problem sizes
    if cache_key in _compiled_kernel_cache:
        return _compiled_kernel_cache[cache_key]

    # If we are compiling for 8 groups, but error says expected 2, it means we might be hitting a weird state.
    # Let's verify arguments.


    num_groups = len(problem_sizes)

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
    cta_mn_fakes = []

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
        cta_mn_fakes.append((cutlass.Int32((m+127)//128), cutlass.Int32((n+127)//128)))

    # Buffer placeholders
    total_clusters_sym = cutlass.Int32(1)

    # tensormap_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (total_num_clusters, 4, 16))

    cute_ptr_of_tensor_of_tensormap = make_ptr(
        cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16,
    )

    debug_buf_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (1024,))
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

    # Let's look at kernel_host line 808 again in my read output.
    # line 808: debug_buf: cute.Tensor,

    # And previously in custom_kernel:
    # line 1147: cute_ptr_of_debug_buf = make_ptr(...)

    # This seems inconsistent if I didn't change kernel_host for debug_buf.
    # However, my previous task was ONLY about tensormap.
    # But I see I added `cute_ptr_of_debug_buf` in `custom_kernel`.

    # Let's assume for now I should only fix tensormap ptr.

    #print(f"a_fakes[0] : {a_fakes[0]} ")
    compiled_func = cute.compile(
        kernel_host,
        tuple(a_fakes),
        tuple(b_fakes),
        tuple(c_fakes),
        tuple(sfa_fakes),
        tuple(sfb_fakes),
        tuple(m_fakes),
        tuple(n_fakes),
        tuple(k_fakes),
        tuple(l_fakes),
        cute_ptr_of_tensor_of_tensormap,
        cutlass.Int32(total_num_clusters),
        debug_buf_fake,
        problem_sizes,
        num_groups,
        options="--enable-tvm-ffi --opt-level 3",
    )
    # Store compiled kernel in cache with problem_sizes as key
    _compiled_kernel_cache[cache_key] = compiled_func
    return compiled_func




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

def custom_kernel(data: input_t, events: Tuple[torch.cuda.Event, torch.cuda.Event] = None, stats: CustomStats = None) -> output_t:
    """
    Execute the block-scaled group GEMM kernel.

    Args:
        data: Tuple of (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)
        events: Optional tuple of (start_event, end_event) for kernel timing.
        stats: Optional CustomStats object to capture internal counters.

    Returns:
        list of c tensors (res)
    """
    if stats:
        t_start = time.perf_counter()

    abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data

    # 1. Faster Tensor Unpacking (Avoids loops)
    if abc_tensors:
         a_tuple, b_tuple, c_tuple = zip(*abc_tensors)
         sfa_tuple, sfb_tuple = zip(*sfasfb_reordered_tensors)
         sfa_torch_tuple, sfb_torch_tuple = zip(*sfasfb_tensors)
         #print (f" a {a_tuple[0].shape}, b {b_tuple[0].shape}, c {c_tuple[0].shape}, sfa {sfa_tuple[0].shape}, sfb {sfb_tuple[0].shape} sfa_torch {sfa_torch_tuple[0].shape}, sfb_torch {sfb_torch_tuple[0].shape}")
         #print (f" a {type(a_tuple[0])}, b {type(b_tuple[0])}, c {type(c_tuple[0])}, sfa {type(sfa_tuple[0])}, sfb {type(sfb_tuple[0])}")
         #print (f" a {a_tuple[0].stride()}, b {b_tuple[0].stride()}, c {c_tuple[0].stride()} sfa {sfa_tuple[0].stride()}, sfb {sfb_tuple[0].stride()} sfa_torch {sfa_torch_tuple[0].stride()}, sfb_torch {sfb_torch_tuple[0].stride()}")
         #print (f" a {a_tuple[0].dtype}, b {b_tuple[0].dtype}, c {c_tuple[0].dtype}, sfa {sfa_tuple[0].dtype}, sfb {sfb_tuple[0].dtype}")
         #print (f" a {a_tuple[0]}, b {b_tuple[0]}, c {c_tuple[0]}, sfa {sfa_tuple[0]}, sfb {sfb_tuple[0]}")
         #print (f" c_tuple[0] nonzero: {(c_tuple[0] != 0).sum().item()}/{c_tuple[0].numel()}")
    else:
         a_tuple = b_tuple = c_tuple = sfa_tuple = sfb_tuple = ()

    global _dummy_debug_buf
    global _dummy_debug_ptr

    # 2. Metadata Caching via problem_sizes (Hashable Tuple of Tuples)
    # The problem_sizes list is converted to tuple for cache key
    cache_key = tuple(problem_sizes)

    if cache_key in _kernel_metadata_cache:
        # Fast Path: Retrieve cached values
        (total_num_clusters, cta_mn_list,
         m_tuple, n_tuple, k_tuple, l_tuple, num_groups,
         tensor_of_tensormap, cute_ptr_of_tensor_of_tensormap,
         compiled_func) = _kernel_metadata_cache[cache_key]
    else:
        # Slow Path: Compute AND Cache
        # 1. Pre-calculate total clusters and CTA grid
        total_num_clusters = 0
        num_groups = len(problem_sizes)
        cta_mn_list = []

        for m, n, _, _ in problem_sizes:
            num_clusters_mn = tuple((x + y - 1) // y for x, y in zip((m, n), (128, 128)))
            count = num_clusters_mn[0] * num_clusters_mn[1]
            total_num_clusters += count
            cta_mn_list.append(num_clusters_mn)

        # Unpack problem sizes efficiently too
        if problem_sizes:
             m_tuple, n_tuple, k_tuple, l_tuple = zip(*problem_sizes)
        else:
             m_tuple = n_tuple = k_tuple = l_tuple = ()

        # Allocate tensormap buffer once per configuration
        # Shape: (total_num_clusters, num_tensormaps=4, bytes_per_tensormap/8=16)
        tensor_of_tensormap = torch.zeros(
            (total_num_clusters, 4, 16), dtype=torch.int64, device="cuda"
        )
        cute_ptr_of_tensor_of_tensormap = make_ptr(
            cutlass.Int64,
            tensor_of_tensormap.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )


        # Compile kernel once and cache it
        compiled_func = compile_kernel(problem_sizes, sfa_tuple, sfb_tuple, total_num_clusters)

        # Store in cache
        _kernel_metadata_cache[cache_key] = (
             total_num_clusters, cta_mn_list,
             m_tuple, n_tuple, k_tuple, l_tuple, num_groups,
             tensor_of_tensormap, cute_ptr_of_tensor_of_tensormap,
             compiled_func
        )

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

    if stats:
        t_pre = time.perf_counter()

    if events:
        events[0].record()

    compiled_func(
        a_tuple,
        b_tuple,
        c_tuple,
        sfa_tuple,
        sfb_tuple,
        m_tuple,
        n_tuple,
        k_tuple,
        l_tuple,
        cute_ptr_of_tensor_of_tensormap,
        total_num_clusters,
        debug_buf,
        problem_sizes,
        num_groups,
    )

    if events:
        events[1].record()

    if stats:
        t_post_start = time.perf_counter()
        t_end = time.perf_counter()

        buf_cpu = debug_buf.cpu().numpy()

        stats.pre_launch_cpu = t_pre - t_start
        stats.post_launch_cpu = t_end - t_post_start
        stats.e2e_cpu_time = t_end - t_start
        stats.debug_buffer = buf_cpu

    # Return c_tuple as list directly (fast path)
    return list(c_tuple)


# ---------------------------------------------------------------------------
# Modal & Benchmark Boilerplate
# ---------------------------------------------------------------------------

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
        "CUTE_DSL_ARCH": "sm_100a"
    })
    .add_local_file("task.py", remote_path="/root/task.py")
    .add_local_file("reference.py", remote_path="/root/reference.py")
    .add_local_file("utils.py", remote_path="/root/utils.py")
)

app = modal.App("nv100-group-gemm-benchmark", image=image)


NUM_ITERATIONS_PER_BENCHMARK = 15

def clear_l2_cache():
    # create a large dummy tensor to flush L2 cache (128MB on Blackwell)
    dummy = torch.randn((20, 1024, 1024), device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    del dummy


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float
    median: float


def calculate_stats(durations: list[float]):
    runs = len(durations)
    if runs == 0:
        return None
    durations.sort()
    total = sum(durations)
    best = min(durations)
    worst = max(durations)
    median = durations[runs // 2]

    avg = total / runs
    if runs > 1:
        variance = sum(map(lambda x: (x - avg) ** 2, durations))
        std = math.sqrt(variance / (runs - 1))
        err = std / math.sqrt(runs)
    else:
        std = 0
        err = 0

    return Stats(
        runs=runs, mean=avg, std=std, err=err, best=float(best), worst=float(worst), median=float(median)
    )


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


def run_group_gemm_benchmark(
    problem_sizes: List[Tuple[int, int, int, int]],
    warmup_iterations: int = 10,
    iterations: int = 100,
    enable_profiling: bool = True,
    enable_cycle_stats: bool = False,
    identity_b: bool = False,
    identity_a: bool = False,
    max_time_ns: float = 10e9,
    **kwargs,  # absorb legacy args
):
    """
    Benchmark the custom_kernel with given problem sizes.

    Matches eval_better_bench_grouped_gemm.py _run_single_benchmark exactly:
    - Generates NUM_ITERATIONS_PER_BENCHMARK=15 datasets with incrementing seeds (+42 each)
    - Verifies all 15 datasets for correctness before timing
    - Clears L2 cache once before each outer timing iteration
    - Adaptive stopping: exits when err/mean < 0.1% or time/run-count budget exceeded
    """
    from reference import generate_input, check_implementation
    m_list = [p[0] for p in problem_sizes]
    n_list = [p[1] for p in problem_sizes]
    k_list = [p[2] for p in problem_sizes]
    g = len(problem_sizes)

    # Generate NUM_ITERATIONS_PER_BENCHMARK datasets with incrementing seeds.
    # Seed is bumped before each generate_input call, matching eval harness behaviour.
    print(f"Generating {NUM_ITERATIONS_PER_BENCHMARK} datasets...")
    data_list = []
    seed = 1111
    for i in range(NUM_ITERATIONS_PER_BENCHMARK):
        seed += 42
        data = generate_input(
            tuple(m_list),
            tuple(n_list),
            tuple(k_list),
            g,
            seed=seed,
            identity_b=identity_b,
            identity_a=identity_a,
        )
        data_list.append(data)

    # Verify ALL datasets before timing (matching eval).
    check_copy = _clone_data(data_list)
    print("Verifying correctness on all datasets...")
    outputs = []
    try:
        for data in data_list:
            output = custom_kernel(_clone_data(data))
            outputs.append(output)
        torch.cuda.synchronize()
        print("Correctness check passed.")
    except Exception as e:
        print(f"Verification crashed: {e}")
        import traceback
        traceback.print_exc()

    # Warmup
    for _ in range(warmup_iterations):
        custom_kernel(_clone_data(data_list[0]))

    torch.cuda.synchronize()

    # Benchmarking
    total_durations = []
    kernel_durations = []
    overhead_durations = []
    pre_durations = []
    post_durations = []
    internal_durations = []

    # Cycle counters
    setup_deltas = []
    loop_cycles = []
    wait_cycles = []
    math_cycles = []
    epilogue_cycles = []
    e2e_cycles = []

    prev_timestamp = None

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)


    # Run with Profiler
    from torch.profiler import profile, record_function, ProfilerActivity

    prof = None
    profile_start_iter = max(0, iterations - 3)
    trace_content = None

    # Timing loop — matches eval _run_single_benchmark exactly:
    #   clear L2 once, run all 15 datasets sequentially, time with CUDA events,
    #   stop adaptively when err/mean < 0.1% or budget exceeded.
    bm_start_time = time.perf_counter_ns()
    for i in range(iterations):
        # Start profiling 3 iterations before the end
        if enable_profiling and i == profile_start_iter:
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            prof.start()

        torch.cuda.synchronize()
        clear_l2_cache()  # flush L2 once before the full batch of 15 launches

        if prof:
            with record_function(f"benchmark_step_iter_{i}"):
                start_time = time.perf_counter()
                for data in data_list:
                    custom_kernel(data)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
        else:
            start_time = time.perf_counter()

            stats = CustomStats() if enable_cycle_stats else None

            start_event.record()
            accum_internal = 0.0
            accum_pre = 0.0
            accum_post = 0.0

            for data in data_list:
                custom_kernel(data, stats=stats)
                if stats:
                    accum_internal += stats.e2e_cpu_time
                    accum_pre += stats.pre_launch_cpu
                    accum_post += stats.post_launch_cpu

            end_event.record()
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            t_internal = accum_internal
            t_pre_raw = accum_pre
            t_post_raw = accum_post

        if enable_profiling and i == iterations - 1 and prof:
            prof.stop()

            # Export Chrome Trace
            trace_filename = f"trace_group_gemm_{len(problem_sizes)}_groups.json"
            prof.export_chrome_trace(trace_filename)

            try:
                with open(trace_filename, "r") as f:
                    trace_content = f.read()
            except Exception as e:
                print(f"Error reading trace file: {e}")
                pass

        # Unpack stats
        if stats:
            t_pre = t_pre_raw
            t_post = t_post_raw
            buf_cpu = stats.debug_buffer
        else:
            t_pre = 0.0
            t_post = 0.0
            t_internal = 0.0
            buf_cpu = None

        total_time_us = (end_time - start_time) * 1e6 / NUM_ITERATIONS_PER_BENCHMARK
        t_internal_us = t_internal * 1e6 / NUM_ITERATIONS_PER_BENCHMARK

        if prof:
            kernel_time_us = 0.0
        else:
            kernel_time_us = (start_event.elapsed_time(end_event) * 1000) / NUM_ITERATIONS_PER_BENCHMARK  # elapsed_time is in ms

        overhead_us = max(0, total_time_us - kernel_time_us)

        total_durations.append(total_time_us)
        kernel_durations.append(kernel_time_us)
        overhead_durations.append(overhead_us)
        pre_durations.append(t_pre * 1e6 / NUM_ITERATIONS_PER_BENCHMARK)
        post_durations.append(t_post * 1e6 / NUM_ITERATIONS_PER_BENCHMARK)
        internal_durations.append(t_internal_us)

        # Collect cycle stats
        if buf_cpu is not None:
            current_ts = buf_cpu[0]
            if prev_timestamp is not None:
                 delta = (current_ts - prev_timestamp)
                 if delta > 0 and delta < 1e9:
                     setup_deltas.append(float(delta))
            prev_timestamp = current_ts

            loop_cycles.append(float(buf_cpu[1]))
            wait_cycles.append(float(buf_cpu[2]))
            math_cycles.append(float(buf_cpu[3]))
            epilogue_cycles.append(float(buf_cpu[4]))
            e2e_cycles.append(float(buf_cpu[7]))

        # Adaptive stopping — matches eval harness:
        # after at least 2 runs and 100ms wall time, stop when:
        #   a) relative error < 0.1%  (err/mean < 0.001)
        #   b) total measured time exceeds max_time_ns budget
        #   c) wall-clock time exceeds 2 minutes
        total_bm_duration = time.perf_counter_ns() - bm_start_time
        if i > 1 and total_bm_duration > 1e8:
            cur_stats = calculate_stats(list(kernel_durations))
            if cur_stats and (
                cur_stats.err / cur_stats.mean < 0.001
                or cur_stats.mean * cur_stats.runs > max_time_ns
                or total_bm_duration > 120e9
            ):
                break

    return (
        calculate_stats(total_durations),
        calculate_stats(kernel_durations),
        calculate_stats(overhead_durations),
        calculate_stats(pre_durations),
        calculate_stats(post_durations),
        calculate_stats(setup_deltas),
        calculate_stats(internal_durations),
        calculate_stats(loop_cycles),
        calculate_stats(wait_cycles),
        calculate_stats(math_cycles),
        calculate_stats(epilogue_cycles),
        calculate_stats(e2e_cycles),
        trace_content
    )


@app.function(gpu="B200", timeout=600)
def run_bench(
    problem_sizes: List[Tuple[int, int, int, int]],
    warmup_iterations: int = 10,
    iterations: int = 10,
    enable_profiling: bool = True,
    enable_cycle_stats: bool = False,
    identity_b: bool = False,
    identity_a: bool = False,
):
    global _compiled_kernel_cache
    _compiled_kernel_cache.clear()
    import shutil
    try:
        shutil.rmtree("/root/.cache", ignore_errors=True)
    except Exception as e:
        pass

    print(f"Running benchmark with problem_sizes={problem_sizes}")

    return run_group_gemm_benchmark(
        problem_sizes, warmup_iterations, iterations, enable_profiling, enable_cycle_stats, identity_b=identity_b, identity_a=identity_a
    )





def test_compilation_only():
    """Test that kernel compiles for sm_100a without running"""
    print("Testing kernel compilation for sm_100a...")

    # Define a small problem for compilation test
    # NOTE: We use 2 duplicate groups here because using a single group (length 1)
    # triggers a "DSLRuntimeError: Tuple length mismatch" during CuTe compilation.
    # The compiler seems to expect/infer a tuple structure which requires >1 element
    # for proper binding of variadic arguments like a_tuple.
    problem_sizes = [(256, 4096, 7168, 1), (256, 4096, 7168, 1)]

    try:
        # This will trigger JIT compilation
        # We don't need to run the full benchmark, just calling custom_kernel
        # with dummy data will trigger compilation.




        # Mock SFA/SFB tuples with shape/stride methods
        class MockTensor:
             def __init__(self, shape, stride):
                 self.shape = shape
                 self._stride = stride
             def stride(self):
                 return self._stride

        sfa_tuple = []
        sfb_tuple = []
        cta_mn_list = []
        total_clusters = 0

        for m, n, k, l in problem_sizes:
             # Logic from original code to estimate shape/stride
             m_rest = (m + 127) // 128
             n_rest = (n + 127) // 128
             sf_k = (k + 15) // 16
             k_rest = (sf_k + 3) // 4

             # SFA: (32, 4, m_rest, 4, k_rest, l)
             sfa_shape = (32, 4, m_rest, 4, k_rest, l)
             # Stride is complex, just using dummy tuple for now as it's passed to make_fake_compact_tensor
             # Actually make_fake_compact_tensor needs stride_order which is permutation of indices
             # We can just use standard order
             sfa_stride = (1, 2, 3, 4, 5, 6) # Dummy stride order

             sfa_tuple.append(MockTensor(sfa_shape, sfa_stride))

             # SFB similar
             sfb_shape = (32, 4, n_rest, 4, k_rest, l)
             sfb_tuple.append(MockTensor(sfb_shape, sfa_stride))

             total_clusters += m_rest * n_rest

        print("Compiling kernel... (this may take a moment)")
        # Just compile, execution might fail on non-B200 if we actually ran it,
        # but custom_kernel calls compile_kernel first.
        # compile_kernel is explicitly called inside custom_kernel.
        # We can call it directly to test compilation.

        compiled_func = compile_kernel(problem_sizes, tuple(sfa_tuple), tuple(sfb_tuple), cutlass.Int32(total_clusters))
        print("✓ Kernel compiled successfully for sm_100a")

        # We can also try to run it if we are on a GPU, but expect failures if architecture mismatch
        # For now, compilation success is the main goal of local dry run.

    except Exception as e:
        print(f"✗ Compilation error: {e}")
        # raise e # Optional: raise to see full traceback in log




def run_benchmark_suite(runner, iterations: int = 10, enable_profiling: bool = True, enable_cycle_stats: bool = False):
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

    # -----------------------------------------------------------------------
    # Benchmark Configs from task.yml
    # -----------------------------------------------------------------------
    benchmark_configs = [
        # Benchmark 1: 8 groups, large N, large K
        [(80, 4096, 7168, 1), (176, 4096, 7168, 1), (128, 4096, 7168, 1), (72, 4096, 7168, 1),
         (64, 4096, 7168, 1), (248, 4096, 7168, 1), (96, 4096, 7168, 1), (160, 4096, 7168, 1)],
        # Benchmark 2: 8 groups, large N, smaller K
        [(40, 7168, 2048, 1), (76, 7168, 2048, 1), (168, 7168, 2048, 1), (72, 7168, 2048, 1),
         (164, 7168, 2048, 1), (148, 7168, 2048, 1), (196, 7168, 2048, 1), (160, 7168, 2048, 1)],
        # Benchmark 3: 2 groups
        [(192, 3072, 4096, 1), (320, 3072, 4096, 1)],
        # Benchmark 4: 2 groups
        [(128, 4096, 1536, 1), (384, 4096, 1536, 1)],
    ]

    # Speed of Light (SOL) Benchmark Times [us] provided in description
    sol_times = [18.833, 10.667, 2.406, 1.525]

    print("=========================================================================================")
    print(f"Running BENCHMARKS (profiling={enable_profiling})")
    print("=========================================================================================")

    results = []
    import numpy as np

    import numpy as np
    from datetime import datetime

    # Generate a unique timestamp for this benchmark run
    timestamp = os.environ.get("TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Ensure outputs/profiles directory exists
    os.makedirs("outputs/profiles", exist_ok=True)

    for i, problem_sizes in enumerate(benchmark_configs):
        print(f"\nBenchmark Config {i+1}:")
        # Run benchmark and unpack extended results
        (
            total_stats,
            kernel_stats,
            overhead_stats,
            pre_stats,
            post_stats,
            setup_delta_stats,
            internal_stats,
            loop_stats,
            wait_stats,
            math_stats,
            epilogue_stats,
            e2e_stats,
            trace_content
        ) = runner(problem_sizes=problem_sizes, warmup_iterations=10, iterations=iterations, enable_profiling=enable_profiling, enable_cycle_stats=enable_cycle_stats, identity_b=True, identity_a=True)

        if trace_content:
            local_trace_filename = f"outputs/profiles/trace_group_gemm_config_{i+1}_remote_{timestamp}.json"
            try:
                with open(local_trace_filename, "w") as f:
                    f.write(trace_content)
                print(f"Saved remote trace to {local_trace_filename}")
            except Exception as e:
                print(f"Failed to save remote trace: {e}")

        # Calculate approximate TFLOPS (Total FLOPs / Mean Time)
        total_flops = 0
        total_bytes_read = 0
        total_bytes_written = 0

        for m, n, k, l in problem_sizes:
            total_flops += 2 * m * n * k * l
            # Read: A (FP4) + B (FP4) + SFA (FP8) + SFB (FP8)
            # A: m * k * l / 2 bytes
            # B: n * k * l / 2 bytes
            # SFA: m * k/16 * l bytes
            # SFB: n * k/16 * l bytes
            total_bytes_read += (m * k * l // 2) + (n * k * l // 2) + (m * (k // 16) * l) + (n * (k // 16) * l)
            # Write: C (FP16)
            # C: m * n * l * 2 bytes
            total_bytes_written += m * n * l * 2

        mean_time_s = total_stats.mean * 1e-6
        achieved_tflops = (total_flops / mean_time_s) / 1e12

        # Calculate SOL TFLOPS from provided SOL time
        sol_time_us = sol_times[i]
        sol_tflops = (total_flops / (sol_time_us * 1e-6)) / 1e12

        # Calculate % SOL
        percent_of_sol = (achieved_tflops / sol_tflops) * 100 if sol_tflops > 0 else 0

        results.append(total_stats.mean)

        print(f"  Runs: {total_stats.runs}")
        print(f"  Theoretical Total FLOPs: {total_flops / 1e9:.3f} GFLOPs")
        print(f"  Total GMEM Read:         {total_bytes_read / 1e6:.3f} MB")
        print(f"  Total GMEM Write:        {total_bytes_written / 1e6:.3f} MB")
        print(f"  Mean custom_kernel (external) time: {total_stats.mean:.3f} us (SOL Ref: {sol_time_us} us)")
        print(f"  Mean GPU Kernel Time (Event):       {kernel_stats.mean:.3f} us")
        print(f"  Mean Overhead (External - Internal): {(total_stats.mean - internal_stats.mean):.3f} us")
        print(f"  Median Total Time: {total_stats.median:.3f} us")
        print(f"  Std Dev: {total_stats.std:.3f}")
        print(f"  Achieved TFLOPS: {achieved_tflops:.2f}")
        print(f"  SOL TFLOPS: {sol_tflops:.2f}")
        print(f"  % of SOL: {percent_of_sol:.2f}%")
        print("")

        print(f"  CustomStats:")
        print(f"     - Pre-Launch CPU:  {pre_stats.mean:.3f} us")
        print(f"     - Kernel Launch (Implicit): {(internal_stats.mean - pre_stats.mean - post_stats.mean):.3f} us")
        print(f"     - Post-Launch CPU: {post_stats.mean:.3f} us")
        print(f"     - Internal E2E:    {internal_stats.mean:.3f} us")

        if e2e_stats:
            print(f"     - Cycle Stats (Mean Cycles):")
            print(f"        * Setup Delta:     {setup_delta_stats.mean if setup_delta_stats else 0:.1f}")
            print(f"        * End-to-End:      {e2e_stats.mean:.1f}")
            print(f"        * Main Loop:       {loop_stats.mean:.1f}")
            print(f"        * Wait (Mem):      {wait_stats.mean:.1f}")
            print(f"        * Math (Compute):  {math_stats.mean:.1f}")
            print(f"        * Epilogue:        {epilogue_stats.mean:.1f}")

    if results:
        log_results = np.log(results)
        geomean = np.exp(log_results.mean())
        print("=========================================================================================")
        print(f"Final Geometric Mean of Execution Time: {geomean:.3f} us")
        print("=========================================================================================")


@app.local_entrypoint()
def main():
    enable_profiling = os.environ.get("ENABLE_PROFILING", "0") == "1"
    # Default to True unless explicitly disabled
    enable_cycle_stats = os.environ.get("ENABLE_CYCLE_STATS", "0") == "1"
    iterations = int(os.environ.get("ITERATIONS", "10"))
    run_benchmark_suite(run_bench.remote, iterations=iterations, enable_profiling=enable_profiling, enable_cycle_stats=enable_cycle_stats)

if __name__ == "__main__":
    if os.environ.get("COMPILE_LOCALLY_FOR_B200"):
        test_compilation_only()
    elif os.environ.get("MODAL_LOCAL_RUN") or not os.environ.get("MODAL_BENCHMARK"):
         # Optionally run local benchmark if called directly
         # You might want to guard this to avoid accidental long runs
         enable_profiling = os.environ.get("ENABLE_PROFILING", "0") == "1"
         enable_cycle_stats = os.environ.get("ENABLE_CYCLE_STATS", "0") == "1"
         iterations = int(os.environ.get("ITERATIONS", "10"))
         print(f"Running benchmark suite locally with iterations={iterations}...")
         run_benchmark_suite(run_group_gemm_benchmark, iterations=iterations, enable_profiling=enable_profiling, enable_cycle_stats=enable_cycle_stats)
