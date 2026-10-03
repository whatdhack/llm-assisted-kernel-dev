"""
Isolates the tf32x3 mechanics from the Cholesky: one [64,32] x [64,32]^T
tcgen05 product, run with 1 and with 3 accumulators, compared against float64
references built from the operands rounded to tf32 three different ways.

Answers two questions:
  * does tcgen05 narrow fp32 -> tf32 by truncation or round-to-nearest?
  * do the two tf32x3 correction terms actually reach the accumulator?

Usage: modal run probe_tf32_b200.py
"""
import os as _os

import modal

LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch", "numpy")
    # PINNED. Modal caches image layers by the definition hash, so an
    # unpinned "nvidia-cutlass-dsl" would keep whatever version was
    # latest when the layer was FIRST built, silently, forever -- and
    # two of this port's bugs were version-sensitive DSL semantics
    # (internal_type=TFloat32 rounding inside the TMA; cute.copy and
    # cute.gemm electing per-warp). Pin it so the number is a fact in
    # the source and the recorded measurements stay reproducible.
    .pip_install("nvidia-cutlass-dsl==4.7.1")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib",
        "CUTE_DSL_ARCH": "sm_100a",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(LOCAL_DIR, "/root/python_standalone")
)

app = modal.App("cholesky-probe-tf32-b200", image=image)

def _print_versions(tag=""):
    """Record what actually ran. The image pins nvidia-cutlass-dsl, but pinning
    is only half of it -- printing the version is what makes a saved artifact
    self-describing months later."""
    import subprocess, sys
    import torch
    import cutlass
    v = [f"torch {torch.__version__}", f"cutlass-dsl {cutlass.__version__}"]
    try:
        import triton
        v.append(f"triton {triton.__version__}")
    except ImportError:
        pass
    v.append(f"gpu {torch.cuda.get_device_name(0)}")
    print(("VERSIONS " + tag + ": " if tag else "VERSIONS: ") + ", ".join(v),
          flush=True)




@app.function(gpu="B200", timeout=900)
def probe():
    import sys

    import torch
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    import cuda.bindings.driver as cuda
    from cutlass.cute import nvgpu
    from cutlass.cute.nvgpu import cpasync, tcgen05
    from cutlass.cute.runtime import from_dlpack

    sys.path.insert(0, "/root/python_standalone")
    import cholesky_cute_tcgen05_blocked as M

    _print_versions()
    BM, BN, NB = M.BLOCK_M, M.BLOCK_N, M.NB
    THREADS = M.SYRK_THREADS

    @cute.jit
    def split(src_flat, dst_flat, tidx, lo_mode: cutlass.Constexpr):
        n: cutlass.Constexpr = M.SPLIT_PER_THREAD
        rU = cute.make_rmem_tensor(n, cutlass.Uint32)
        rX = cute.recast_tensor(rU, cutlass.Float32)
        rHU = cute.make_rmem_tensor(n, cutlass.Uint32)
        rH = cute.recast_tensor(rHU, cutlass.Float32)
        rLo = cute.make_rmem_tensor(n, cutlass.Float32)
        cute.autovec_copy(src_flat[(None, tidx)], rU)
        for i in cutlass.range_constexpr(n):
            if cutlass.const_expr(lo_mode == "rn"):
                rHU[i] = (rU[i] + cutlass.Uint32(0x1000)) & cutlass.Uint32(0xFFFFE000)
            else:
                rHU[i] = rU[i] & cutlass.Uint32(0xFFFFE000)
        for i in cutlass.range_constexpr(n):
            rLo[i] = rX[i] - rH[i]
        cute.autovec_copy(rLo, dst_flat[(None, tidx)])

    @cute.kernel
    def k(tma_a, mA, tma_b, mB, mC, sA_lay, sB_lay, sC_lay, mma,
          ncols: cutlass.Constexpr, nterms: cutlass.Constexpr,
          lo_mode: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        smem = utils.SmemAllocator()
        pa = smem.allocate_array(cutlass.Float32, BM * NB, byte_alignment=1024)
        pb = smem.allocate_array(cutlass.Float32, BN * NB, byte_alignment=1024)
        pal = smem.allocate_array(cutlass.Float32, BM * NB, byte_alignment=1024)
        pbl = smem.allocate_array(cutlass.Float32, BN * NB, byte_alignment=1024)
        pc = smem.allocate_array(cutlass.Float32, BM * BN, byte_alignment=1024)
        mbar = smem.allocate_array(cutlass.Int64, 1, byte_alignment=8)
        hold = smem.allocate_array(cutlass.Int32, 1, byte_alignment=8)

        def view(p, lay):
            return cute.make_tensor(
                cute.recast_ptr(p, lay.inner, dtype=cutlass.Float32), lay.outer)

        sA, sB = view(pa, sA_lay), view(pb, sB_lay)
        sAl, sBl = view(pal, sA_lay), view(pbl, sB_lay)
        sC = view(pc, sC_lay)
        flat = cute.make_layout((M.SPLIT_PER_THREAD, THREADS),
                                stride=(1, M.SPLIT_PER_THREAD))
        fA = cute.make_tensor(cute.recast_ptr(pa, dtype=cutlass.Uint32), flat)
        fB = cute.make_tensor(cute.recast_ptr(pb, dtype=cutlass.Uint32), flat)
        fAl = cute.make_tensor(pal, flat)
        fBl = cute.make_tensor(pbl, flat)

        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar, 1)
        cute.arch.mbarrier_init_fence()

        acc_fake = mma.make_fragment_C(mma.partition_shape_C((BM, BN)))
        nb = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        tm = utils.TmemAllocator(hold, barrier_for_retrieve=nb)
        tm.allocate(ncols)
        tm.wait_for_alloc()
        p = tm.retrieve_ptr(cutlass.Float32)
        cols = tcgen05.find_tmem_tensor_col_offset(acc_fake)
        acc = [cute.make_tensor(p + i * cols, acc_fake.layout) for i in range(3)]

        gA = cute.local_tile(mA, (BM, NB), (0, 0))
        gB = cute.local_tile(mB, (BN, NB), (0, 0))
        gC = cute.local_tile(mC, (BM, BN), (0, 0))
        thr_mma = mma.get_slice(0)
        tSA, tGA = cpasync.tma_partition(
            tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA), 0, 3))
        tSB, tGB = cpasync.tma_partition(
            tma_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gB), 0, 3))

        nbytes = cute.size_in_bytes(cutlass.Float32, sA.layout)
        cute.arch.barrier()
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, 2 * nbytes)
            cute.copy(tma_a, tGA, tSA, tma_bar_ptr=mbar)
            cute.copy(tma_b, tGB, tSB, tma_bar_ptr=mbar)
        cute.arch.mbarrier_wait(mbar, 0)

        split(fA, fAl, tidx, lo_mode)
        split(fB, fBl, tidx, lo_mode)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()

        rA = mma.make_fragment_A(sA)
        rB = mma.make_fragment_B(sB)
        rAl = mma.make_fragment_A(sAl)
        rBl = mma.make_fragment_B(sBl)
        nk: cutlass.Constexpr = cute.size(rA, mode=[2])
        if warp == 0:
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(nk):
                c = (None, None, kb)
                cute.gemm(mma, acc[0], rA[c], rB[c], acc[0])
                if cutlass.const_expr(nterms == 3):
                    cute.gemm(mma, acc[1], rA[c], rBl[c], acc[1])
                    cute.gemm(mma, acc[2], rAl[c], rB[c], acc[2])
                mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(mbar)
        cute.arch.mbarrier_wait(mbar, 1)

        op = sm100_utils.get_tmem_load_op((BM, BN, NB), utils.LayoutEnum.ROW_MAJOR,
                                          cutlass.Float32, cutlass.Float32,
                                          (BM, BN), False)
        tc = tcgen05.make_tmem_copy(op, M._epi_view(acc[0])[(None, None, 0, 0)])
        thr = tc.get_slice(tidx)
        epi: cutlass.Constexpr = (None, None, None, 0, 0)
        tD = thr.partition_D(gC)
        r = [cute.make_rmem_tensor(tD.shape, cutlass.Float32) for _ in range(3)]
        for i in cutlass.range_constexpr(3):
            cute.copy(tc, thr.partition_S(M._epi_view(acc[i]))[epi], r[i])
        cute.arch.fence_view_async_tmem_load()
        out = r[0].load()
        if cutlass.const_expr(nterms == 3):
            out = out + r[1].load() + r[2].load()
        r[0].store(out)
        cute.autovec_copy(r[0], tD)
        cute.arch.barrier()
        tm.relinquish_alloc_permit()
        tm.free(p, ncols)

    @cute.jit
    def h(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
          stream: cuda.CUstream, nterms: cutlass.Constexpr,
          lo_mode: cutlass.Constexpr, internal_tf32: cutlass.Constexpr):
        major: cutlass.Constexpr = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.TFloat32, cutlass.TFloat32, major, major,
            cutlass.Float32, tcgen05.CtaGroup.ONE, (BM, BN))
        sA_lay = cute.slice_(sm100_utils.make_smem_layout_a(
            mma, (BM, BN, NB), cutlass.TFloat32, 1), (None, None, None, 0))
        sB_lay = cute.slice_(sm100_utils.make_smem_layout_b(
            mma, (BM, BN, NB), cutlass.TFloat32, 1), (None, None, None, 0))
        sC_lay = cute.slice_(sm100_utils.make_smem_layout_epi(
            cutlass.Float32, utils.LayoutEnum.ROW_MAJOR, (BM, BN), 1), (None, None, 0))
        itype: cutlass.Constexpr = cutlass.TFloat32 if internal_tf32 else None
        ta, tA = nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(), mA, sA_lay, (BM, BN, NB), mma,
            internal_type=itype)
        tb, tB = nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(), mB, sB_lay, (BM, BN, NB), mma,
            internal_type=itype)
        acc = mma.make_fragment_C(mma.partition_shape_C((BM, BN)))
        nc: cutlass.Constexpr = utils.get_num_tmem_alloc_cols([acc, acc, acc])
        k(ta, tA, tb, tB, mC, sA_lay, sB_lay, sC_lay, mma, nc, nterms,
          lo_mode).launch(
            grid=(1, 1, 1), block=(THREADS, 1, 1), smem=1024 + (4 * BM * NB + BM * BN) * 4,
            stream=stream)

    gen = torch.Generator(device="cuda").manual_seed(3)
    A = torch.randn((BM, NB), device="cuda", dtype=torch.float32, generator=gen)
    B = torch.randn((BN, NB), device="cuda", dtype=torch.float32, generator=gen)
    C = torch.zeros((BM, BN), device="cuda", dtype=torch.float32)

    def bits(t, mode):
        u = t.view(torch.int32)
        if mode == "trunc":
            v = u & -8192                                   # 0xFFFFE000
        else:
            v = (u + 0x1000) & -8192
        return v.view(torch.float32)

    ref64 = A.double() @ B.double().T
    ref_tr = bits(A, "trunc").double() @ bits(B, "trunc").double().T
    ref_rn = bits(A, "rn").double() @ bits(B, "rn").double().T

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    lines = [f"scale (max |C|) = {ref64.abs().amax().item():.4f}",
             f"{'internal_tf32':>13} {'lo':>6} {'terms':>6} | {'vs fp64':>11} "
             f"{'vs trunc-tf32':>13} {'vs rn-tf32':>11}"]
    for internal_tf32 in (True, False):
        for lo_mode in ("trunc", "rn"):
            for nterms in (1, 3):
                C.zero_()
                args = (from_dlpack(A, assumed_align=16),
                        from_dlpack(B, assumed_align=16),
                        from_dlpack(C, assumed_align=16))
                f = cute.compile(h, *args, cuda.CUstream(0), nterms, lo_mode,
                                 internal_tf32)
                f(*args, stream)
                torch.cuda.synchronize()
                c = C.double()
                lines.append(
                    f"{str(internal_tf32):>13} {lo_mode:>6} {nterms:>6} | "
                    f"{(c - ref64).abs().amax().item():11.3e} "
                    f"{(c - ref_tr).abs().amax().item():13.3e} "
                    f"{(c - ref_rn).abs().amax().item():11.3e}")
    text = "\n".join(lines)
    print("RESULT\n" + text, flush=True)
    return text


@app.local_entrypoint()
def main():
    print(probe.remote())
