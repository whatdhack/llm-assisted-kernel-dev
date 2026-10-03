"""
Port the kloop two-phase / inner-K-loop structure onto the CURRENT kernel.

cholesky_gluon_tcgen05_kloop.py already has the K-loop, but paired with the
OLD panel (masked-reduce row extraction, Lp in registers, nb_valid passed
in). The current kernel has the improved panel but a rank-NB SYRK. This
builds the combination: improved panel + K-loop SYRK, and keeps this
session's SYRK gains too --

    * 3 independent accumulators, one wait per K-chunk (not 3)
    * A issued up front and waited on only in the epilogue, so it streams in
      behind the WHOLE K-loop (worth more here: the shadow is much longer)
    * TMA descriptors built inside the triangular guard, so blocks with no
      tile skip construction and its constant-cache invalidation

Structure per super-panel of NB_OUTER columns:
    for each NB_INNER sub-step:
        panel factor + TRSM over the full remaining height
        rank-NB_INNER SYRK, columns CLIPPED to the strip   (narrow, cheap)
    one rank-NB_OUTER SYRK over the whole trailing submatrix  (the win)

Expected shape from the recorded kloop numbers: loses at small/medium n
(NB_INNER must rise to 64 to align with BLOCK_N, and panel cost grows with
NB), wins at n=32768. The question is whether the improved panel and SYRK
move the crossover.

Usage: modal run verify_kloop_port_b200.py
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

app = modal.App("cholesky-kloop-port-b200", image=image)

SHAPES = [(1, 32768), (1, 16384), (1, 8192), (1, 4096), (2, 4096),
          (8, 2048), (640, 512), (4096, 32)]

NEW_SYRK = '''@gluon.jit
def _syrk_tile_tcgen05(l_desc, a_desc, k_start, off_m, off_n,
                       NUM_K_CHUNKS: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = a_desc.block_type.shape[1]
    KB: gl.constexpr = l_desc.block_type.shape[1]

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    bar_a = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar_a, count=1)

    # A is loaded ONCE and written back ONCE no matter how many K-chunks
    # accumulate into it -- that amortization is the whole point. It is also
    # issued here but NOT waited on until the epilogue, so it streams in behind
    # the entire K-loop.
    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    mbarrier.expect(bar_a, a_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m, off_n], bar_a, a_smem)

    # every L buffer is KB-wide, never rank-wide: shared memory is f(KB), a
    # constant, so the update's rank is free to grow.
    l_m_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_n_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_m_hi_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_m_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_n_hi_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)
    l_n_lo_smem = gl.allocate_shared_memory(l_desc.dtype, l_desc.block_type.shape, l_desc.layout)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc0 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    acc1 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    acc2 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    # sizePerThread[1] must be KB//2 so dim-1 extent lands exactly on KB
    REG_LAYOUT: gl.constexpr = gl.BlockedLayout([1, KB // 2], [16, 2], [num_warps, 1], [1, 0])

    use_acc = False
    for kc in range(NUM_K_CHUNKS):
        k_off = k_start + kc * KB

        mbarrier.init(bar, count=1)
        mbarrier.expect(bar, 2 * l_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(l_desc, [off_m, k_off], bar, l_m_smem)
        tma.async_copy_global_to_shared(l_desc, [off_n, k_off], bar, l_n_smem)
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)

        l_m_reg = l_m_smem.load(REG_LAYOUT)
        l_m_hi = _round_tf32(l_m_reg)
        l_m_lo = _round_tf32(l_m_reg - l_m_hi)
        l_n_reg = l_n_smem.load(REG_LAYOUT)
        l_n_hi = _round_tf32(l_n_reg)
        l_n_lo = _round_tf32(l_n_reg - l_n_hi)
        l_m_hi_smem.store(l_m_hi)
        l_m_lo_smem.store(l_m_lo)
        l_n_hi_smem.store(l_n_hi)
        l_n_lo_smem.store(l_n_lo)
        fence_async_shared()

        # three independent products, three MMAs back to back, ONE wait.
        # use_acc is False only on the first chunk; later chunks accumulate.
        mbarrier.init(bar, count=3)
        tcgen05_mma(l_m_hi_smem, l_n_hi_smem.permute((1, 0)), acc0, use_acc=use_acc, mbarriers=[bar], mbarrier_preds=[True])
        tcgen05_mma(l_m_hi_smem, l_n_lo_smem.permute((1, 0)), acc1, use_acc=use_acc, mbarriers=[bar], mbarrier_preds=[True])
        tcgen05_mma(l_m_lo_smem, l_n_hi_smem.permute((1, 0)), acc2, use_acc=use_acc, mbarriers=[bar], mbarrier_preds=[True])
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)

        use_acc = True

    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, num_warps)
    update = acc0.load(acc_reg_layout) + acc1.load(acc_reg_layout) + acc2.load(acc_reg_layout)

    mbarrier.wait(bar_a, phase=0)
    mbarrier.invalidate(bar_a)
    a_reg = a_smem.load(acc_reg_layout)
    result = a_reg - update

    out_smem = gl.allocate_shared_memory(gl.float32, a_desc.block_type.shape, a_desc.layout)
    out_smem.store(result)
    fence_async_shared()
    tma.async_copy_shared_to_global(a_desc, [off_m, off_n], out_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def _syrk_kernel_tcgen05(A_ptr, L_ptr, k_start, start_m, start_n,
                         stride_b, stride_r, stride_c, n,
                         KB: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                         NUM_K_CHUNKS: gl.constexpr, num_warps: gl.constexpr):
    b = gl.program_id(0)
    pid_m = gl.program_id(1)
    pid_n = gl.program_id(2)

    l_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, KB], gl.float32)
    a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float32)

    off_m = start_m + pid_m * BLOCK_M
    off_n = start_n + pid_n * BLOCK_N

    # Lower-triangular skip in GLOBAL coordinates, so one kernel serves both
    # the square trailing update (start_m == start_n) and the rectangular
    # strip update. The pid_n_lo/pid_n_hi grid-halving trick is dropped: it
    # assumed a square triangular region, which the strip is not.
    # Descriptors are built INSIDE the guard so blocks with no tile skip
    # on-device construction and its constant-cache invalidation.
    if off_n <= off_m:
        l_desc = tma.make_tensor_descriptor(
            L_ptr + b * stride_b, [n, n], [stride_r, stride_c], [BLOCK_M, KB], l_layout,
        )
        a_desc = tma.make_tensor_descriptor(
            A_ptr + b * stride_b, [n, n], [stride_r, stride_c], [BLOCK_M, BLOCK_N], a_layout,
        )
        _syrk_tile_tcgen05(l_desc, a_desc, k_start, off_m, off_n,
                           NUM_K_CHUNKS=NUM_K_CHUNKS, num_warps=num_warps)
'''

NEW_HOST = '''def custom_kernel(data: input_t) -> output_t:
    global _allocator_set
    if not _allocator_set:
        triton.set_allocator(_alloc_fn)
        _allocator_set = True

    A = data.to(torch.float32).contiguous().clone()
    batch, n, _ = A.shape
    L = torch.zeros_like(A)

    stride_b, stride_r, stride_c = L.stride()
    BLOCK_I = 32
    BLOCK_M = BLOCK_N = 64
    KB = 32
    NB_INNER = _NB_INNER
    NB_OUTER = _NB_OUTER
    assert NB_INNER % BLOCK_N == 0, "strip tiles must align to the super-panel edge"
    assert NB_INNER % KB == 0 and NB_OUTER % KB == 0

    for bk_o in range(0, n, NB_OUTER):
        end_o = min(bk_o + NB_OUTER, n)

        # Phase 1: factor the super-panel strip [bk_o:n, bk_o:end_o).
        # Right-looking, but every SYRK's columns are CLIPPED to the strip, so
        # the wide trailing submatrix is left untouched until phase 2.
        for bk in range(bk_o, end_o, NB_INNER):
            nb_valid = min(NB_INNER, n - bk)
            grid_panel = (batch, triton.cdiv(n - bk, BLOCK_I))
            _panel_trsm_kernel[grid_panel](
                A, L, bk, stride_b, stride_r, stride_c, n,
                NB=NB_INNER, BLOCK_I=BLOCK_I, num_warps=1,
            )
            s = bk + nb_valid
            if s < end_o and nb_valid == NB_INNER:
                _syrk_kernel_tcgen05[(batch, triton.cdiv(n - s, BLOCK_M),
                                      triton.cdiv(end_o - s, BLOCK_N))](
                    A, L, bk, s, s, stride_b, stride_r, stride_c, n,
                    KB=KB, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    NUM_K_CHUNKS=nb_valid // KB, num_warps=4,
                )

        # Phase 2: ONE rank-(end_o - bk_o) update of the whole trailing block.
        if end_o < n:
            k_len = end_o - bk_o
            _syrk_kernel_tcgen05[(batch, triton.cdiv(n - end_o, BLOCK_M),
                                  triton.cdiv(n - end_o, BLOCK_N))](
                A, L, bk_o, end_o, end_o, stride_b, stride_r, stride_c, n,
                KB=KB, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                NUM_K_CHUNKS=k_len // KB, num_warps=4,
            )

    return L
'''


def _write_variant(nb_inner, nb_outer, name):
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    s = src.index("@gluon.jit\ndef _syrk_tile_tcgen05(")
    e = src.index("def _alloc_fn(")
    o = src[:s] + NEW_SYRK + "\n\n" + src[e:]
    hs = o.index("def custom_kernel(")
    he = o.index('if __name__ == "__main__":')
    o = o[:hs] + NEW_HOST.replace("_NB_INNER", str(nb_inner)).replace("_NB_OUTER", str(nb_outer)) + "\n\n" + o[he:]
    open(f"/root/python_standalone/{name}.py", "w").write(o)


@app.function(gpu="B200", timeout=7200)
def bench(nb_inner: int, nb_outer: int):
    import importlib, json, statistics, sys, time, traceback
    import torch
    sys.path.insert(0, "/root/python_standalone")
    name = f"_v_kloop_{nb_inner}_{nb_outer}"
    _write_variant(nb_inner, nb_outer, name)
    base = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    try:
        new = importlib.import_module(name)
    except Exception:
        return {"fail": traceback.format_exc()[-3000:]}

    import bench_leaderboard
    starter = importlib.import_module("starter")
    rows = []
    # OFFICIAL protocol: all 15 specs, bench_leaderboard.bench_one
    # (warmup=2, median of 5, fresh input per case)
    for spec in bench_leaderboard.BENCHMARKS:
        batch, n = spec["batch"], spec["n"]
        row = {"batch": batch, "n": n}
        for tag, m in (("base", base), ("kloop", new), ("starter", starter)):
            try:
                row[f"{tag}_ms"] = bench_leaderboard.bench_one(m.custom_kernel, spec) * 1e3
            except Exception:
                return {"fail": f"{batch}x{n} {tag}\n" + traceback.format_exc()[-2500:]}
        A = bench_leaderboard.generate_input(batch, n, spec["cond"], spec["seed"])
        for tag, m in (("base", base), ("kloop", new)):
            L = m.custom_kernel(A)
            torch.cuda.synchronize()
            row[f"{tag}_err"] = (L @ L.transpose(-1, -2) - A).abs().amax().item()
            row[f"{tag}_nan"] = int((~torch.isfinite(L)).sum().item())
            del L
        rows.append(row)
        del A
        torch.cuda.empty_cache()
    return {"gpu": torch.cuda.get_device_name(0), "rows": rows}


@app.local_entrypoint()
def main(nb_inner: int = 64, nb_outer: int = 1024):
    r = bench.remote(nb_inner, nb_outer)
    if "fail" in r:
        print("VARIANT FAILED:\n", r["fail"]); return
    print("=" * 100)
    print(f"{r['gpu']}   NB_INNER={nb_inner} NB_OUTER={nb_outer} KB=32")
    print("=" * 100)
    import math
    print(f"{'batch':>6} {'n':>6} | {'starter':>9} {'current':>9} {'kloop':>9} {'DISPATCH':>9} |"
          f" {'kl/cur':>7} | {'err':>10}")
    bs, ks, ds, ss = [], [], [], []
    for x in r["rows"]:
        # dispatch = whichever of the two is better for that shape
        d = min(x["base_ms"], x["kloop_ms"])
        bs.append(x["base_ms"]); ks.append(x["kloop_ms"]); ds.append(d); ss.append(x["starter_ms"])
        pick = "K" if x["kloop_ms"] < x["base_ms"] else "c"
        print(f"{x['batch']:6} {x['n']:6} | {x['starter_ms']:9.3f} {x['base_ms']:9.3f}"
              f" {x['kloop_ms']:9.3f} {d:8.3f}{pick} | {x['base_ms']/x['kloop_ms']:6.3f}x |"
              f" {x['kloop_err']:10.2e}")
    g = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 92)
    print(f"{'GEOMEAN':>13} | {g(ss):9.3f} {g(bs):9.3f} {g(ks):9.3f} {g(ds):9.3f} |")
    print()
    print(f"  vs cuSOLVER:  current {g(bs)/g(ss):.3f}x   kloop {g(ks)/g(ss):.3f}x"
          f"   DISPATCH {g(ds)/g(ss):.3f}x")
    print("  NaNs:", sum(x["kloop_nan"] for x in r["rows"]))
