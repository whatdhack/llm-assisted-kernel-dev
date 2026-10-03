"""Pre-split the L panel into tf32 hi/lo ONCE per panel step, instead of
re-deriving the split inside every SYRK block.

MOTIVATION (measured 2026-08-17, see leaderboard_benchmark_results.txt):
each element of L is converted to tf32 4n/(3*BLOCK_M) ~= 173 times at
n=8192. A tile is loaded T+1 times per SYRK launch (once as l_m for each
valid off_n, once as l_n for each valid off_m) and each load re-runs the
same pure function on the same bits. 172 of the 173 are redundant. The
rounding is 31.3% of the SYRK instruction stream, plus ~10% for the
smem->register->smem staging round-trip that also disappears.

THIS VARIANT: a separate _split_tf32_kernel writes L_hi and L_lo for the
current panel strip, and the SYRK TMAs pre-split operands straight to
shared memory. The panel kernel is untouched, so tf32 knowledge stays on
the SYRK side.

  - conversions per element: ~173 -> 1
  - the register round-trip (smem load, round, smem store, fence) is gone;
    operands go global -> smem by TMA and are never in registers
  - shared memory unchanged: 4 L tiles + a, still ~49KB, still 4 blocks/SM
  - output must be BITWISE identical: same rounding, same inputs, so the
    MMAs see the same operand bits. The harness checks this.

THE COST THIS RUN IS MEANT TO DECIDE: one extra launch per panel step, 255
of them at n=8192. Launch overhead ~2-3us each is ~0.5-0.8ms against a
SYRK budget of 8.19ms, so the launch count may eat most of the saving. The
L strip itself is only ~1MB and L2-resident, so the added traffic (269MB
over the whole factorization, vs 34.9GB of SYRK reads) is not the risk.

Usage: modal run verify_presplit_b200.py
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

app = modal.App("cholesky-presplit-b200", image=image)

NEW_SYRK = '''@gluon.jit
def _split_tf32_kernel(L_ptr, Lhi_ptr, Llo_ptr, bk, start,
                       stride_b, stride_r, stride_c, n,
                       NB: gl.constexpr, BLOCK_R: gl.constexpr,
                       num_warps: gl.constexpr):
    """Split L[start:n, bk:bk+NB] into tf32 hi/lo once, for the SYRK to read.

    Covers exactly the rows the SYRK reads (off_m, off_n >= start); the
    diagonal block is never a SYRK operand. Reads beyond n are clamped to
    zero by TMA on the consumer side, so no initialization is needed.
    """
    LAYOUT: gl.constexpr = gl.BlockedLayout([1, 16], [16, 2], [num_warps, 1], [1, 0])
    ROW: gl.constexpr = gl.SliceLayout(dim=1, parent=LAYOUT)
    COL: gl.constexpr = gl.SliceLayout(dim=0, parent=LAYOUT)

    b = gl.program_id(0)
    pid = gl.program_id(1)

    r = start + pid * BLOCK_R + gl.arange(0, BLOCK_R, layout=ROW)
    c = bk + gl.arange(0, NB, layout=COL)
    mask = r[:, None] < n
    off = b * stride_b + r[:, None] * stride_r + c[None, :] * stride_c

    x = gl.load(L_ptr + off, mask=mask, other=0.0)
    hi = _round_tf32(x)
    lo = _round_tf32(x - hi)
    gl.store(Lhi_ptr + off, hi, mask=mask)
    gl.store(Llo_ptr + off, lo, mask=mask)


@gluon.jit
def _syrk_tile_tcgen05(l_hi_desc, l_lo_desc, a_desc, bk, off_m, off_n,
                       num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = a_desc.block_type.shape[1]

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    # Four L tiles arrive pre-split by TMA. Same 4 x 8KB the rounding path
    # used (hi reused the source tile), so shared memory is unchanged.
    l_m_hi_smem = gl.allocate_shared_memory(l_hi_desc.dtype, l_hi_desc.block_type.shape, l_hi_desc.layout)
    l_n_hi_smem = gl.allocate_shared_memory(l_hi_desc.dtype, l_hi_desc.block_type.shape, l_hi_desc.layout)
    l_m_lo_smem = gl.allocate_shared_memory(l_lo_desc.dtype, l_lo_desc.block_type.shape, l_lo_desc.layout)
    l_n_lo_smem = gl.allocate_shared_memory(l_lo_desc.dtype, l_lo_desc.block_type.shape, l_lo_desc.layout)
    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)

    # Split arrival barriers kept: the L tiles gate the MMAs, a_smem does not.
    # Now 4 tiles instead of 2, so the L wait covers 2x the bytes -- but no
    # rounding follows it, so the critical path after the wait is just MMA.
    mbarrier.expect(bar, 4 * l_hi_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(l_hi_desc, [off_m, bk], bar, l_m_hi_smem)
    tma.async_copy_global_to_shared(l_hi_desc, [off_n, bk], bar, l_n_hi_smem)
    tma.async_copy_global_to_shared(l_lo_desc, [off_m, bk], bar, l_m_lo_smem)
    tma.async_copy_global_to_shared(l_lo_desc, [off_n, bk], bar, l_n_lo_smem)

    bar_a = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar_a, count=1)
    mbarrier.expect(bar_a, a_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [off_m, off_n], bar_a, a_smem)

    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # No _round_tf32, no smem->reg load, no reg->smem store, no
    # fence_async_shared: TMA + the mbarrier wait already order the data.
    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc0 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    acc1 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    acc2 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    mbarrier.init(bar, count=3)
    tcgen05_mma(l_m_hi_smem, l_n_hi_smem.permute((1, 0)), acc0, use_acc=False, mbarriers=[bar], mbarrier_preds=[True])
    tcgen05_mma(l_m_hi_smem, l_n_lo_smem.permute((1, 0)), acc1, use_acc=False, mbarriers=[bar], mbarrier_preds=[True])
    tcgen05_mma(l_m_lo_smem, l_n_hi_smem.permute((1, 0)), acc2, use_acc=False, mbarriers=[bar], mbarrier_preds=[True])
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, num_warps)
    update = acc0.load(acc_reg_layout) + acc1.load(acc_reg_layout) + acc2.load(acc_reg_layout)

    mbarrier.wait(bar_a, phase=0)
    mbarrier.invalidate(bar_a)

    result = a_smem.load(acc_reg_layout) - update

    a_smem.store(result)
    fence_async_shared()
    tma.async_copy_shared_to_global(a_desc, [off_m, off_n], a_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def _syrk_kernel_tcgen05(A_ptr, Lhi_ptr, Llo_ptr, bk, start,
                         stride_b, stride_r, stride_c, n,
                         NB: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                         NUM_N_TILES: gl.constexpr, num_warps: gl.constexpr):
    b = gl.program_id(0)
    pid_m = gl.program_id(1)
    pid_n_lo = gl.program_id(2)
    pid_n_hi = NUM_N_TILES - 1 - pid_n_lo

    l_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, NB], gl.float32)
    a_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float32)
    off_m = start + pid_m * BLOCK_M

    # Descriptors still built lazily inside the guard. Three now instead of
    # two, which is a known cost: making construction lazy was worth 5.7%.
    if pid_n_lo <= pid_m:
        l_hi_desc = tma.make_tensor_descriptor(
            Lhi_ptr + b * stride_b, [n, n], [stride_r, stride_c], [BLOCK_M, NB], l_layout,
        )
        l_lo_desc = tma.make_tensor_descriptor(
            Llo_ptr + b * stride_b, [n, n], [stride_r, stride_c], [BLOCK_M, NB], l_layout,
        )
        a_desc = tma.make_tensor_descriptor(
            A_ptr + b * stride_b, [n, n], [stride_r, stride_c], [BLOCK_M, BLOCK_N], a_layout,
        )
        off_n = start + pid_n_lo * BLOCK_N
        _syrk_tile_tcgen05(l_hi_desc, l_lo_desc, a_desc, bk, off_m, off_n, num_warps=num_warps)
        if pid_n_hi <= pid_m and pid_n_hi != pid_n_lo:
            off_n = start + pid_n_hi * BLOCK_N
            _syrk_tile_tcgen05(l_hi_desc, l_lo_desc, a_desc, bk, off_m, off_n, num_warps=num_warps)
'''

NEW_HOST = '''def custom_kernel(data: input_t) -> output_t:
    global _allocator_set
    if not _allocator_set:
        triton.set_allocator(_alloc_fn)
        _allocator_set = True

    A = data.to(torch.float32).contiguous().clone()
    batch, n, _ = A.shape
    L = torch.zeros_like(A)

    # hi/lo scratch. torch.empty is safe: the split kernel writes every row
    # in [start, n) of the current column strip, and the SYRK reads only
    # that strip; TMA clamps rows past n to zero.
    Lhi = torch.empty_like(A)
    Llo = torch.empty_like(A)

    stride_b, stride_r, stride_c = L.stride()
    NB = 32
    BLOCK_I = 32
    BLOCK_M = BLOCK_N = 64
    BLOCK_R = 64

    for bk in range(0, n, NB):
        grid_panel = (batch, triton.cdiv(n - bk, BLOCK_I))
        _panel_trsm_kernel[grid_panel](
            A, L, bk, stride_b, stride_r, stride_c, n,
            NB=NB, BLOCK_I=BLOCK_I,
            num_warps=1,
        )

        trailing = n - bk - NB
        if trailing > 0:
            start = bk + NB
            _split_tf32_kernel[(batch, triton.cdiv(n - start, BLOCK_R))](
                L, Lhi, Llo, bk, start, stride_b, stride_r, stride_c, n,
                NB=NB, BLOCK_R=BLOCK_R, num_warps=4,
            )

            num_m_tiles = triton.cdiv(trailing, BLOCK_M)
            num_n_tiles = triton.cdiv(trailing, BLOCK_N)
            grid_syrk = (batch, num_m_tiles, triton.cdiv(num_n_tiles, 2))
            _syrk_kernel_tcgen05[grid_syrk](
                A, Lhi, Llo, bk, start, stride_b, stride_r, stride_c, n,
                NB=NB, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                NUM_N_TILES=num_n_tiles, num_warps=4,
            )

    return L
'''


def _write_variant():
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    s = src.index("@gluon.jit\ndef _syrk_tile_tcgen05(")
    e = src.index("def _alloc_fn(")
    o = src[:s] + NEW_SYRK + "\n\n" + src[e:]
    hs = o.index("def custom_kernel(")
    he = o.index('if __name__ == "__main__":')
    o = o[:hs] + NEW_HOST + "\n\n" + o[he:]
    open("/root/python_standalone/_v_presplit.py", "w").write(o)


@app.function(gpu="B200", timeout=7200, max_containers=2)
def bench():
    import importlib, json, sys, traceback
    import torch
    sys.path.insert(0, "/root/python_standalone")
    _write_variant()

    import bench_leaderboard
    base = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    try:
        new = importlib.import_module("_v_presplit")
    except Exception:
        return {"fail": traceback.format_exc()[-4000:]}

    # correctness first: must be BITWISE identical, same rounding on same bits
    bit = []
    for batch, n in [(1, 256), (2, 512), (1, 1024), (3, 200), (1, 8192)]:
        try:
            A = bench_leaderboard.generate_input(batch, n, 2, 40000 + n)
            Lb = base.custom_kernel(A)
            Ln = new.custom_kernel(A)
            torch.cuda.synchronize()
            bit.append({
                "batch": batch, "n": n,
                "maxdiff": (Lb - Ln).abs().amax().item(),
                "err_base": (Lb @ Lb.transpose(-1, -2) - A).abs().amax().item(),
                "err_new": (Ln @ Ln.transpose(-1, -2) - A).abs().amax().item(),
                "nan_new": int((~torch.isfinite(Ln)).sum().item()),
            })
            del A, Lb, Ln
            torch.cuda.empty_cache()
        except Exception:
            return {"fail": f"correctness {batch}x{n}\n" + traceback.format_exc()[-3000:]}

    rows = []
    for spec in bench_leaderboard.BENCHMARKS:
        row = {"batch": spec["batch"], "n": spec["n"]}
        for tag, m in (("base", base), ("presplit", new)):
            try:
                row[f"{tag}_ms"] = bench_leaderboard.bench_one(m.custom_kernel, spec) * 1e3
            except Exception:
                return {"fail": f"{spec['batch']}x{spec['n']} {tag}\n"
                                + traceback.format_exc()[-3000:], "bit": bit}
        rows.append(row)
        torch.cuda.empty_cache()

    # per-kernel split at n=8192 so the launch-overhead cost is visible
    res = {}
    try:
        A = bench_leaderboard.generate_input(1, 8192, 2, 48192)
        for tag, m in (("base", base), ("presplit", new)):
            m.custom_kernel(A)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                m.custom_kernel(A)
                torch.cuda.synchronize()
            p = f"/tmp/t_{tag}.json"
            prof.export_chrome_trace(p)
            agg = {}
            for e in json.load(open(p))["traceEvents"]:
                if e.get("cat") != "kernel":
                    continue
                a = e.get("args", {})
                k = e["name"].split("(")[0][:40]
                d = agg.setdefault(k, {"n": 0, "ms": 0.0, "smem": a.get("shared memory"),
                                       "regs": a.get("registers per thread")})
                d["n"] += 1
                d["ms"] += float(e.get("dur", 0)) / 1e3
            res[tag] = agg
        del A
    except Exception:
        res = {"fail": traceback.format_exc()[-2000:]}

    return {"gpu": torch.cuda.get_device_name(0), "bit": bit, "rows": rows, "res": res}


@app.local_entrypoint()
def main():
    import json, math
    r = bench.remote()
    if "fail" in r:
        print("FAILED:\n", r["fail"])
        if "bit" in r:
            print("bitwise results so far:", json.dumps(r["bit"], indent=1))
        return
    print("RAW", json.dumps({"bit": r["bit"], "rows": r["rows"]}))

    print("=" * 78)
    print(r["gpu"], "  pre-split L into tf32 hi/lo, one pass per panel step")
    print("=" * 78)
    print("BITWISE CHECK (must be 0.00e+00: same rounding, same input bits)")
    for x in r["bit"]:
        flag = "OK " if x["maxdiff"] == 0.0 and x["nan_new"] == 0 else "DIFF"
        print(f"  {flag} batch={x['batch']:3} n={x['n']:6}  max|base-new|={x['maxdiff']:.3e}"
              f"  err {x['err_base']:.2e} -> {x['err_new']:.2e}  nan={x['nan_new']}")

    print()
    print(f"{'batch':>6} {'n':>6} | {'base':>9} {'presplit':>9} | {'speedup':>8}")
    bs, ps = [], []
    for x in r["rows"]:
        bs.append(x["base_ms"]); ps.append(x["presplit_ms"])
        print(f"{x['batch']:6} {x['n']:6} | {x['base_ms']:9.3f} {x['presplit_ms']:9.3f} |"
              f" {x['base_ms'] / x['presplit_ms']:7.3f}x")
    g = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 46)
    print(f"{'GEOMEAN':>13} | {g(bs):9.3f} {g(ps):9.3f} | {g(bs) / g(ps):7.3f}x")
    print(f"  wins {sum(1 for x in r['rows'] if x['presplit_ms'] < x['base_ms'])}/15")

    res = r["res"]
    if "fail" in res:
        print("\nper-kernel capture failed:\n", res["fail"])
        return
    print()
    print("PER-KERNEL at batch=1 n=8192 (launch count is the risk here)")
    for tag in ("base", "presplit"):
        print(f"  {tag}:")
        for k, d in sorted(res[tag].items(), key=lambda kv: -kv[1]["ms"]):
            print(f"    {k:<34} {d['n']:5} launches {d['ms']:8.3f} ms"
                  f"  smem {d['smem']}  regs {d['regs']}")
