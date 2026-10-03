"""
Attributes the ~44 us of _panel_trsm_kernel (cholesky_gluon_tcgen05_blocked.py)
to its two serial loops, on a real B200 via Modal.

Method: one gluon kernel whose two loops have constexpr trip counts, so the
compiler specializes each variant. Setting a trip count to 0 replaces that
loop with a cheap stand-in that keeps the other loop's values live (a tile
load feeding loop 2, a tile store consuming loop 1), so nothing is DCE'd:

    shell       L1_ITERS=0,  L2_ITERS=0   -> tile load + tile store only
    L1xk        L1_ITERS=k,  L2_ITERS=0   -> factorization loop, k iters
    L2xk        L1_ITERS=0,  L2_ITERS=k   -> TRSM-rows loop, k iters
    full        L1_ITERS=32, L2_ITERS=32  -> should reproduce the real kernel

Sweeping k gives per-iteration cost as the slope and fixed kernel overhead
as the intercept. Durations are read back out of a Kineto trace so they are
measured exactly the same way as the numbers in outputs/traces/ (pure kernel
duration, no launch gap).

Usage: modal run ablate_panel_b200.py
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

app = modal.App("cholesky-ablate-panel-b200", image=image)

KERNEL_SRC = r'''
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _panel_ablate_kernel(
    A_ptr, L_ptr,
    bk,
    stride_b, stride_r, stride_c,
    n,
    NB: gl.constexpr,
    BLOCK_I: gl.constexpr,
    L1_ITERS: gl.constexpr,
    L2_ITERS: gl.constexpr,
    NO_LOAD: gl.constexpr = False,
    DEFER_STORE: gl.constexpr = False,
):
    TILE_LAYOUT: gl.constexpr = gl.BlockedLayout([1, NB], [32, 1], [1, 1], [1, 0])
    ROW_LAYOUT: gl.constexpr = gl.SliceLayout(dim=1, parent=TILE_LAYOUT)
    COL_LAYOUT: gl.constexpr = gl.SliceLayout(dim=0, parent=TILE_LAYOUT)

    b = gl.program_id(0)
    pid_i = gl.program_id(1)

    idx_row = gl.arange(0, NB, layout=ROW_LAYOUT)
    idx_col = gl.arange(0, NB, layout=COL_LAYOUT)
    idx_row_valid = idx_row < NB
    row_idx = idx_row[:, None]
    col_idx = idx_col[None, :]

    panel_ptr = (L_ptr + b * stride_b
                 + (bk + idx_row)[:, None] * stride_r
                 + (bk + idx_col)[None, :] * stride_c)

    if L1_ITERS > 0:
        Lp = gl.zeros((NB, NB), dtype=gl.float32, layout=TILE_LAYOUT)
        for jp in range(L1_ITERS):
            row_jp = gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)
            contrib = gl.sum(Lp * row_jp[None, :], axis=1)

            if NO_LOAD:
                # same dependency chain, no global memory traffic
                a_col = (idx_row + jp).to(gl.float32) * 0.01 + 1.0
            else:
                a_col = gl.load(
                    A_ptr + b * stride_b + (bk + idx_row) * stride_r + (bk + jp) * stride_c,
                    mask=idx_row_valid, other=0.0,
                )
            diff = a_col - contrib
            ljj = gl.sqrt(gl.sum(gl.where(idx_row == jp, diff, 0.0), axis=0))
            new_col = gl.where(idx_row == jp, ljj, gl.where(idx_row > jp, diff / ljj, 0.0))
            Lp = gl.where(col_idx == jp, new_col[:, None], Lp)
    else:
        # stand-in panel so loop 2 still has a real dependency to consume
        Lp = gl.load(panel_ptr)

    if L2_ITERS > 0:
        i_idx = gl.arange(0, BLOCK_I, layout=ROW_LAYOUT)
        i = bk + pid_i * BLOCK_I + i_idx
        row_mask = i < n

        L_rows = gl.zeros((BLOCK_I, NB), dtype=gl.float32, layout=TILE_LAYOUT)
        for jp in range(L2_ITERS):
            row_jp = gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)
            contrib = gl.sum(L_rows * row_jp[None, :], axis=1)

            if NO_LOAD:
                a_val = (i_idx + jp).to(gl.float32) * 0.01 + 1.0
            else:
                a_val = gl.load(
                    A_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
                    mask=row_mask, other=0.0,
                )
            diff = a_val - contrib
            ljj = gl.sum(gl.where(idx_col == jp, row_jp, 0.0), axis=0)

            is_diag = i == (bk + jp)
            val = gl.where(is_diag, ljj, diff / ljj)
            L_rows = gl.where(col_idx == jp, val[:, None], L_rows)

            if not DEFER_STORE:
                gl.store(
                    L_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
                    val, mask=row_mask,
                )
        if DEFER_STORE:
            # one coalesced tile store instead of 32 in-loop column scatters
            gl.store(
                L_ptr + b * stride_b + i[:, None] * stride_r
                + (bk + idx_col)[None, :] * stride_c,
                L_rows, mask=row_mask[:, None],
            )
    else:
        # consume Lp so loop 1 cannot be dead-code eliminated
        gl.store(panel_ptr, Lp)
'''

# (name, L1_ITERS, L2_ITERS, NO_LOAD, DEFER_STORE)
VARIANTS = [
    ("shell", 0, 0, False, False),
    ("L1x8", 8, 0, False, False),
    ("L1x16", 16, 0, False, False),
    ("L1x24", 24, 0, False, False),
    ("L1x32", 32, 0, False, False),
    ("L2x8", 0, 8, False, False),
    ("L2x16", 0, 16, False, False),
    ("L2x24", 0, 24, False, False),
    ("L2x32", 0, 32, False, False),
    ("full", 32, 32, False, False),
    # memory-vs-math isolation at the real trip count
    ("L1x32_noload", 32, 0, True, False),
    ("L2x32_noload", 0, 32, True, False),
    ("L2x32_tilest", 0, 32, False, True),
    ("L2x32_nomem", 0, 32, True, True),
    ("full_nomem", 32, 32, True, True),
]

REPS = 20


@app.function(gpu="B200", timeout=3600)
def ablate(n: int = 8192, batch: int = 1):
    import collections
    import json
    import sys

    import torch
    import triton

    sys.path.insert(0, "/root/python_standalone")
    # triton's @jit requires the function to live in a real .py file
    with open("/root/_panel_ablate.py", "w") as f:
        f.write(KERNEL_SRC)
    sys.path.insert(0, "/root")
    import _panel_ablate
    kern = _panel_ablate._panel_ablate_kernel

    NB, BLOCK_I = 32, 32
    A = torch.randn((batch, n, n), device="cuda", dtype=torch.float32)
    A = (A @ A.transpose(-1, -2)) / n
    A.diagonal(dim1=-2, dim2=-1).add_(1.0)
    A = A.contiguous()
    L = torch.zeros_like(A)
    sb, sr, sc = L.stride()

    # grid matches the real kernel's FIRST launch (bk=0, n=8192 -> 256 blocks)
    grid = (batch, triton.cdiv(n, BLOCK_I))

    def launch(l1, l2, nl, ds):
        return kern[grid](A, L, 0, sb, sr, sc, n,
                          NB=NB, BLOCK_I=BLOCK_I, L1_ITERS=l1, L2_ITERS=l2,
                          NO_LOAD=nl, DEFER_STORE=ds,
                          num_warps=1)

    handles = {}
    for name, l1, l2, nl, ds in VARIANTS:  # warmup / compile
        handles[name] = launch(l1, l2, nl, ds)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for name, l1, l2, nl, ds in VARIANTS:
            for _ in range(REPS):
                launch(l1, l2, nl, ds)
            torch.cuda.synchronize()
    prof.export_chrome_trace("/tmp/ablate.json")

    ev = json.load(open("/tmp/ablate.json"))["traceEvents"]
    ks = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    ks = [e for e in ks if "_panel_ablate_kernel" in e["name"]]
    assert len(ks) == len(VARIANTS) * REPS, (len(ks), len(VARIANTS) * REPS)

    durs, regs = {}, {}
    for vi, (name, l1, l2, nl, ds) in enumerate(VARIANTS):
        chunk = [e["dur"] for e in ks[vi * REPS:(vi + 1) * REPS]]
        chunk.sort()
        durs[name] = {
            "median": chunk[len(chunk) // 2],
            "min": chunk[0],
            "max": chunk[-1],
        }
        regs[name] = ks[vi * REPS]["args"].get("registers per thread", -1)

    # PTX instruction mix per variant (DCE-proof: counted post-optimization)
    def ptx_mix(handle):
        ptx = handle.asm["ptx"]
        c = collections.Counter()
        for line in ptx.splitlines():
            line = line.strip()
            if not line or line.startswith(("//", ".", "$", "@%", "{", "}")):
                continue
            op = line.split()[0].lstrip("@!%p0123456789 ")
            if not op or not op[0].isalpha():
                continue
            c[op.split(".")[0]] += 1
        return dict(c)

    mixes = {v[0]: ptx_mix(handles[v[0]]) for v in VARIANTS}

    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "grid": list(grid),
        "durs": durs,
        "mixes": mixes,
        "regs": regs,
    }


@app.local_entrypoint()
def main(n: int = 8192, batch: int = 1):
    r = ablate.remote(n, batch)
    print("=" * 78)
    print(f"{r['gpu']}  torch {r['torch']}  triton {r['triton']}")
    print(f"case: batch={batch} n={n} grid={r['grid']} num_warps=1, {REPS} reps/variant")
    print("=" * 78)
    print(f"{'variant':10} {'median us':>10} {'min':>8} {'max':>8} {'regs':>6}")
    for name, *_ in VARIANTS:
        d = r["durs"][name]
        print(f"{name:10} {d['median']:10.2f} {d['min']:8.2f} {d['max']:8.2f} {r['regs'][name]:6}")

    shell = r["durs"]["shell"]["median"]
    print()
    print("--- loop attribution (isolated variant minus shell) ---")
    l1 = r["durs"]["L1x32"]["median"] - shell
    l2 = r["durs"]["L2x32"]["median"] - shell
    fu = r["durs"]["full"]["median"]
    print(f"  loop 1 (diag factorization), 32 iters : {l1:6.2f} us  ({l1/32*1000:5.0f} ns/iter)")
    print(f"  loop 2 (TRSM rows),          32 iters : {l2:6.2f} us  ({l2/32*1000:5.0f} ns/iter)")
    print(f"  fixed shell (prologue+epilogue+launch): {shell:6.2f} us")
    print(f"  sum {l1+l2+shell:6.2f} us  vs measured full {fu:6.2f} us"
          f"   (overcount {100*(l1+l2+shell-fu)/fu:+.1f}% = cross-loop overlap)")
    print(f"  => share of full: L1 {100*l1/fu:.0f}%, L2 {100*l2/fu:.0f}%, shell {100*shell/fu:.0f}%")

    print()
    print("--- L1 sweep is quadratic in trip count (compiler folds Lp zero cols) ---")
    for k in (8, 16, 24, 32):
        t = r["durs"][f"L1x{k}"]["median"] - shell
        print(f"  L1x{k:<2}: loop {t:6.2f} us   t/k {t/k*1000:6.0f} ns   t/k^2 {t/k**2*1000:6.1f} ns")

    print()
    print("--- memory vs math at the real trip count (32 iters) ---")
    g = lambda k: r["durs"][k]["median"]
    print(f"  loop1: with gather {g('L1x32'):6.2f}  no gather {g('L1x32_noload'):6.2f}"
          f"   -> gather costs {g('L1x32')-g('L1x32_noload'):5.2f} us")
    print(f"  loop2: baseline   {g('L2x32'):6.2f}  no gather {g('L2x32_noload'):6.2f}"
          f"   -> gather costs {g('L2x32')-g('L2x32_noload'):5.2f} us")
    print(f"  loop2: baseline   {g('L2x32'):6.2f}  tile store{g('L2x32_tilest'):6.2f}"
          f"   -> 32 scatters cost {g('L2x32')-g('L2x32_tilest'):5.2f} us")
    print(f"  loop2: baseline   {g('L2x32'):6.2f}  no mem    {g('L2x32_nomem'):6.2f}"
          f"   -> all mem costs {g('L2x32')-g('L2x32_nomem'):5.2f} us")
    print(f"  full : baseline   {g('full'):6.2f}  no mem    {g('full_nomem'):6.2f}"
          f"   -> all mem costs {g('full')-g('full_nomem'):5.2f} us"
          f"  ({100*(g('full')-g('full_nomem'))/g('full'):.0f}% of kernel)")
    print(f"  => residual (cross-lane reductions + fp math, no memory): {g('full_nomem'):.2f} us")

    print()
    print("--- is the loop unrolled? raw PTX totals across the k-sweep ---")
    print(f"{'variant':10} {'ptx total':>10} {'shfl':>7} {'fma':>7} {'ld':>7} {'st':>7} {'setp':>7}")
    for name, *_ in VARIANTS:
        m = r["mixes"][name]
        tot = sum(m.values())
        print(f"{name:10} {tot:10} {m.get('shfl',0):7} {m.get('fma',0):7}"
              f" {m.get('ld',0):7} {m.get('st',0):7} {m.get('setp',0):7}")

    print()
    print("--- PTX loop-body mix (delta L1x32 - L1x24, i.e. cost of 8 more iters) ---")
    keys = set(r["mixes"]["L1x32"]) | set(r["mixes"]["L2x32"])
    rows = []
    for op in sorted(keys):
        d1 = (r["mixes"]["L1x32"].get(op, 0) - r["mixes"]["L1x24"].get(op, 0)) / 8
        d2 = (r["mixes"]["L2x32"].get(op, 0) - r["mixes"]["L2x24"].get(op, 0)) / 8
        if abs(d1) > 0.05 or abs(d2) > 0.05:
            rows.append((op, d1, d2))
    print(f"{'ptx op':14} {'L1 /iter':>10} {'L2 /iter':>10}")
    for op, d1, d2 in sorted(rows, key=lambda x: -(x[1] + x[2])):
        print(f"{op:14} {d1:10.2f} {d2:10.2f}")
