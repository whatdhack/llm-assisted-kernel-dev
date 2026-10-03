"""hi = untouched tile (hw narrowing); compare three lo formulations.
Also probes whether tcgen05 narrows fp32->tf32 by truncation or round-to-nearest.
"""
import modal
import os as _os
# Kernel sources live one level up, in the cholesky_py problem directory.
LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "triton").add_local_dir(LOCAL_DIR, "/root/python_standalone"))
app = modal.App("cholesky-trunc-both", image=image)

TRUNC = '''@gluon.jit
def _trunc_tf32(x):
    bits = x.to(gl.uint32, bitcast=True)
    return (bits & 0xFFFFE000).to(gl.float32, bitcast=True)


'''
OLD = """    l_m_reg = l_m_smem.load(REG_LAYOUT)
    l_m_hi = _round_tf32(l_m_reg)
    l_m_lo = _round_tf32(l_m_reg - l_m_hi)
    l_n_reg = l_n_smem.load(REG_LAYOUT)
    l_n_hi = _round_tf32(l_n_reg)
    l_n_lo = _round_tf32(l_n_reg - l_n_hi)"""
VARIANTS = {
 "A_rnlo":  """    l_m_reg = l_m_smem.load(REG_LAYOUT)
    l_m_lo = _round_tf32(l_m_reg - _trunc_tf32(l_m_reg))
    l_n_reg = l_n_smem.load(REG_LAYOUT)
    l_n_lo = _round_tf32(l_n_reg - _trunc_tf32(l_n_reg))""",
 "C_rawlo": """    l_m_reg = l_m_smem.load(REG_LAYOUT)
    l_m_lo = l_m_reg - _trunc_tf32(l_m_reg)
    l_n_reg = l_n_smem.load(REG_LAYOUT)
    l_n_lo = l_n_reg - _trunc_tf32(l_n_reg)"""}

PROBE = '''
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_reg_layout,
    mbarrier, tcgen05_mma, fence_async_shared)
M: gl.constexpr = 64
N: gl.constexpr = 64
K: gl.constexpr = 32
@gluon.jit
def probe(x_ptr, out_ptr, num_warps: gl.constexpr):
    lay: gl.constexpr = gl.NVMMASharedLayout.get_default_for([M, K], gl.float32)
    REG: gl.constexpr = gl.BlockedLayout([1, 16], [16, 2], [num_warps, 1], [1, 0])
    r: gl.constexpr = gl.SliceLayout(dim=1, parent=REG)
    c: gl.constexpr = gl.SliceLayout(dim=0, parent=REG)
    m = gl.arange(0, M, layout=r)[:, None]
    k = gl.arange(0, K, layout=c)[None, :]
    a = gl.load(x_ptr + m + k * 0)
    b = gl.where(k == 0, 1.0, 0.0) + m * 0.0
    a_s = gl.allocate_shared_memory(gl.float32, [M, K], lay)
    b_s = gl.allocate_shared_memory(gl.float32, [N, K], lay)
    a_s.store(a); b_s.store(b); fence_async_shared()
    tl: gl.constexpr = TensorMemoryLayout([M, N], col_stride=1)
    acc = allocate_tensor_memory(gl.float32, [M, N], tl)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma(a_s, b_s.permute((1, 0)), acc, use_acc=False, mbarriers=[bar], mbarrier_preds=[True])
    mbarrier.wait(bar, phase=0); mbarrier.invalidate(bar)
    arl: gl.constexpr = get_tmem_reg_layout(gl.float32, (M, N), tl, num_warps)
    o = acc.load(arl)
    mm = gl.arange(0, M, layout=gl.SliceLayout(dim=1, parent=arl))[:, None]
    nn = gl.arange(0, N, layout=gl.SliceLayout(dim=0, parent=arl))[None, :]
    gl.store(out_ptr + mm * N + nn, o)
'''

def _write(tag, block):
    s = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert OLD in s
    s = s.replace(OLD, block, 1)
    i = s.index("@gluon.jit\ndef _syrk_tile_tcgen05(")
    s = s[:i] + TRUNC + s[i:]
    open(f"/root/python_standalone/_v_{tag}.py", "w").write(s)
    return f"_v_{tag}"

@app.function(gpu="B200", timeout=7200, max_containers=2)
def bench():
    import importlib, struct, sys, traceback, math
    import torch
    sys.path.insert(0, "/root/python_standalone")
    out = {}
    # --- probe ---
    open("/root/python_standalone/_probe.py", "w").write(PROBE)
    try:
        import _probe
        f2b = lambda f: struct.unpack('<I', struct.pack('<f', f))[0]
        b2f = lambda b: struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]
        base = f2b(1.5)
        vals = [b2f((base & 0xFFFFE000) | ((i * 127) & 0x1FFF)) for i in range(64)]
        x = torch.tensor(vals, dtype=torch.float32, device="cuda")
        o = torch.zeros((64, 64), dtype=torch.float32, device="cuda")
        _probe.probe[(1,)](x, o, num_warps=4); torch.cuda.synchronize()
        nt = nr = no = 0
        for v, g in zip(vals, o[:, 0].tolist()):
            t = b2f(f2b(v) & 0xFFFFE000); rr = b2f((f2b(v) + 0x1000) & 0xFFFFE000)
            if t == rr: continue
            nt += (g == t); nr += (g == rr); no += (g != t and g != rr)
        out["probe"] = {"trunc": int(nt), "rn": int(nr), "other": int(no)}
    except Exception:
        out["probe"] = {"fail": traceback.format_exc()[-1500:]}
    # --- A/B ---
    import bench_leaderboard
    mods = [("cur", importlib.import_module("cholesky_gluon_tcgen05_blocked"))]
    for tag, blk in VARIANTS.items():
        try: mods.append((tag, importlib.import_module(_write(tag, blk))))
        except Exception: return {"fail": f"{tag}\n" + traceback.format_exc()[-3000:], **out}
    acc = []
    for bt, n in [(1, 512), (1, 2048), (1, 8192)]:
        A = bench_leaderboard.generate_input(bt, n, 2, 40000 + n)
        row = {"n": n}
        for t, m in mods:
            L = m.custom_kernel(A); torch.cuda.synchronize()
            row[t] = (L @ L.transpose(-1, -2) - A).abs().amax().item(); del L
        acc.append(row); del A; torch.cuda.empty_cache()
    rows = []
    for spec in bench_leaderboard.BENCHMARKS:
        r = {"batch": spec["batch"], "n": spec["n"]}
        for t, m in mods:
            try: r[t] = bench_leaderboard.bench_one(m.custom_kernel, spec) * 1e3
            except Exception: return {"fail": traceback.format_exc()[-2000:], "acc": acc, **out}
        rows.append(r); torch.cuda.empty_cache()
    out.update({"acc": acc, "rows": rows})
    return out

@app.local_entrypoint()
def main():
    import json, math
    r = bench.remote()
    if "fail" in r: print("FAILED:\n", r["fail"]); return
    p = r["probe"]
    print("PROBE tcgen05 fp32->tf32 narrowing:", 
          f"truncation={p.get('trunc')} rn={p.get('rn')} other={p.get('other')}" if "fail" not in p else p["fail"])
    tags = ["cur", "A_rnlo", "C_rawlo"]
    print("\nACC max|LL^T-A|")
    for x in r["acc"]:
        print(f"  n={x['n']:6} " + "  ".join(f"{t}={x[t]:.3e}" for t in tags))
    print(f"\n{'batch':>6} {'n':>6} |" + "".join(f"{t:>10}" for t in tags))
    g = {t: [] for t in tags}
    for x in r["rows"]:
        for t in tags: g[t].append(x[t])
        print(f"{x['batch']:6} {x['n']:6} |" + "".join(f"{x[t]:10.3f}" for t in tags))
    gm = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 46)
    print(f"{'GEOMEAN':>13} |" + "".join(f"{gm(g[t]):10.3f}" for t in tags))
