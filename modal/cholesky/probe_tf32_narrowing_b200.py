"""Does tcgen05_mma narrow an fp32 operand to tf32 by truncation or round-to-nearest?

acc[m,n] = sum_k a[m,k]*b[n,k]. Set b[n,k]=(k==0), a[m,k]=x[m].
Then acc[m,0] = narrow(x[m]) * 1.0, and 1.0 is exact in tf32.
Compare readback bits against trunc(x) and rn(x) on values where they differ.
"""
import modal

import os as _os
# Kernel sources live one level up, in the cholesky_py problem directory.
LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "triton")
         .add_local_dir(LOCAL_DIR, "/root/python_standalone"))
app = modal.App("probe-tf32-narrowing", image=image)

SRC = '''
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_reg_layout,
    mbarrier, tcgen05_mma, fence_async_shared,
)

@gluon.jit
def probe(x_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
          num_warps: gl.constexpr):
    lay: gl.constexpr = gl.NVMMASharedLayout.get_default_for([M, K], gl.float32)
    REG: gl.constexpr = gl.BlockedLayout([1, 16], [16, 2], [num_warps, 1], [1, 0])
    rr: gl.constexpr = gl.SliceLayout(dim=1, parent=REG)
    cc: gl.constexpr = gl.SliceLayout(dim=0, parent=REG)

    m = gl.arange(0, M, layout=rr)[:, None]
    k = gl.arange(0, K, layout=cc)[None, :]

    a_reg = gl.load(x_ptr + m + k * 0)
    b_reg = gl.where(k == 0, 1.0, 0.0) + m * 0.0

    a_smem = gl.allocate_shared_memory(gl.float32, [M, K], lay)
    b_smem = gl.allocate_shared_memory(gl.float32, [N, K], lay)
    a_smem.store(a_reg)
    b_smem.store(b_reg)
    fence_async_shared()

    tl: gl.constexpr = TensorMemoryLayout([M, N], col_stride=1)
    acc = allocate_tensor_memory(gl.float32, [M, N], tl)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma(a_smem, b_smem.permute((1, 0)), acc,
                use_acc=False, mbarriers=[bar], mbarrier_preds=[True])
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    arl: gl.constexpr = get_tmem_reg_layout(gl.float32, (M, N), tl, num_warps)
    o = acc.load(arl)
    mm = gl.arange(0, M, layout=gl.SliceLayout(dim=1, parent=arl))[:, None]
    nn = gl.arange(0, N, layout=gl.SliceLayout(dim=0, parent=arl))[None, :]
    gl.store(out_ptr + mm * N + nn, o)
'''


@app.function(gpu="B200", timeout=1800)
def run():
    import struct, sys, traceback
    sys.path.insert(0, "/root/python_standalone")
    open("/root/python_standalone/_probe.py", "w").write(SRC)
    import torch
    f2b = lambda f: struct.unpack('<I', struct.pack('<f', f))[0]
    b2f = lambda b: struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]
    try:
        import _probe
        base = f2b(1.5)
        vals = [b2f((base & 0xFFFFE000) | ((i * 127) & 0x1FFF)) for i in range(64)]
        x = torch.tensor(vals, dtype=torch.float32, device="cuda")
        out = torch.zeros((64, 64), dtype=torch.float32, device="cuda")
        _probe.probe[(1,)](x, out, M=64, N=64, K=32, num_warps=4)
        torch.cuda.synchronize()
    except Exception:
        return {"fail": traceback.format_exc()[-3000:]}

    nt = nr = no = 0
    ex = []
    for v, g in zip(vals, out[:, 0].tolist()):
        t = b2f(f2b(v) & 0xFFFFE000)
        rn = b2f((f2b(v) + 0x1000) & 0xFFFFE000)
        if t == rn:
            continue
        if g == t: nt += 1
        elif g == rn: nr += 1
        else: no += 1
        if len(ex) < 5:
            ex.append((f"{f2b(v):08x}", f"{f2b(g):08x}", f"{f2b(t):08x}", f"{f2b(rn):08x}"))
    return {"trunc": nt, "rn": nr, "other": no, "tested": nt + nr + no, "ex": ex}


@app.local_entrypoint()
def main():
    r = run.remote()
    if "fail" in r:
        print("FAILED:\n", r["fail"]); return
    print(f"cases where trunc != rn : {r['tested']}")
    print(f"  matched TRUNCATION     : {r['trunc']}")
    print(f"  matched ROUND-TO-NEAREST: {r['rn']}")
    print(f"  matched neither        : {r['other']}")
    print(f"\n{'x':>10}{'mma out':>10}{'trunc':>10}{'rn':>10}")
    for a, b, c, d in r["ex"]:
        print(f"{a:>10}{b:>10}{c:>10}{d:>10}")
