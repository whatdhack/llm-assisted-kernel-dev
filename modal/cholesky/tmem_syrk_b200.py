"""
How much TMEM (Blackwell tensor memory) does the SYRK tile actually allocate,
and what would 3 independent accumulators cost?

The 3 tf32x3 products (hi_m*hi_n, hi_m*lo_n, lo_m*hi_n) are mathematically
INDEPENDENT -- they are just summed. Today they are serialised only because
all three target one acc_tmem with use_acc=True, each followed by a full
mbarrier round-trip. Giving each its own accumulator would let them issue
back to back with a single wait, at the cost of 3x the TMEM.

TMEM is a hard per-SM resource (unlike registers it cannot spill), so this
measures whether 3 accumulators would cap block residency.

Usage: modal run tmem_syrk_b200.py
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

app = modal.App("cholesky-tmem-syrk-b200", image=image)

# one accumulator (current) vs three (independent products), at each BLOCK size
SRC_3ACC_OLD = """    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
"""
SRC_3ACC_NEW = """    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    acc_tmem2 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    acc_tmem3 = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
"""


@app.function(gpu="B200", timeout=1800)
def probe():
    import importlib
    import sys

    import torch
    import triton

    sys.path.insert(0, "/root/python_standalone")
    base = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert SRC_3ACC_OLD in base

    out = []
    for bmn in (64, 128):
        for nacc in (1, 3):
            s = base.replace("    BLOCK_M = BLOCK_N = 64\n",
                             f"    BLOCK_M = BLOCK_N = {bmn}\n", 1)
            if nacc == 3:
                # allocate 3 accumulators; the extra two are consumed so they
                # are not dead-code eliminated
                s = s.replace(SRC_3ACC_OLD, SRC_3ACC_NEW, 1)
                s = s.replace(
                    "    update = acc_tmem.load(acc_reg_layout)\n",
                    "    update = (acc_tmem.load(acc_reg_layout)\n"
                    "              + acc_tmem2.load(acc_reg_layout)\n"
                    "              + acc_tmem3.load(acc_reg_layout))\n", 1)
            name = f"_tm_b{bmn}_a{nacc}"
            open(f"/root/python_standalone/{name}.py", "w").write(s)
            try:
                mod = importlib.import_module(name)
                A = torch.randn(1, 512, 512, device="cuda")
                A = (A @ A.transpose(-1, -2)) / 512
                A.diagonal(dim1=-2, dim2=-1).add_(1.0)
                mod.custom_kernel(A.contiguous())
                torch.cuda.synchronize()
                # find the compiled syrk kernel and read its metadata
                k = mod._syrk_kernel_tcgen05
                cache = list(k.device_caches.values())[0][0]
                kern = next(iter(cache.values()))
                md = kern.metadata
                out.append({
                    "blk": bmn, "nacc": nacc, "ok": True,
                    "tmem": getattr(md, "tmem_size", None),
                    "smem": getattr(md, "shared", None),
                    "regs": getattr(md, "num_regs", getattr(md, "n_regs", None)),
                })
            except Exception as e:
                out.append({"blk": bmn, "nacc": nacc, "ok": False,
                            "msg": f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"})
            torch.cuda.empty_cache()
    return out


@app.local_entrypoint()
def main():
    r = probe.remote()
    TMEM_TOTAL = 256 * 1024          # 256 KB per SM on sm_100
    TMEM_COLS = 512                  # 128 lanes x 512 cols x 4B
    SMEM_SM = 233472
    print("=" * 86)
    print("SYRK TMEM / SMEM allocation per block  (B200, 256KB TMEM = 128 lanes x 512 cols)")
    print("=" * 86)
    print(f"{'BLOCK':>6} {'accs':>5} {'tmem cols':>10} {'tmem B':>9} {'smem B':>9} "
          f"{'blk/SM(tmem)':>13} {'blk/SM(smem)':>13}")
    for d in r:
        if not d["ok"]:
            print(f"{d['blk']:6} {d['nacc']:5}   FAIL  {d['msg']}")
            continue
        cols = d["tmem"]
        tb = (cols or 0) * 128 * 4
        bt = TMEM_COLS // cols if cols else 0
        bs = SMEM_SM // (d["smem"] + 1024) if d["smem"] else 0
        print(f"{d['blk']:6} {d['nacc']:5} {cols:10} {tb:9} {d['smem']:9} {bt:13} {bs:13}")
    print()
    print("blk/SM(tmem) = 512 columns / columns-per-block; blk/SM(smem) = 233,472 / bytes-per-block")
    print("residency is the MIN of these (and of the register and 32-block limits)")
