"""
Full 15-case leaderboard suite on a Modal B200 for all four points in this
session's progression, measured in ONE container on identical hardware:

    starter   cuSOLVER reference
    reduce    original masked-reduce row extraction   (== "tcgen05(batched)")
    gather    gl.gather row extraction
    smem      gather + Lp staged in shared memory     (current working tree)

Both older variants are reconstructed by reverse-patching the working tree,
so all three custom kernels differ only in the row-extraction strategy.

Also captures a Kineto trace of the current kernel at batch=1/n=8192.

Usage: modal run bench_smem_b200.py
"""
import os
import time

import modal

import os as _os
# Kernel sources live one level up, in the cholesky_py problem directory.
LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
TRACE_DIR = os.path.join(LOCAL_DIR, "outputs", "traces")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton")
    .add_local_dir(LOCAL_DIR, "/root/python_standalone")
)

app = modal.App("cholesky-bench-smem-b200", image=image)

GATHER_LOOP2 = '''    i_idx = gl.arange(0, BLOCK_I, layout=ROW_LAYOUT)
    i = bk + pid_i * BLOCK_I + i_idx
    row_mask = i < n

    # L21
    L_rows = gl.zeros((BLOCK_I, NB), dtype=gl.float32, layout=TILE_LAYOUT)
    for jp in range(nb_valid):
        jp_idx = gl.zeros([1, NB], gl.int32, layout=TILE_LAYOUT) + jp
        row_jp = gl.sum(gl.gather(Lp, jp_idx, 0), axis=0)
        contrib = gl.sum(L_rows * row_jp[None, :], axis=1)

        a_val = gl.load(
            A_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
            mask=row_mask, other=0.0,
        )
        diff = a_val - contrib
        ljj = gl.sum(gl.where(idx_col == jp, row_jp, 0.0), axis=0)

        is_diag = i == (bk + jp)
        val = gl.where(is_diag, ljj, diff / ljj)
        L_rows = gl.where(col_idx == jp, val[:, None], L_rows)

        gl.store(
            L_ptr + b * stride_b + i * stride_r + (bk + jp) * stride_c,
            val, mask=row_mask,
        )
'''


def _write_variants():
    import re
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert "Lp_smem" in src and "gl.gather" in src, "expected the smem working tree"

    # smem -> gather: swap loop 2 back to the cross-lane gather form
    start = src.index("    # Stage the finished factor into shared memory")
    end = src.index("@gluon.jit\ndef _round_tf32")
    gather = src[:start] + GATHER_LOOP2 + "\n\n" + src[end:]
    assert "Lp_smem" not in gather and "static_range" not in gather
    open("/root/python_standalone/_v_gather.py", "w").write(gather)

    # gather -> reduce: both row_jp extractions back to the masked butterfly
    reduce_ = re.sub(
        r" *jp_idx = gl\.zeros\(\[1, NB\], gl\.int32, layout=TILE_LAYOUT\) \+ jp\n"
        r" *row_jp = gl\.sum\(gl\.gather\(Lp, jp_idx, 0\), axis=0\).*\n",
        "        row_jp = gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)\n",
        gather)
    # check the CODE, not the prose: "gl.gather" still occurs in a comment
    assert "gl.gather(Lp" not in reduce_, "reverse-patch to reduce failed"
    assert reduce_.count("gl.where(row_idx == jp, Lp, 0.0)") == 2
    open("/root/python_standalone/_v_reduce.py", "w").write(reduce_)


@app.function(gpu="B200", timeout=7200)
def bench():
    import importlib
    import json
    import math
    import sys

    import torch
    import triton

    sys.path.insert(0, "/root/python_standalone")
    _write_variants()
    import bench_leaderboard

    MODULES = [
        ("starter", "starter"),
        ("reduce", "_v_reduce"),
        ("gather", "_v_gather"),
        ("smem", "cholesky_gluon_tcgen05_blocked"),
    ]

    results = {}
    for label, modname in MODULES:
        mod = importlib.import_module(modname)
        times, errs, nans = [], [], 0
        for spec in bench_leaderboard.BENCHMARKS:
            times.append(bench_leaderboard.bench_one(mod.custom_kernel, spec) * 1e3)
            if spec["n"] <= 2048:
                A = bench_leaderboard.generate_input(spec["batch"], spec["n"],
                                                     spec["cond"], spec["seed"])
                L = mod.custom_kernel(A)
                torch.cuda.synchronize()
                errs.append((L @ L.transpose(-1, -2) - A).abs().amax().item())
                nans += int((~torch.isfinite(L)).sum().item())
                del A, L
                torch.cuda.empty_cache()
        results[label] = {
            "times": times,
            "geomean": math.exp(sum(math.log(t) for t in times) / len(times)),
            "max_err": max(errs) if errs else None,
            "nans": nans,
        }

    # Kineto of the current kernel
    mod = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    A = bench_leaderboard.generate_input(1, 8192, 2, 48192)
    mod.custom_kernel(A)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        mod.custom_kernel(A)
        torch.cuda.synchronize()
    prof.export_chrome_trace("/tmp/kineto.json")
    ev = json.load(open("/tmp/kineto.json"))["traceEvents"]
    ks = [e for e in ev if e.get("cat") == "kernel"]
    agg = {}
    for e in ks:
        key = "panel" if "panel" in e["name"] else ("syrk" if "syrk" in e["name"] else "other")
        a = agg.setdefault(key, [0, 0.0])
        a[0] += 1
        a[1] += e["dur"]
    pk = next(e for e in ks if "panel" in e["name"])
    with open("/tmp/kineto.json", "rb") as f:
        trace = f.read()

    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "triton": triton.__version__,
        "specs": [(s["batch"], s["n"]) for s in bench_leaderboard.BENCHMARKS],
        "results": results,
        "kernel_split": {k: {"n": v[0], "ms": v[1] / 1000} for k, v in agg.items()},
        "panel_regs": pk["args"].get("registers per thread"),
        "panel_smem": pk["args"].get("shared memory"),
        "trace": trace,
    }


@app.local_entrypoint()
def main():
    import math
    r = bench.remote()
    print("=" * 104)
    print(f"{r['gpu']}  torch {r['torch']}  triton {r['triton']}")
    print("=" * 104)
    st = r["results"]["starter"]["times"]
    rd = r["results"]["reduce"]["times"]
    ga = r["results"]["gather"]["times"]
    sm = r["results"]["smem"]["times"]
    print(f"{'batch':>6} {'n':>6} | {'starter':>9} {'reduce':>9} {'gather':>9} {'smem':>9} |"
          f" {'vs reduce':>10} {'vs cuSOLVER':>12}")
    print("-" * 104)
    for (b, n), s, o, g, m in zip(r["specs"], st, rd, ga, sm):
        best = min(s, m)
        mark = "*" if m <= s else " "
        print(f"{b:6} {n:6} | {s:9.3f} {o:9.3f} {g:9.3f} {m:8.3f}{mark} |"
              f" {o/m:9.2f}x {m/s:11.2f}x")
    print("-" * 104)
    gs = r["results"]["starter"]["geomean"]
    gr = r["results"]["reduce"]["geomean"]
    gg = r["results"]["gather"]["geomean"]
    gm = r["results"]["smem"]["geomean"]
    print(f"{'GEOMEAN':>13} | {gs:9.3f} {gr:9.3f} {gg:9.3f} {gm:9.3f} |"
          f" {gr/gm:9.2f}x {gm/gs:11.2f}x")
    print()
    for lab in ("reduce", "gather", "smem"):
        d = r["results"][lab]
        print(f"  {lab:8} max |L L^T - A| (n<=2048) = {d['max_err']:.3e}   nonfinite: {d['nans']}")
    print()
    print(f"--- kernel split, batch=1 n=8192 (smem) --- panel regs {r['panel_regs']}, "
          f"smem {r['panel_smem']}B")
    for k, v in sorted(r["kernel_split"].items(), key=lambda x: -x[1]["ms"]):
        print(f"  {k:8} {v['ms']:8.3f} ms  x{v['n']}")

    os.makedirs(TRACE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    p = os.path.join(TRACE_DIR, f"cholesky_gluon_tcgen05_blocked__custom_kernel__{stamp}.json")
    with open(p, "wb") as f:
        f.write(r["trace"])
    print("\nkineto trace ->", p)
