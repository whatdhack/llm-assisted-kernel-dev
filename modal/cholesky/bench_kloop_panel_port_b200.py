"""Measure the panel port into cholesky_gluon_tcgen05_kloop.py on the
official 15-case suite, and capture kloop's occupancy resources.

THE PORT (four changes, all inside _panel_trsm_kernel, all NB-independent):
    1. gl.gather for the loop-1 row extraction
    2. Lp staged in shared memory for loop 2, via gl.static_range + jp_valid
    3. nb_valid derived in-kernel with gl.minimum
    4. the now-dead nb_valid host parameter dropped
Nothing else changes: kloop keeps NB_INNER=64, BLOCK_N=64, its chained-
accumulator SYRK, and no warp specialization -- all of those decisions depend
on THIS measurement.

Three kernels are timed per shape so the 5.7ms panel gap at n=8192 can be
SPLIT rather than assumed:
    stale  = kloop as of 2026-08-08, reconstructed here by reverse-patching
             the ported file back to the masked-reduce panel
    kloop  = kloop with the port
    base   = cholesky_gluon_tcgen05_blocked.py (2.068 geomean reference)
(stale - kloop) is the part of the gap that was stale panel code;
(kloop - base) is what is left, i.e. the real NB_INNER=64 / k-loop cost.

PROTOCOL: bench_leaderboard.bench_one (warmup 2, median of 5, fresh input),
all 15 official specs, variants INTERLEAVED within each shape -- run-to-run
drift on the sub-0.15ms shapes has buried a real 0.6% effect before when
measured variant-at-a-time.

Usage: modal run bench_kloop_panel_port_b200.py
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

# max_containers=2: an unbounded fan-out has tripped the 10-GPU account limit.
app = modal.App("cholesky-kloop-panel-port", image=image)

# The panel kernel exactly as it stood on 2026-08-08, before the port. Used to
# rebuild the stale variant in-container so both points are measured in ONE
# process on ONE GPU, rather than compared against a number from another run.
OLD_PANEL = '''@gluon.jit
def _panel_trsm_kernel(
    A_ptr, L_ptr,
    bk,
    stride_b, stride_r, stride_c,
    n,
    NB: gl.constexpr,
    nb_valid: gl.constexpr,
    BLOCK_I: gl.constexpr,
):
    TILE_LAYOUT: gl.constexpr = gl.BlockedLayout([1, NB], [32, 1], [1, 1], [1, 0])
    ROW_LAYOUT: gl.constexpr = gl.SliceLayout(dim=1, parent=TILE_LAYOUT)
    COL_LAYOUT: gl.constexpr = gl.SliceLayout(dim=0, parent=TILE_LAYOUT)

    b = gl.program_id(0)
    pid_i = gl.program_id(1)

    idx_row = gl.arange(0, NB, layout=ROW_LAYOUT)
    idx_col = gl.arange(0, NB, layout=COL_LAYOUT)
    idx_row_valid = idx_row < nb_valid
    row_idx = idx_row[:, None]
    col_idx = idx_col[None, :]

    Lp = gl.zeros((NB, NB), dtype=gl.float32, layout=TILE_LAYOUT)
    for jp in range(nb_valid):
        row_jp = gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)
        contrib = gl.sum(Lp * row_jp[None, :], axis=1)

        a_col = gl.load(
            A_ptr + b * stride_b + (bk + idx_row) * stride_r + (bk + jp) * stride_c,
            mask=idx_row_valid, other=0.0,
        )
        diff = a_col - contrib
        ljj = gl.sqrt(gl.sum(gl.where(idx_row == jp, diff, 0.0), axis=0))
        new_col = gl.where(idx_row == jp, ljj, gl.where(idx_row > jp, diff / ljj, 0.0))
        Lp = gl.where(col_idx == jp, new_col[:, None], Lp)

    i_idx = gl.arange(0, BLOCK_I, layout=ROW_LAYOUT)
    i = bk + pid_i * BLOCK_I + i_idx
    row_mask = i < n

    L_rows = gl.zeros((BLOCK_I, NB), dtype=gl.float32, layout=TILE_LAYOUT)
    for jp in range(nb_valid):
        row_jp = gl.sum(gl.where(row_idx == jp, Lp, 0.0), axis=0)
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

# The pre-port host call site: nb_valid was a constexpr kernel argument.
OLD_CALL = """                A, L, bk, stride_b, stride_r, stride_c, n,
                NB=NB_INNER, nb_valid=nb_valid, BLOCK_I=BLOCK_I,
                num_warps=1,
            )"""
NEW_CALL = """                A, L, bk, stride_b, stride_r, stride_c, n,
                NB=NB_INNER, BLOCK_I=BLOCK_I,
                num_warps=1,
            )"""

STALE_NAME = "_kloop_stale"


def _write_stale():
    """Reverse-patch the ported kloop back to its 2026-08-08 panel.

    Triton's @jit needs the source in a real .py file on disk (exec() of a
    string does not work), so the variant is written out and imported.
    """
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_kloop.py").read()
    s = src.index("@gluon.jit\ndef _panel_trsm_kernel(")
    e = src.index("@gluon.jit\ndef _round_tf32(")
    out = src[:s] + OLD_PANEL + "\n\n" + src[e:]
    assert OLD_CALL not in out and NEW_CALL in out, "host call site not as expected"
    out = out.replace(NEW_CALL, OLD_CALL)
    open(f"/root/python_standalone/{STALE_NAME}.py", "w").write(out)


def _resources(mods, spec):
    """Per-kernel shared memory / registers / blocks-per-SM from a Kineto trace.

    smem and regs come from the trace's kernel args ("shared memory",
    "registers per thread"); there is no metadata.num_regs to read.
    """
    import glob, json, os, torch
    import bench_leaderboard

    props = torch.cuda.get_device_properties(0)
    smem_per_sm = getattr(props, "shared_memory_per_multiprocessor", 233472)
    regs_per_sm = getattr(props, "regs_per_multiprocessor", 65536)
    thr_per_sm = getattr(props, "max_threads_per_multi_processor", 2048)

    A = bench_leaderboard.generate_input(spec["batch"], spec["n"], spec["cond"], spec["seed"])
    out = {"smem_per_sm": smem_per_sm, "regs_per_sm": regs_per_sm,
           "threads_per_sm": thr_per_sm, "kernels": {}}
    for tag, m in mods:
        m.custom_kernel(A)
        torch.cuda.synchronize()
        path = f"/tmp/trace_{tag}.json"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], with_stack=False,
        ) as prof:
            m.custom_kernel(A)
            torch.cuda.synchronize()
        prof.export_chrome_trace(path)
        ev = json.load(open(path))["traceEvents"]
        agg = {}
        for e in ev:
            if e.get("cat") != "kernel":
                continue
            a = e.get("args", {})
            smem = a.get("shared memory")
            regs = a.get("registers per thread")
            if smem is None or regs is None:
                continue
            blk = a.get("block", [0, 0, 0])
            threads = int(blk[0]) * max(int(blk[1]), 1) * max(int(blk[2]), 1)
            k = (e["name"].split("(")[0][:60], int(smem), int(regs), threads)
            d = agg.setdefault(k, {"launches": 0, "us": 0.0})
            d["launches"] += 1
            d["us"] += float(e.get("dur", 0))
        rows = []
        for (name, smem, regs, threads), d in sorted(agg.items(), key=lambda kv: -kv[1]["us"]):
            by_smem = smem_per_sm // smem if smem else 99
            by_regs = regs_per_sm // (regs * threads) if regs * threads else 99
            by_thr = thr_per_sm // threads if threads else 99
            rows.append({
                "name": name, "smem": smem, "regs": regs, "threads": threads,
                "launches": d["launches"], "ms": d["us"] / 1e3,
                "blocks_sm_smem": by_smem, "blocks_sm_regs": by_regs,
                "blocks_sm_threads": by_thr,
                "blocks_sm": min(by_smem, by_regs, by_thr),
            })
        out["kernels"][tag] = rows
        os.remove(path)
    del A
    torch.cuda.empty_cache()
    return out


@app.function(gpu="B200", timeout=7200, max_containers=2)
def bench():
    import importlib, sys, traceback
    import torch
    sys.path.insert(0, "/root/python_standalone")
    _write_stale()

    import bench_leaderboard
    try:
        mods = [
            ("starter", importlib.import_module("starter")),
            ("base", importlib.import_module("cholesky_gluon_tcgen05_blocked")),
            ("kloop", importlib.import_module("cholesky_gluon_tcgen05_kloop")),
            ("stale", importlib.import_module(STALE_NAME)),
        ]
    except Exception:
        return {"fail": traceback.format_exc()[-3000:]}
    mod = dict(mods)

    rows = []
    for spec in bench_leaderboard.BENCHMARKS:
        batch, n = spec["batch"], spec["n"]
        row = {"batch": batch, "n": n}
        # INTERLEAVED: every variant is timed inside this shape's block, so
        # slow drift hits all of them equally instead of one column.
        for tag, m in mods:
            try:
                row[f"{tag}_ms"] = bench_leaderboard.bench_one(m.custom_kernel, spec) * 1e3
            except Exception:
                return {"fail": f"{batch}x{n} {tag}\n" + traceback.format_exc()[-2500:]}
        A = bench_leaderboard.generate_input(batch, n, spec["cond"], spec["seed"])
        for tag in ("base", "kloop", "stale"):
            L = mod[tag].custom_kernel(A)
            torch.cuda.synchronize()
            row[f"{tag}_err"] = (L @ L.transpose(-1, -2) - A).abs().amax().item()
            row[f"{tag}_nan"] = int((~torch.isfinite(L)).sum().item())
            del L
        rows.append(row)
        del A
        torch.cuda.empty_cache()

    # Residency gate for the warp-specialization question, plus the panel/SYRK
    # split at the shape where the two kernels are closest to even. Guarded:
    # a profiler failure must not discard 15 cases of timing already in hand.
    try:
        res = _resources([("kloop", mod["kloop"]), ("base", mod["base"]),
                          ("stale", mod["stale"])],
                         {"batch": 1, "n": 8192, "cond": 2, "seed": 48192})
    except Exception:
        res = {"fail": traceback.format_exc()[-2000:]}
    return {"gpu": torch.cuda.get_device_name(0), "rows": rows, "res": res}


@app.local_entrypoint()
def main():
    import json, math
    r = bench.remote()
    if "fail" in r:
        print("RUN FAILED:\n", r["fail"])
        return
    # raw dump first: a formatting bug below must not cost a B200 run
    print("RAW_ROWS", json.dumps(r["rows"]))
    print("RAW_RES", json.dumps(r["res"]))

    print("=" * 104)
    print(f"{r['gpu']}   kloop NB_INNER=64 NB_OUTER=1024 KB=32 BLOCK_M=BLOCK_N=64")
    print("  stale = kloop @ 08-08 panel   kloop = + the four panel changes   "
          "base = tcgen05_blocked")
    print("=" * 104)
    print(f"{'batch':>6} {'n':>6} | {'cuSOLVER':>9} {'base':>9} {'stale':>9} {'kloop':>9} |"
          f" {'port':>7} {'kl/base':>8} | {'err':>9} {'nan':>4}")
    ss, bs, ks, ts, ds = [], [], [], [], []
    for x in r["rows"]:
        ss.append(x["starter_ms"]); bs.append(x["base_ms"])
        ks.append(x["kloop_ms"]); ts.append(x["stale_ms"])
        ds.append(min(x["base_ms"], x["kloop_ms"]))
        print(f"{x['batch']:6} {x['n']:6} | {x['starter_ms']:9.3f} {x['base_ms']:9.3f}"
              f" {x['stale_ms']:9.3f} {x['kloop_ms']:9.3f} |"
              f" {x['stale_ms'] / x['kloop_ms']:6.3f}x {x['base_ms'] / x['kloop_ms']:7.3f}x |"
              f" {x['kloop_err']:9.2e} {x['kloop_nan']:4}")
    g = lambda v: math.exp(sum(math.log(t) for t in v) / len(v))
    print("-" * 104)
    print(f"{'GEOMEAN':>13} | {g(ss):9.3f} {g(bs):9.3f} {g(ts):9.3f} {g(ks):9.3f} |"
          f" {g(ts) / g(ks):6.3f}x {g(bs) / g(ks):7.3f}x")
    print()
    print(f"  vs cuSOLVER:  base {g(bs) / g(ss):.3f}x   stale-kloop {g(ts) / g(ss):.3f}x"
          f"   ported-kloop {g(ks) / g(ss):.3f}x   oracle-dispatch {g(ds) / g(ss):.3f}x")
    print(f"  oracle dispatch min(base, kloop) geomean: {g(ds):.3f} ms"
          f"   kloop wins {sum(1 for x in r['rows'] if x['kloop_ms'] < x['base_ms'])}/15")
    print(f"  max err: base {max(x['base_err'] for x in r['rows']):.2e}"
          f"  kloop {max(x['kloop_err'] for x in r['rows']):.2e}"
          f"   total nonfinite: base {sum(x['base_nan'] for x in r['rows'])}"
          f"  kloop {sum(x['kloop_nan'] for x in r['rows'])}"
          f"  stale {sum(x['stale_nan'] for x in r['rows'])}")

    res = r["res"]
    if "fail" in res:
        print("\nRESOURCE CAPTURE FAILED (timings above are unaffected):\n", res["fail"])
        return
    print()
    print("=" * 104)
    print(f"RESOURCES at batch=1 n=8192 (Kineto trace args).  SM budget:"
          f" smem {res['smem_per_sm']}B  regs {res['regs_per_sm']}  threads {res['threads_per_sm']}")
    print("=" * 104)
    print(f"{'variant':>7} {'kernel':<34} {'smem':>8} {'regs':>5} {'thr':>5}"
          f" {'launch':>7} {'ms':>8} | {'blk/SM':>6} (smem/regs/thr)")
    for tag in ("kloop", "base", "stale"):
        for k in res["kernels"].get(tag, []):
            print(f"{tag:>7} {k['name']:<34} {k['smem']:8} {k['regs']:5} {k['threads']:5}"
                  f" {k['launches']:7} {k['ms']:8.3f} | {k['blocks_sm']:6}"
                  f" ({k['blocks_sm_smem']}/{k['blocks_sm_regs']}/{k['blocks_sm_threads']})")
    print()
    print("  GATE: >= 3 blocks/SM on kloop's SYRK means warp specialization would")
    print("  repeat warpspec's 22% loss (it trades inter-block overlap for intra-);")
    print("  1 means the cost argument that sank it does not apply.")
