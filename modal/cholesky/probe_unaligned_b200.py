"""
Determines exactly which n break cholesky_gluon_tcgen05_blocked.py, and
whether the nb_valid removal changed that. Each (n, variant) runs in its OWN
container: a CUDA misaligned-address error poisons the context, so probes
must not share one.

Usage: modal run probe_unaligned_b200.py
"""
import modal

import os as _os
# Kernel sources live one level up, in the cholesky_py problem directory.
LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "triton")
         .add_local_dir(LOCAL_DIR, "/root/python_standalone"))
app = modal.App("cholesky-probe-unaligned-b200", image=image)

NS = [8192, 8191, 8190, 8188, 4097, 2049, 257, 100, 36, 33]


def _make_orig():
    import re
    src = open("/root/python_standalone/cholesky_gluon_tcgen05_blocked.py").read()
    assert "gl.minimum" in src
    o = src.replace("    NB: gl.constexpr,\n    BLOCK_I: gl.constexpr,\n):",
                    "    NB: gl.constexpr,\n    nb_valid: gl.constexpr,\n    BLOCK_I: gl.constexpr,\n):", 1)
    o = re.sub(r"\n *# Width of this panel.*?\n *nb_valid = gl\.minimum\(NB, n - bk\)\n", "\n", o, flags=re.S)
    o = o.replace("    for bk in range(0, n, NB):\n        grid_panel",
                  "    for bk in range(0, n, NB):\n        nb_valid = min(NB, n - bk)\n        grid_panel", 1)
    o = o.replace("            NB=NB, BLOCK_I=BLOCK_I,", "            NB=NB, nb_valid=nb_valid, BLOCK_I=BLOCK_I,", 1)
    o = re.sub(r"\n *# Only a full NB-wide panel.*?\n *trailing = n - bk - NB\n *if trailing > 0:\n",
               "\n        trailing = n - (bk + nb_valid)\n        if trailing > 0:\n", o, flags=re.S)
    o = o.replace("            start = bk + NB", "            start = bk + nb_valid", 1)
    assert "nb_valid: gl.constexpr" in o and "gl.minimum" not in o
    open("/root/python_standalone/_orig_probe.py", "w").write(o)


# Each probe needs its OWN container (a CUDA misaligned-address error poisons
# the context), but starmap would otherwise fan out to one container per job
# and blow through the workspace GPU limit. Cap it: this just runs the probes
# in waves instead of all at once.
@app.function(gpu="B200", timeout=900, max_containers=2)
def probe(n: int, which: str):
    import importlib, sys, torch
    sys.path.insert(0, "/root/python_standalone")
    if which == "old":
        _make_orig()
        mod = importlib.import_module("_orig_probe")
    else:
        mod = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    import bench_leaderboard
    A = bench_leaderboard.generate_input(1, n, 2, 7)
    try:
        L = mod.custom_kernel(A)
        torch.cuda.synchronize()
        err = (L @ L.transpose(-1, -2) - A).abs().amax().item()
        return (n, which, "ok", err)
    except Exception as e:
        return (n, which, type(e).__name__, str(e).splitlines()[0][:60])


@app.local_entrypoint()
def main():
    jobs = [(n, w) for n in NS for w in ("old", "new")]
    res = {(n, w): r for n, w, *r in [x for x in probe.starmap(jobs)]}
    print("=" * 78)
    print(f"{'n':>6} {'n%32':>5} {'n*4%16':>7} | {'old':>28} | {'new':>28}")
    print("-" * 78)
    for n in NS:
        o, nw = res[(n, "old")], res[(n, "new")]
        fo = f"{o[0]} {o[1]:.2e}" if o[0] == "ok" else f"{o[0]}"
        fn = f"{nw[0]} {nw[1]:.2e}" if nw[0] == "ok" else f"{nw[0]}"
        same = "SAME" if (o[0] == "ok") == (nw[0] == "ok") else "*** DIFFERS ***"
        print(f"{n:6} {n%32:5} {n*4%16:7} | {fo:>28} | {fn:>28}  {same}")
