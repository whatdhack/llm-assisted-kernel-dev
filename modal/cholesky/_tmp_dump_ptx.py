"""One-off: dump Triton-generated PTX for the _syrk_kernel_tcgen05 gluon
kernel and check for .loc/.file line-info directives, to determine whether
line info is dropped in Triton's own codegen (no .loc in PTX at all) or
later at ptxas/ncu-import (.loc present in PTX but missing from the cubin
NCU sees).
"""
import os as _os
import modal

LOCAL_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "triton")
         .add_local_dir(LOCAL_DIR, "/root/python_standalone"))
app = modal.App("dump-ptx", image=image)


@app.function(gpu="B200", timeout=600)
def dump():
    import os
    os.environ["TRITON_DISABLE_LINE_INFO"] = "0"
    import sys
    sys.path.insert(0, "/root/python_standalone")
    import importlib
    import torch
    import bench_leaderboard

    mod = importlib.import_module("cholesky_gluon_tcgen05_blocked")
    A = bench_leaderboard.generate_input(1, 8192, 2, 48192)
    mod.custom_kernel(A)
    torch.cuda.synchronize()

    kern = mod._syrk_kernel_tcgen05
    info = {"type": str(type(kern)), "debug": getattr(kern, "debug", None),
            "starting_line_number": getattr(kern, "starting_line_number", None)}

    cache = getattr(kern, "device_caches", None)
    info["cache_type"] = str(type(cache))
    ptx_text = None
    asm_keys = None
    try:
        if isinstance(cache, dict):
            for dev_id, entries in cache.items():
                # device_caches[dev_id] is typically (cache_dict, target) or
                # similar; inspect and walk whatever dict we find.
                candidates = entries if isinstance(entries, (list, tuple)) else [entries]
                for cand in candidates:
                    if isinstance(cand, dict):
                        for key, compiled in cand.items():
                            asm = getattr(compiled, "asm", None)
                            if asm:
                                asm_keys = list(asm.keys())
                                if "ptx" in asm:
                                    ptx_text = asm["ptx"]
                                break
                    if ptx_text:
                        break
                if ptx_text:
                    break
    except Exception as e:
        info["cache_walk_error"] = repr(e)

    info["asm_keys"] = asm_keys
    info["ptx_len"] = len(ptx_text) if ptx_text else 0

    cubin_bytes = None
    try:
        if isinstance(cache, dict):
            for dev_id, entries in cache.items():
                candidates = entries if isinstance(entries, (list, tuple)) else [entries]
                for cand in candidates:
                    if isinstance(cand, dict):
                        for key, compiled in cand.items():
                            asm = getattr(compiled, "asm", None)
                            if asm and "cubin" in asm:
                                cubin_bytes = asm["cubin"]
                                break
                    if cubin_bytes:
                        break
                if cubin_bytes:
                    break
    except Exception as e:
        info["cubin_walk_error"] = repr(e)

    if cubin_bytes:
        info["cubin_len"] = len(cubin_bytes)
        info["has_debug_line_section"] = b".debug_line" in cubin_bytes
        info["has_debug_str_section"] = b".debug_str" in cubin_bytes
        info["has_debug_info_section"] = b".debug_info" in cubin_bytes
        info["has_nv_debug_ptx_txt"] = b".nv.debug_ptx_txt" in cubin_bytes
        info["has_source_filename_in_cubin"] = b"cholesky_gluon_tcgen05_blocked.py" in cubin_bytes
        # crude ELF section-name harvest: strings that look like section names
        import re
        sec_names = sorted(set(re.findall(rb"\.[a-zA-Z0-9_.]{3,30}", cubin_bytes)))
        info["cubin_section_like_strings_sample"] = [s.decode(errors="replace") for s in sec_names[:60]]

    if ptx_text:
        loc_lines = [l for l in ptx_text.splitlines() if l.strip().startswith(".loc")]
        file_lines = [l for l in ptx_text.splitlines() if l.strip().startswith(".file")]
        info["loc_count"] = len(loc_lines)
        info["file_count"] = len(file_lines)
        info["file_lines_sample"] = file_lines[:10]
        info["loc_lines_sample"] = loc_lines[:10]
        info["ptx_head"] = "\n".join(ptx_text.splitlines()[:40])

    return info, ptx_text


@app.local_entrypoint()
def main():
    info, ptx_text = dump.remote()
    import json
    print(json.dumps({k: v for k, v in info.items() if k != "ptx_head"}, indent=2, default=str))
    print("----- ptx head -----")
    print(info.get("ptx_head", ""))
    if ptx_text:
        with open("/tmp/syrk_tcgen05.ptx", "w") as f:
            f.write(ptx_text)
        print("full ptx written to /tmp/syrk_tcgen05.ptx")
