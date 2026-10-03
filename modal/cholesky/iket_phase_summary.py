"""Per-launch phase summary of a run-iket JSON trace (modal/iket_cute_b200.py).

Stream-parses the trace, so a multi-hundred-MB file fits in ~120MB of RAM.
For each panel and SYRK launch: launch span, and per iket range the median
per-warp duration and the number of warp-ranges recorded.

Usage: python iket_phase_summary.py outputs/traces/iket_cute_.../iket_pid_*.trace.json
"""
import re, sys, json
from array import array
path = sys.argv[1]
RX = re.compile(rb'"gridDimX":(\d+),"gridDimY":(\d+),"gridDimZ":\d+,"gridId":\d+,"kernelName":"kernel_cutlass__(\w{5})'
                rb'|"endTs":(\d+),"internalEvents":\[[^\]]*\],"rangeColor":\d+,"rangeId":\d+,"rangeNameIdx":(\d+),"rangeScope":\d+,"rangeType":\d+,"startTs":(\d+)')
launches = []   # dicts: kind, grid, t0, t1, dur{name_idx: array}
f = open(path, "rb"); buf = b""
while True:
    chunk = f.read(16 << 20)
    if not chunk and not buf: break
    buf += chunk; last = 0
    for m in RX.finditer(buf):
        if m.group(3):
            launches.append({"kind": m.group(3).decode(), "grid": int(m.group(1)) * int(m.group(2)),
                             "t0": 1 << 62, "t1": 0, "dur": {}})
        else:
            e, i, s = int(m.group(4)), int(m.group(5)), int(m.group(6))
            L = launches[-1]
            if not 0 <= e - s < 10**9:     # corrupt timestamp: count it, don't use it
                L["bad"] = L.get("bad", 0) + 1
                continue
            L["dur"].setdefault(i, array("q")).append(e - s)
            L["t0"] = min(L["t0"], s); L["t1"] = max(L["t1"], e)
        last = m.end()
    buf = buf[last:] if last else buf[-4096:]
    if not chunk: break
f.seek(0, 2); f.seek(max(0, f.tell() - 4096)); names = json.loads(b"[" + f.read().rsplit(b'"stringTable":[', 1)[1].rstrip(b"}]") + b"]")

def med(a):
    b = sorted(a); return b[len(b) // 2]
for kind, label in (("panel", "PANEL"), ("syrk_", "SYRK")):
    Ls = [L for L in launches if L["kind"] == kind]
    idxs = sorted({i for L in Ls for i in L["dur"]})
    print(f"\n{label}: median per-warp range duration in us (count of warp ranges)")
    print(f"{'':12}" + "".join(f"{'launch %d grid %d' % (k, L['grid']):>24}" for k, L in enumerate(Ls)))
    print(f"{'bad ts':12}" + "".join(f"{L.get('bad', 0):>24}" for L in Ls))
    print(f"{'span':12}" + "".join(f"{(L['t1'] - L['t0']) / 1e3:>24.1f}" for L in Ls))
    for i in idxs:
        row = ""
        for L in Ls:
            a = L["dur"].get(i)
            row += f"{'-':>24}" if not a else f"{med(a) / 1e3:>12.2f} ({len(a):>8,})"
        print(f"{names[i]:12}{row}")
    print(f"{'p99 tmem_alloc' if label == 'SYRK' else '':12}" + ("".join(
        f"{sorted(L['dur'][3])[int(len(L['dur'][3]) * .99)] / 1e3:>24.2f}" for L in Ls) if label == "SYRK" else ""))
