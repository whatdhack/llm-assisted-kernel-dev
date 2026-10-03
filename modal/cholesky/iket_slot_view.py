"""Re-lay an IKET trace so each SM shows its CTAs on a few reusable rows.

run-iket's own .pftrace builds GPC -> TPC|VSM -> CTA -> Warp and gives EVERY
CTA its own track. A SYRK launch has ~55 CTAs per SM, each live for only a few
us, so an expanded VSM is ~55 mostly-empty rows and the first screenful only
shows the CTAs scheduled first -- which reads as "activity only at the start of
the launch". The data is all there; the layout hides it. run-iket has no option
for this (postprocess takes only --format/--output-dir).

This converter reads the run-iket JSON (the decoded results, streamed so a
400MB+ file stays in modest memory) and writes a Chrome-trace JSON that
Perfetto loads directly:

    process  = one per SM, named "GPC g | TPC t | VSM s", sorted by SM id
    threads  = SLOTS: CTAs on that SM greedily packed onto the fewest rows
               whose intervals do not overlap -- i.e. residency, so a kernel
               running 4 CTAs/SM shows exactly 4 slot rows, back to back in time

--detail picks how much goes under each slot:
    cta    one slice per CTA (default; small, the residency overview)
    warp0  plus warp 0's iket ranges beneath each slot
    all    plus every warp's iket ranges (large -- pair with --launch)

Timestamps are rebased to the first event and written in us. Gaps BETWEEN
launches in an IKET trace are profiler overhead (~70 us each, vs ~2 us under
Kineto); only timing inside a launch is meaningful.

Usage:
    python iket_slot_view.py <trace.json> [--detail cta|warp0|all] [--launch N]
    -> writes <trace>.slots[.<detail>][.launchN].json next to the input
"""
import argparse
import json
import os
import re
from array import array
from collections import defaultdict

RX = re.compile(
    rb'"gridDimX":(\d+),"gridDimY":(\d+),"gridDimZ":\d+,"gridId":\d+,"kernelName":"kernel_cutlass__(\w+?)_'
    rb'|"endTs":(\d+),"internalEvents":\[[^\]]*\],"rangeColor":\d+,"rangeId":\d+,"rangeNameIdx":(\d+),'
    rb'"rangeScope":\d+,"rangeType":\d+,"startTs":(\d+),"warpLocIdxs":\[(\d+)'
    rb'|\{"ctaId":\[(\d+),(\d+),(\d+)\],"gpcId":(-?\d+),"smId":(-?\d+),"tpcId":(-?\d+),"warpId":(\d+)\}'
)


def parse(path):
    """Stream the run-iket JSON: launches, per-launch ranges, location table,
    string table. Ranges are kept in flat arrays, not tuples, to bound memory."""
    launches, locs = [], []
    with open(path, "rb") as f:
        buf = b""
        while True:
            chunk = f.read(16 << 20)
            if not chunk and not buf:
                break
            buf += chunk
            last = 0
            for m in RX.finditer(buf):
                g = m.groups()
                if g[2]:
                    launches.append({"kind": g[2].decode(), "loc": array("q"),
                                     "name": array("q"), "s": array("q"), "e": array("q")})
                elif g[3]:
                    e, s = int(g[3]), int(g[5])
                    if 0 <= e - s < 10**9:          # drop the rare corrupt timestamp
                        L = launches[-1]
                        L["loc"].append(int(g[6])); L["name"].append(int(g[4]))
                        L["s"].append(s); L["e"].append(e)
                else:
                    locs.append(tuple(int(x) for x in g[7:]))
                last = m.end()
            buf = buf[last:] if last else buf[-4096:]
            if not chunk:
                break
        f.seek(0, 2)
        f.seek(max(0, f.tell() - (1 << 16)))
        names = json.loads(b"[" + f.read().rsplit(b'"stringTable":[', 1)[1].rstrip(b"}] \n") + b"]")
    return launches, locs, names


def pack_slots(intervals):
    """Greedy interval colouring: fewest rows with no overlap on any row.
    intervals: list of (start, end, key). Returns {key: slot}."""
    slot_end, out = [], {}
    for s, e, key in sorted(intervals):
        k = next((i for i, end in enumerate(slot_end) if end <= s), None)
        if k is None:
            k = len(slot_end)
            slot_end.append(e)
        else:
            slot_end[k] = e
        out[key] = k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--detail", choices=["cta", "warp0", "all"], default="cta")
    ap.add_argument("--launch", type=int, default=None, help="only this launch index")
    a = ap.parse_args()

    launches, locs, names = parse(a.trace)
    t0 = min(min(L["s"]) for L in launches if len(L["s"]))

    events, meta_pids = [], {}
    slots_used = defaultdict(int)            # (sm) -> max slots over all launches
    warp_rows = a.detail != "cta"
    for li, L in enumerate(launches):
        if a.launch is not None and li != a.launch:
            continue
        # CTA interval = min start / max end over all its warps' ranges.
        cta = {}
        for loc, s, e in zip(L["loc"], L["s"], L["e"]):
            cx, cy, cz, gpc, sm, tpc, wid = locs[loc]
            key = (sm, cx, cy, cz)
            v = cta.get(key)
            cta[key] = (s, e) if v is None else (min(v[0], s), max(v[1], e))
        per_sm = defaultdict(list)
        for (sm, cx, cy, cz), (s, e) in cta.items():
            per_sm[sm].append((s, e, (sm, cx, cy, cz)))
        slot = {}
        for sm, iv in per_sm.items():
            slot.update(pack_slots(iv))
            slots_used[sm] = max(slots_used[sm], 1 + max(slot[k] for *_, k in iv))

        stride = 5 if warp_rows else 1        # per slot: CTA row (+ 4 warp rows)
        for (sm, cx, cy, cz), (s, e) in cta.items():
            k = slot[(sm, cx, cy, cz)]
            events.append({"name": f"{L['kind']} cta({cx},{cy})", "ph": "X", "pid": sm + 1,
                           "tid": k * stride, "ts": (s - t0) / 1e3, "dur": (e - s) / 1e3,
                           "args": {"launch": li, "cta": [cx, cy, cz]}})
        if warp_rows:
            for loc, ni, s, e in zip(L["loc"], L["name"], L["s"], L["e"]):
                cx, cy, cz, gpc, sm, tpc, wid = locs[loc]
                if a.detail == "warp0" and wid != 0:
                    continue
                k = slot[(sm, cx, cy, cz)]
                events.append({"name": names[ni], "ph": "X", "pid": sm + 1,
                               "tid": k * stride + 1 + wid, "ts": (s - t0) / 1e3,
                               "dur": (e - s) / 1e3})
        for (cx, cy, cz, gpc, sm, tpc, wid) in locs:
            meta_pids.setdefault(sm, (gpc, tpc))

    meta = []
    for sm, (gpc, tpc) in sorted(meta_pids.items()):
        if sm not in slots_used:
            continue
        meta.append({"name": "process_name", "ph": "M", "pid": sm + 1,
                     "args": {"name": f"GPC {gpc} | TPC {tpc} | VSM {sm}"}})
        meta.append({"name": "process_sort_index", "ph": "M", "pid": sm + 1,
                     "args": {"sort_index": sm}})
        for k in range(slots_used[sm]):
            rows = [(k * 5, f"slot {k}")] + [(k * 5 + 1 + w, f"slot {k} warp {w}")
                                             for w in range(4)] if warp_rows else [(k, f"slot {k}")]
            for tid, label in rows:
                meta.append({"name": "thread_name", "ph": "M", "pid": sm + 1, "tid": tid,
                             "args": {"name": label}})
                meta.append({"name": "thread_sort_index", "ph": "M", "pid": sm + 1, "tid": tid,
                             "args": {"sort_index": tid}})

    base = os.path.splitext(a.trace)[0]
    if base.endswith(".trace"):
        base = base[: -len(".trace")]
    out = f"{base}.slots" + ("" if a.detail == "cta" else f".{a.detail}") \
        + ("" if a.launch is None else f".launch{a.launch}") + ".json"
    with open(out, "w") as f:
        f.write("[\n")
        for i, ev in enumerate(meta + events):
            f.write(("" if i == 0 else ",\n") + json.dumps(ev, separators=(",", ":")))
        f.write("\n]\n")
    hist = defaultdict(int)
    for n in slots_used.values():
        hist[n] += 1
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB, {len(events):,} slices)")
    print("max concurrent CTAs (slot rows) per SM:", dict(sorted(hist.items())))


if __name__ == "__main__":
    main()
