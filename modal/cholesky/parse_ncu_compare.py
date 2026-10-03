"""Parse ncu raw CSVs into a side-by-side comparison table.

Kept separate from the collection script so the profiles can be re-analysed
as many times as needed without paying for another B200 run.

Usage:
    python parse_ncu_compare.py label=path/to/a.raw.csv [label=b.raw.csv ...]
"""
import csv
import sys

# (display name, column, scale, format)
METRICS = [
    ("duration",              "gpu__time_duration.sum",                              1.0, "{:.2f}"),
    ("elapsed cycles",        "sm__cycles_elapsed.avg",                              1.0, "{:,.0f}"),
    ("warp instructions",     "smsp__inst_executed.sum",                             1.0, "{:,.0f}"),
    ("registers/thread",      "launch__registers_per_thread",                        1.0, "{:.0f}"),
    ("smem/block (B)",        "launch__shared_mem_per_block_allocated",              1.0, "{:,.0f}"),
    ("grid blocks",           "launch__grid_size",                                   1.0, "{:,.0f}"),
    ("waves/SM",              "launch__waves_per_multiprocessor",                    1.0, "{:.2f}"),
    ("achieved occupancy %",  "sm__warps_active.avg.pct_of_peak_sustained_active",   1.0, "{:.2f}"),
    ("SM throughput %",       "sm__throughput.avg.pct_of_peak_sustained_elapsed",    1.0, "{:.2f}"),
    # dram__throughput lives behind an unstable collection-unit prefix and is
    # often blank; bytes_read/write are always populated under --set full.
    ("DRAM read (MB)",        "dram__bytes_read.sum",                                1.0, "{:,.1f}"),
    ("DRAM write (MB)",       "dram__bytes_write.sum",                               1.0, "{:,.1f}"),
    ("DRAM read %peak",       "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",  1.0, "{:.2f}"),
    ("LSU pipe %",            "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active", 1.0, "{:.2f}"),
    ("ALU pipe %",            "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active", 1.0, "{:.2f}"),
    ("FMA pipe %",            "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active", 1.0, "{:.2f}"),
    ("warp cyc/inst issued",  "smsp__average_warp_latency_per_inst_issued.ratio",     1.0, "{:.2f}"),
]

STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"


def load(path):
    rows = list(csv.reader(open(path)))
    header, units, data = rows[0], rows[1], rows[2:]
    data = [d for d in data if d and d[0].strip()]
    idx = {h: i for i, h in enumerate(header)}
    return header, units, data, idx


def find(idx, col):
    """Exact column, else the first header containing it.

    ncu prefixes some metrics with a collection-unit tag (e.g.
    FBSP.TriageCompute.dram__throughput...) and the prefix is not stable
    across captures, so an exact-only lookup silently yields '-'.
    """
    if col in idx:
        return idx[col]
    hits = [h for h in idx if col in h]
    return idx[hits[0]] if hits else None


def num(v):
    try:
        return float(str(v).replace(",", ""))
    except (ValueError, TypeError):
        return None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(path):
    """One summary per DISTINCT kernel in the file.

    Averaging across kernel names would blend a panel launch with a syrk
    launch into a meaningless row, so rows are grouped by kernel first.
    """
    header, units, data, idx = load(path)
    groups = {}
    for d in data:
        groups.setdefault(d[idx["Kernel Name"]].split("(")[0].strip(), []).append(d)

    summaries = {}
    for kern, rows in groups.items():
        out = {"launches": len(rows), "kernel": kern,
               "grid": rows[0][idx["Grid Size"]], "block": rows[0][idx["Block Size"]],
               "metrics": {}, "units": {}, "stalls": {}}
        for name, col, scale, fmt in METRICS:
            i = find(idx, col)
            if i is None:
                out["metrics"][name] = None
                continue
            v = mean([num(d[i]) for d in rows])
            out["metrics"][name] = v * scale if v is not None else None
            out["units"][name] = units[i] if i < len(units) else ""
        for h in header:
            if h.startswith(STALL_PREFIX) and h.endswith(STALL_SUFFIX):
                v = mean([num(d[idx[h]]) for d in rows])
                if v:
                    out["stalls"][h[len(STALL_PREFIX):-len(STALL_SUFFIX)]] = v
        summaries[kern] = out
    return summaries


def main(args):
    pairs = [a.split("=", 1) for a in args]
    s = {}
    for label, path in pairs:
        per_kernel = summarize(path)
        if len(per_kernel) == 1:
            s[label] = next(iter(per_kernel.values()))
        else:  # mixed file: keep the kernels apart rather than averaging them
            for kern, summ in per_kernel.items():
                short = "panel" if "panel" in kern else ("syrk" if "syrk" in kern else kern[:10])
                s[f"{label}:{short}"] = summ
    labels = list(s)

    # panels and syrks side by side
    groups = [[l for l in labels if "panel" in l], [l for l in labels if "syrk" in l]]
    for group in groups:
        if not group:
            continue
        w = 20
        print("=" * (26 + w * len(group)))
        print(f"{'metric':<24}" + "".join(f"{l:>{w}}" for l in group))
        print("=" * (26 + w * len(group)))
        for l in group:
            pass
        print(f"{'kernel':<24}" + "".join(f"{s[l]['kernel'][:18]:>{w}}" for l in group))
        print(f"{'launches profiled':<24}" + "".join(f"{s[l]['launches']:>{w}}" for l in group))
        print(f"{'grid / block':<24}"
              + "".join(f"{s[l]['grid'] + ' / ' + s[l]['block']:>{w}}" for l in group))
        print("-" * (26 + w * len(group)))
        for name, col, scale, fmt in METRICS:
            cells = []
            for l in group:
                v = s[l]["metrics"].get(name)
                cells.append(fmt.format(v) if v is not None else "-")
            u = s[group[0]]["units"].get(name, "") or ""
            print(f"{name + (' (' + u + ')' if u else ''):<24}"
                  + "".join(f"{c:>{w}}" for c in cells))
        # top stalls, union of both columns ranked by the max across them
        allst = set()
        for l in group:
            allst |= set(s[l]["stalls"])
        ranked = sorted(allst, key=lambda k: -max(s[l]["stalls"].get(k, 0) for l in group))
        print("-" * (26 + w * len(group)))
        print(f"{'TOP WARP STALLS (cyc/inst)':<24}")
        for k in ranked[:8]:
            print(f"  {k:<22}"
                  + "".join(f"{s[l]['stalls'].get(k, 0):>{w}.2f}" for l in group))
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
