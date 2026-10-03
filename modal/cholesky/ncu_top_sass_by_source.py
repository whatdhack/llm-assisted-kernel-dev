"""Top-N SASS instructions by TIME, with the Python line each came from.

Reads a saved __source_pycorr.txt (ncu --page source --print-source cuda,sass),
so it needs no GPU. That file interleaves each Python source line with the SASS
it generated, which is what lets a hot instruction be named in Python terms.

Ranks by "Warp Stall Sampling (All Samples)" -- ncu's PC-sampling estimate of
where warp cycles are spent. That is a TIME proxy, unlike ncu_top_instructions.py
which ranks by "Instructions Executed" (a COUNT). The two answer different
questions and routinely disagree: a cheap instruction executed a million times
tops the count list, a single barrier tops the time list.

Launches are parsed separately (see ncu_top_instructions.py for why summing
structurally different launches describes no kernel that exists).

Usage:
    python ncu_top_sass_by_source.py [--top 10] [--launch 0] <file>__source_pycorr.txt
"""
import collections
import os
import re
import sys


def spans_of(lines):
    ruler = next(l for l in lines if set(l.strip()) <= set("- ") and "---" in l)
    out, pos = [], 0
    for grp in ruler.split(" "):
        if grp:
            out.append((pos, pos + len(grp)))
        pos += len(grp) + 1
    return out


def metrics_by_addr(path, launch):
    """addr -> (stall samples, sass text), read from the sass-only export where
    the fixed-width columns are intact."""
    lines = open(path, errors="replace").read().splitlines()
    sp = spans_of(lines)
    (a0, a1), (s0, s1), (w0, w1) = sp[0], sp[1], sp[2]
    starts = [i for i, l in enumerate(lines) if l.startswith("Kernel Name")]
    chunks = [lines[s:e] for s, e in zip(starts, starts[1:] + [len(lines)])]
    if launch != "all":
        chunks = [chunks[int(launch)]]
    out = {}
    for chunk in chunks:
        for l in chunk:
            if len(l) < w1 or l.startswith(("Address", "Kernel", "---")):
                continue
            a, t, w = l[a0:a1].strip(), l[s0:s1].strip(), l[w0:w1].strip().replace(",", "")
            if a.startswith("0x") and t and w.isdigit():
                out[a] = (int(w), t)
    return out


def parse(path, launch, metrics=None):
    lines = open(path, errors="replace").read().splitlines()
    sp = spans_of(lines)
    a0, a1 = sp[0]           # Address / python line number
    s0, s1 = sp[1]           # Source text
    w0, w1 = sp[2]           # Warp Stall Sampling (All Samples)

    # the cuda,sass view delimits launches with a "File Path:" banner rather
    # than the "Kernel Name" header the sass-only page uses
    starts = [i for i, l in enumerate(lines)
              if l.startswith("File Path:") or l.startswith("Kernel Name")]
    chunks = [lines[s:e] for s, e in zip(starts, starts[1:] + [len(lines)])]
    if launch != "all":
        chunks = [chunks[int(launch)]]

    rows, per_line, py_txt = [], collections.Counter(), {}
    for chunk in chunks:
        cur_line, cur_src = None, ""
        for l in chunk:
            if len(l) < a1 or l.startswith(("Address", "Kernel", "---", "/*",
                                            "File Path", "Function Name")):
                continue
            addr = l[a0:a1].strip()
            # A source row whose text is longer than the Source column is
            # WRAPPED: the line number sits on the first physical line, which
            # is then too short to reach the metric columns, and the metrics
            # land on the continuation line. Requiring full width here would
            # drop the line number and silently charge that line's SASS to
            # whichever source row came before it.
            if addr.isdigit():
                cur_line = int(addr)
                cur_src = l[s0:s1].strip() if len(l) > s0 else ""
                py_txt[cur_line] = cur_src
                continue
            if len(l) < w1:
                continue
            src = l[s0:s1].strip()
            w = l[w0:w1].strip().replace(",", "")
            n = int(w) if w.isdigit() else 0     # truncated row: metric comes
            if not (w.isdigit() or addr.startswith("0x")):   # from the sibling
                continue
            if addr.startswith("0x") and src:        # a SASS row
                n, src = metrics.get(addr, (n, src)) if metrics else (n, src)
                rows.append((n, addr, src, cur_line, cur_src))
                if cur_line is not None:
                    per_line[cur_line] += n
    return rows, per_line, py_txt


def main():
    args = [a for a in sys.argv[1:]]
    top = int(args.pop(args.index("--top") + 1)) if "--top" in args else 10
    if "--top" in args:
        args.remove("--top")
    launch = args.pop(args.index("--launch") + 1) if "--launch" in args else "0"
    if "--launch" in args:
        args.remove("--launch")
    path = args[0]

    sib = path.replace("__source_pycorr.txt", ".source.txt")
    metrics = None
    if os.path.exists(sib):
        metrics = metrics_by_addr(sib, launch)
    else:
        print(f"WARNING: {sib} not found -- metrics taken from the correlated\n"
              f"         view, which drops instructions with long operand lists.\n")
    rows, per_line, py_txt = parse(path, launch, metrics)
    total = sum(r[0] for r in rows) or 1
    print(f"{path.split('/')[-1]}   launch {launch}   "
          f"{len(rows)} SASS sites, {total} stall samples\n")

    print(f"top {top} SASS instructions by warp-stall samples (time proxy)")
    print(f"{'samp':>7} {'%':>6}  {'address':<16} {'SASS':<44} python")
    print("-" * 118)
    for n, addr, src, line, ptxt in sorted(rows, reverse=True)[:top]:
        loc = f"{line}: {ptxt[:38]}" if line else "-"
        print(f"{n:7} {100*n/total:5.1f}%  {addr:<16} {src[:44]:<44} {loc}")

    print(f"\ntop {top} python lines (same samples, rolled up per line)")
    print(f"{'samp':>7} {'%':>6}  {'line':>5}  source")
    print("-" * 90)
    for line, n in per_line.most_common(top):
        print(f"{n:7} {100*n/total:5.1f}%  {line:>5}  {py_txt.get(line,'')[:66]}")

    print(f"\ntop {top} by opcode (same samples, merged across sites)")
    agg = collections.Counter()
    for n, _, src, _, _ in rows:
        toks = [t for t in src.split() if not t.startswith("@")]  # drop predicates
        if toks:
            agg[toks[0].split(".")[0]] += n
    print(f"{'samp':>7} {'%':>6}  opcode")
    print("-" * 34)
    for op, n in agg.most_common(top):
        print(f"{n:7} {100*n/total:5.1f}%  {op}")


if __name__ == "__main__":
    main()
