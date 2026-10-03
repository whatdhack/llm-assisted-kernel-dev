"""Top-N SASS instruction mix from an ncu source page.

Aggregates the source page's "Instructions Executed" column by SASS opcode.
Reads the saved .source.txt files, so it needs no GPU.

Launches are delimited by "Kernel Name" lines and parsed separately: a
report can hold structurally different launches (kloop's SYRK capture spans
one rank-1024 phase-2 update and two narrow phase-1 strip updates), and
summing those would describe no kernel that actually exists. Default is
launch 0; pass --launch N or --launch all.

Usage:
    python ncu_top_instructions.py [--top 10] [--launch 0] label=a.source.txt ...
"""
import collections
import sys


def columns(lines):
    """Column spans come from the dash ruler; the source page is fixed-width."""
    ruler = next(l for l in lines if set(l.strip()) <= set("- ") and "---" in l)
    spans, pos = [], 0
    for grp in ruler.split(" "):
        if grp:
            spans.append((pos, pos + len(grp)))
        pos += len(grp) + 1
    hdr = next(l for l in lines if l.startswith("Address"))
    names = [hdr[a:b].strip() for a, b in spans]
    return spans, names


def split_launches(lines):
    starts = [i for i, l in enumerate(lines) if l.startswith("Kernel Name")]
    return [lines[s:e] for s, e in zip(starts, starts[1:] + [len(lines)])]


def opcode_mix(path, launch):
    lines = open(path, errors="replace").read().splitlines()
    spans, names = columns(lines)
    ic = next(i for i, n in enumerate(names) if n.startswith("Instru"))
    sc = next(i for i, n in enumerate(names) if n.startswith("Source"))

    chunks = split_launches(lines)
    chunks = chunks if launch == "all" else [chunks[int(launch)]]

    agg = collections.Counter()
    for chunk in chunks:
        for l in chunk:
            if len(l) < spans[ic][1] or l.startswith(("Address", "---", "Kernel")):
                continue
            v = l[spans[ic][0]:spans[ic][1]].strip().replace(",", "")
            s = l[spans[sc][0]:spans[sc][1]].strip()
            if not (v.isdigit() and s):
                continue
            tok = s.split()[0]
            if not tok or not tok[0].isalpha():
                continue
            agg[tok.split(".")[0]] += int(v)   # base mnemonic, variants merged
    return agg


def main(argv):
    top, launch, pairs = 10, "0", []
    i = 0
    while i < len(argv):
        if argv[i] == "--top":
            top = int(argv[i + 1]); i += 2
        elif argv[i] == "--launch":
            launch = argv[i + 1]; i += 2
        else:
            pairs.append(argv[i].split("=", 1)); i += 1

    mixes = {label: opcode_mix(path, launch) for label, path in pairs}
    labels = list(mixes)

    # union of each kernel's top-N, ranked by the largest share anywhere
    keep = set()
    for m in mixes.values():
        keep |= {k for k, _ in m.most_common(top)}
    share = lambda lab, k: 100.0 * mixes[lab][k] / max(sum(mixes[lab].values()), 1)
    ranked = sorted(keep, key=lambda k: -max(share(l, k) for l in labels))

    w = 22
    print(f"{'opcode':<12}" + "".join(f"{l:>{w}}" for l in labels))
    print("-" * (12 + w * len(labels)))
    for k in ranked:
        cells = [f"{mixes[l][k]:>12,} {share(l, k):5.1f}%" for l in labels]
        print(f"{k:<12}" + "".join(f"{c:>{w}}" for c in cells))
    print("-" * (12 + w * len(labels)))
    tot = [f"{sum(mixes[l].values()):>12,} {100.0:5.1f}%" for l in labels]
    print(f"{'TOTAL':<12}" + "".join(f"{t:>{w}}" for t in tot))
    print(f"{'distinct':<12}" + "".join(f"{len(mixes[l]):>{w},}" for l in labels))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
