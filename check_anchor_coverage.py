"""
Verify each anchor in gemma_code_GB01_de.json (Alist/Blist) appears in
the corresponding side of gemma_code_GB01.json.

Reports per-anchor: EXACT (with region + offset) or FUZZY (with longest match)
or MISSING.
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
GB01 = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
DE = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01_de.json"


def find_in(haystack, needle, label):
    pos = haystack.find(needle)
    if pos >= 0:
        return ("EXACT", label, pos, len(needle))
    return None


def fuzzy_longest(haystack, needle, min_size=15):
    sm = difflib.SequenceMatcher(None, haystack, needle, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= min_size]
    if not blocks:
        return None
    best = max(blocks, key=lambda b: b.size)
    matched_total = sum(b.size for b in blocks)
    return {
        "haystack_pos": best.a,
        "longest": best.size,
        "matched_total": matched_total,
        "needle_len": len(needle),
        "matched_pct": matched_total / len(needle),
    }


def check_side(side_name, anchors, item):
    cot = item.get("cot", "")
    resp = item.get("response", "")
    full = cot + "\n<<RESPONSE>>\n" + resp
    print(f"\n=== {side_name}  ({len(anchors)} anchors) ===")
    summary = {"exact": 0, "fuzzy": 0, "missing": 0}
    for i, anchor in enumerate(anchors, 1):
        head = anchor.replace("\n", "\\n")[:80]
        # Try exact in cot, response
        hit = find_in(cot, anchor, "cot") or find_in(resp, anchor, "response")
        if hit:
            kind, region, pos, n = hit
            print(f"  [{i:>2}] EXACT  region={region:8s} pos={pos:>5d} len={n:>4d}  {head!r}")
            summary["exact"] += 1
            continue
        # Fuzzy on the full-side
        fz = fuzzy_longest(full, anchor)
        if fz and fz["matched_pct"] >= 0.5:
            print(f"  [{i:>2}] FUZZY  longest={fz['longest']}/{fz['needle_len']} "
                  f"matched_pct={fz['matched_pct']:.2f}  {head!r}")
            summary["fuzzy"] += 1
        else:
            extra = ""
            if fz:
                extra = f" (best longest={fz['longest']}, matched_pct={fz['matched_pct']:.2f})"
            print(f"  [{i:>2}] MISS{extra}  {head!r}")
            summary["missing"] += 1
    return summary


def main():
    data = json.loads(GB01.read_text(encoding="utf-8"))
    de = json.loads(DE.read_text(encoding="utf-8"))

    item_a = data["A"][0]
    item_b = data["B"][0]
    a_list = de["Alist"]
    b_list = de["Blist"]

    sa = check_side("Alist  vs  A", a_list, item_a)
    sb = check_side("Blist  vs  B", b_list, item_b)

    print("\n--- summary ---")
    print(f"  Alist:  exact={sa['exact']}  fuzzy={sa['fuzzy']}  missing={sa['missing']}")
    print(f"  Blist:  exact={sb['exact']}  fuzzy={sb['fuzzy']}  missing={sb['missing']}")
    total_anchors = len(a_list) + len(b_list)
    found = sa["exact"] + sa["fuzzy"] + sb["exact"] + sb["fuzzy"]
    print(f"  overall coverage: {found}/{total_anchors} ({100*found/total_anchors:.1f}%)")


if __name__ == "__main__":
    main()
