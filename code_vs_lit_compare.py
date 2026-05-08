"""
Compare code-domain neurons (outputs/code_qa01/diff.pt) against the existing
literature-domain known neurons. Produces three lists:
  1) overlap top neurons in both domains  -> likely *general* functional axes
  2) code-domain top with lit-domain low   -> code/frontend-specific candidates
  3) status of every currently-known neuron under the code QA
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
CODE_DIFF = ROOT / "outputs" / "code_qa01" / "diff.pt"
LIT_DIR = ROOT / "outputs"

KNOWN = [
    (27, 10024, "L27#10024 d1_detail_neg"),
    (27, 6644,  "L27#6644  d3_depth_pos"),
    (32, 9383,  "L32#9383  d2_structure_neg"),
    (26, 5430,  "L26#5430  global_tone_neg"),
    (32, 8474,  "L32#8474  d1_detail_pos"),
    (27, 8968,  "L27#8968  broad_conclusion_pos"),
    (34, 8522,  "L34#8522  d2_marker_pos"),
    (31, 604,   "L31#604   d2_structure_neg"),
    (15, 8146,  "L15#8146  glue_fingerprint_pos"),
    (28, 7686,  "L28#7686  d1_biased_neg"),
    (27, 5590,  "L27#5590  d3_depth_neg"),
    (27, 2890,  "L27#2890  d3_unstable_neg"),
    (34, 6608,  "L34#6608  d3_late_suppress_neg"),
    (34, 1614,  "L34#1614  d3_late_suppress_neg"),
    (23, 287,   "L23#287   d3_mid_suppress_neg"),
]


def load_lit_mean_abs(n_layers: int, sizes: list[int]):
    """Mean |diff| across the 5 literature QAs (qa01..qa05/diff.pt).
    Falls back gracefully if some are missing."""
    qa_dirs = sorted(d for d in LIT_DIR.iterdir()
                     if d.is_dir() and d.name.startswith("qa")
                     and (d / "diff.pt").exists())
    if not qa_dirs:
        return None, []
    stacks = [[] for _ in range(n_layers)]
    used = []
    for d in qa_dirs:
        data = torch.load(d / "diff.pt", weights_only=False)
        if data["n_layers"] != n_layers:
            continue
        diff = data["diff"]
        for li in range(n_layers):
            stacks[li].append(diff[li].abs())
        used.append(d.name)
    mean_abs = [torch.stack(s, dim=0).mean(dim=0) if s else torch.zeros(sz)
                for s, sz in zip(stacks, sizes)]
    return mean_abs, used


def main():
    if not CODE_DIFF.exists():
        print(f"[err] missing {CODE_DIFF}; run probe_code.py first")
        return

    code = torch.load(CODE_DIFF, weights_only=False)
    n_layers = code["n_layers"]
    sizes = code["sizes"]
    code_diff = code["diff"]
    code_abs = [d.abs() for d in code_diff]
    code_rate_a = code["stats_a"]["active_rate"]
    code_rate_b = code["stats_b"]["active_rate"]

    lit_abs, lit_runs = load_lit_mean_abs(n_layers, sizes)
    print(f"[load] code QA: code_qa01 (ref={code.get('ref_model','?')})")
    print(f"[load] lit QAs aggregated: {lit_runs}")

    # rank code-domain top
    code_pairs = []
    for li in range(n_layers):
        for n in range(sizes[li]):
            code_pairs.append((code_abs[li][n].item(), li, n, code_diff[li][n].item()))
    code_pairs.sort(reverse=True)

    print("\n=== CODE-domain global top-30 |diff| ===")
    print(f"{'rank':>4} {'layer#nrn':>11} {'code|d|':>8} {'code d':>8} {'lit|d|':>7}")
    for k in range(30):
        _, li, n, v = code_pairs[k]
        ld = lit_abs[li][n].item() if lit_abs is not None else float("nan")
        print(f"  {k+1:>2}  L{li:02d}#{n:<5d}  {abs(v):.3f}  {v:+.3f}  {ld:.3f}")

    if lit_abs is None:
        print("\n[warn] no literature diffs found; skipping overlap/specificity")
        return

    # Overlap: high in BOTH (geometric mean style score)
    overlap = []
    code_specific = []
    for li in range(15, n_layers):
        cd = code_abs[li]
        ld = lit_abs[li]
        for n in range(cd.shape[0]):
            ca = cd[n].item()
            la = ld[n].item()
            if ca < 0.2:
                continue
            ra = code_rate_a[li][n].item()
            rb = code_rate_b[li][n].item()
            if max(ra, rb) < 0.3:
                continue
            overlap_score = (ca * la) ** 0.5
            spec_score = ca - la
            overlap.append((overlap_score, li, n, ca, la, code_diff[li][n].item()))
            code_specific.append((spec_score, li, n, ca, la, code_diff[li][n].item()))

    overlap.sort(reverse=True)
    code_specific.sort(reverse=True)

    print("\n=== Likely GENERAL functional (high in code AND lit, L>=15) ===")
    print(f"{'rank':>4} {'layer#nrn':>11} {'overlap':>8} {'code|d|':>8} {'lit|d|':>7} {'code d':>8}")
    for k, (sc, li, n, ca, la, v) in enumerate(overlap[:20]):
        print(f"  {k+1:>2}  L{li:02d}#{n:<5d}  {sc:.3f}  {ca:.3f}  {la:.3f}  {v:+.3f}")

    print("\n=== CODE-SPECIFIC candidates (code high, lit low, L>=15) ===")
    print(f"{'rank':>4} {'layer#nrn':>11} {'spec':>7} {'code|d|':>8} {'lit|d|':>7} {'code d':>8}")
    for k, (sc, li, n, ca, la, v) in enumerate(code_specific[:25]):
        print(f"  {k+1:>2}  L{li:02d}#{n:<5d}  {sc:+.3f}  {ca:.3f}  {la:.3f}  {v:+.3f}")

    # Status of currently-known lit neurons under the code QA
    print("\n=== Currently-known LIT neurons under the CODE QA ===")
    print(f"{'neuron':<35} {'code|d|':>8} {'code d':>8} {'lit|d|':>7} {'rate_A':>7} {'rate_B':>7}")
    for li, n, label in KNOWN:
        ca = code_abs[li][n].item()
        cv = code_diff[li][n].item()
        la = lit_abs[li][n].item() if lit_abs is not None else float("nan")
        ra = code_rate_a[li][n].item()
        rb = code_rate_b[li][n].item()
        print(f"  {label:<35} {ca:.3f}  {cv:+.3f}  {la:.3f}  {ra:.3f}  {rb:.3f}")


if __name__ == "__main__":
    main()
