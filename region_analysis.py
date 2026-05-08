"""
Cross-QA region-aware neuron analysis.

Reads outputs/qa*/diff_regions.pt and identifies neurons whose activation
gap (Opus mean - Gemma mean) is concentrated in specific output regions.

Key outputs:
  1. Per-region cross-QA top neurons  (mean|d|, sign consistency)
  2. Specificity scores:
       d3_specific  = mean|d|_d3 - mean|d|_glue
       d3_vs_d1     = mean|d|_d3 - mean|d|_d1   (depth axis)
  3. Sign-split rankings (Opus-enhances vs Opus-suppresses) per region
  4. Comparison to the prior global cross-QA top-6 (does its signal localize
     to one region?)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
OUT_DIR = ROOT / "outputs"

REGION_NAMES = ["unlabeled", "glue", "d1_surface", "d2_structural", "d3_thematic"]
REGIONS_OF_INTEREST = [1, 2, 3, 4]  # skip unlabeled
REGION_SHORT = {1: "glue", 2: "d1", 3: "d2", 4: "d3"}

# the prior global cross-QA top-6 from the previous run, for localization check
PRIOR_TOP6 = [
    (27, 10024, "L27#10024"),
    (28, 7686,  "L28#7686"),
    (32, 9383,  "L32#9383"),
    (26, 5430,  "L26#5430"),
    (32, 8474,  "L32#8474"),
    (27, 8968,  "L27#8968"),
]


def load_runs():
    qa_dirs = sorted(d for d in OUT_DIR.iterdir()
                     if d.is_dir() and d.name.startswith("qa")
                     and (d / "diff_regions.pt").exists())
    runs = []
    for d in qa_dirs:
        data = torch.load(d / "diff_regions.pt", weights_only=False)
        model = data.get("model", "?") or "?"
        runs.append({
            "qa_id": data["qa_id"],
            "model": model,
            "thinking": "Extended" in model,
            "diff_per_region": data["diff_per_region"],  # [layer][region] -> tensor or None
            "region_counts": data["region_token_counts"],
            "n_layers": data["n_layers"],
            "sizes": data["sizes"],
        })
    return runs


def aggregate_per_region(runs, region_id):
    """For a fixed region, build:
      mean_abs[layer]: tensor [neurons]   (mean of |diff| across QAs that have data)
      sign_cons[layer]: tensor [neurons]  (mean of sign across QAs)
      n_qas[layer]: tensor [neurons]      (count of QAs with data; same per layer)
    Skips QAs whose region had 0 tokens.
    """
    n_layers = runs[0]["n_layers"]
    sizes = runs[0]["sizes"]
    mean_abs = [torch.zeros(s, dtype=torch.float32) for s in sizes]
    sign_sum = [torch.zeros(s, dtype=torch.float32) for s in sizes]
    abs_per_qa = [[] for _ in range(n_layers)]
    qa_count = [0 for _ in range(n_layers)]

    for r in runs:
        if r["region_counts"][region_id] == 0:
            continue
        for li in range(n_layers):
            d = r["diff_per_region"][li][region_id]
            if d is None:
                continue
            abs_per_qa[li].append(d.abs())
            sign_sum[li] += d.sign()
            qa_count[li] += 1
        # qa_count is identical across layers in practice; treat as global
    n_qas_global = max(qa_count) if qa_count else 0
    for li in range(n_layers):
        if abs_per_qa[li]:
            stacked = torch.stack(abs_per_qa[li], dim=0)
            mean_abs[li] = stacked.mean(dim=0)
            sign_cons_li = sign_sum[li] / max(1, qa_count[li])
        else:
            sign_cons_li = sign_sum[li]
        sign_sum[li] = sign_cons_li  # in-place repurpose
    return mean_abs, sign_sum, n_qas_global


def render_region_top(label, mean_abs, sign_cons, runs, region_id, top=20, l_lo=15):
    print(f"\n=== {label} (top {top}, L>={l_lo}) ===")
    pairs = []
    for li in range(l_lo, len(mean_abs)):
        t = mean_abs[li]
        for n in range(t.shape[0]):
            pairs.append((t[n].item(), li, n))
    pairs.sort(reverse=True)
    print(f"{'rank':>4} {'layer#nrn':>11} {'mean|d|':>8} {'sign':>6}  per-QA diffs (region)")
    for k, (m, li, n) in enumerate(pairs[:top]):
        s = sign_cons[li][n].item()
        per_qa = []
        for r in runs:
            if r["region_counts"][region_id] == 0:
                per_qa.append("  -  ")
            else:
                d = r["diff_per_region"][li][region_id]
                per_qa.append(f"{d[n].item():+.2f}")
        print(f"  {k+1:>2}  L{li:02d}#{n:<5d}  {m:.3f}  {s:+.2f}  [{', '.join(per_qa)}]")


def render_specificity(name, score, mean_abs_d3, sign_cons_d3, runs, top=20, l_lo=15):
    """Rank by `score[layer][neuron]` (a tensor list) descending; print mean|d|_d3 + sign for context."""
    print(f"\n=== {name} (top {top}, L>={l_lo}) ===")
    pairs = []
    for li in range(l_lo, len(score)):
        t = score[li]
        for n in range(t.shape[0]):
            pairs.append((t[n].item(), li, n))
    pairs.sort(reverse=True)
    print(f"{'rank':>4} {'layer#nrn':>11} {'score':>8} {'|d|_d3':>8} {'sign_d3':>8}")
    for k, (sc, li, n) in enumerate(pairs[:top]):
        m3 = mean_abs_d3[li][n].item()
        s3 = sign_cons_d3[li][n].item()
        print(f"  {k+1:>2}  L{li:02d}#{n:<5d}  {sc:+.3f}  {m3:.3f}   {s3:+.2f}")


def main():
    runs = load_runs()
    print(f"[load] {len(runs)} QAs loaded")
    for r in runs:
        flag = "🧠" if r["thinking"] else "  "
        print(f"  {flag} {r['qa_id']}  model={r['model']}  region_counts={r['region_counts']}")

    # --- Aggregate per region ---
    region_data = {}
    for rid in REGIONS_OF_INTEREST:
        ma, sc, n_qas = aggregate_per_region(runs, rid)
        region_data[rid] = {"mean_abs": ma, "sign_cons": sc, "n_qas": n_qas}
        print(f"\n[agg] region={REGION_SHORT[rid]} usable QAs: {n_qas}")

    # --- Per-region top neurons ---
    for rid in REGIONS_OF_INTEREST:
        rd = region_data[rid]
        render_region_top(
            f"Top neurons in region '{REGION_SHORT[rid]}'",
            rd["mean_abs"], rd["sign_cons"], runs, rid, top=15,
        )

    # --- Specificity: d3 vs glue (functional vs filler) ---
    # score[layer][neuron] = mean|d|_d3 - mean|d|_glue
    d3 = region_data[4]
    glue = region_data[1]
    d2 = region_data[3]
    d1 = region_data[2]

    n_layers = runs[0]["n_layers"]
    sizes = runs[0]["sizes"]

    spec_d3_vs_glue = [d3["mean_abs"][li] - glue["mean_abs"][li] for li in range(n_layers)]
    spec_d3_vs_d1   = [d3["mean_abs"][li] - d1["mean_abs"][li]   for li in range(n_layers)]
    spec_d2_vs_glue = [d2["mean_abs"][li] - glue["mean_abs"][li] for li in range(n_layers)]
    spec_d3_vs_d2   = [d3["mean_abs"][li] - d2["mean_abs"][li]   for li in range(n_layers)]

    render_specificity("Conclusion-specific: d3 - glue (CORE THEMATIC vs FILLER)",
                       spec_d3_vs_glue, d3["mean_abs"], d3["sign_cons"], runs, top=20)
    render_specificity("Depth gradient: d3 - d1 (THEMATIC vs SURFACE)",
                       spec_d3_vs_d1, d3["mean_abs"], d3["sign_cons"], runs, top=20)
    render_specificity("Conclusion-specific: d2 - glue (STRUCTURAL vs FILLER)",
                       spec_d2_vs_glue, d2["mean_abs"], d2["sign_cons"], runs, top=15)
    render_specificity("Depth gap: d3 - d2 (CORE vs STRUCTURAL)",
                       spec_d3_vs_d2, d3["mean_abs"], d3["sign_cons"], runs, top=15)

    # --- Sign-split per region: Opus-enhances (positive) vs Opus-suppresses (negative) ---
    print("\n=== Sign-split rankings per region (L>=15, |sign|>=0.8) ===")
    for rid in REGIONS_OF_INTEREST:
        rd = region_data[rid]
        pos_pairs, neg_pairs = [], []
        for li in range(15, n_layers):
            ma_t = rd["mean_abs"][li]
            sc_t = rd["sign_cons"][li]
            for n in range(ma_t.shape[0]):
                if abs(sc_t[n].item()) < 0.8:
                    continue
                if sc_t[n].item() > 0:
                    pos_pairs.append((ma_t[n].item(), li, n))
                else:
                    neg_pairs.append((ma_t[n].item(), li, n))
        pos_pairs.sort(reverse=True); neg_pairs.sort(reverse=True)
        print(f"\n  -- region {REGION_SHORT[rid]} --")
        print(f"     POS (Opus enhances): " +
              " ".join(f"L{li}#{n}({m:.2f})" for m, li, n in pos_pairs[:8]))
        print(f"     NEG (Opus suppresses): " +
              " ".join(f"L{li}#{n}({m:.2f})" for m, li, n in neg_pairs[:8]))

    # --- Localization of the prior global top-6 ---
    print("\n=== Prior global top-6: where does each one's signal live? ===")
    print(f"{'neuron':>11} " + " ".join(f"{REGION_SHORT[r]:>10}" for r in REGIONS_OF_INTEREST))
    for li, n, label in PRIOR_TOP6:
        cells = []
        for rid in REGIONS_OF_INTEREST:
            ma = region_data[rid]["mean_abs"][li][n].item()
            sc = region_data[rid]["sign_cons"][li][n].item()
            cells.append(f"{ma:+.2f}|{sc:+.1f}")
        print(f"  {label:>11} " + " ".join(f"{c:>10}" for c in cells))
    print("  (cells: mean|d|_region | sign_consistency_region)")

    # --- Save artifact ---
    out = OUT_DIR / "region_summary.pt"
    torch.save(
        {
            "region_data": region_data,
            "spec_d3_vs_glue": spec_d3_vs_glue,
            "spec_d3_vs_d1":   spec_d3_vs_d1,
            "spec_d2_vs_glue": spec_d2_vs_glue,
            "spec_d3_vs_d2":   spec_d3_vs_d2,
            "runs": [{"qa_id": r["qa_id"], "model": r["model"],
                      "thinking": r["thinking"],
                      "region_counts": r["region_counts"]} for r in runs],
        },
        out,
    )
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
