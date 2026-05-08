"""
Cross-mode comparison: sweep all crutch experiments and produce a single
sweet-spot curve.

Usage:
  python crutch_compare.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")

# Maps mode label → raw.pt path
KNOWN_MODES = {
    "baseline (no SFT)": None,                               # placeholder
    "Mode A (multi_qa, no suppr)": ROOT / "outputs/opus47_multi_qa_intent/raw.pt",
    "Mode A_off (suppress 23)":   ROOT / "outputs/opus47_crutch_off/raw.pt",
    "Mode C (suppress 53)":       ROOT / "outputs/opus47_crutch_C/raw.pt",
    "Mode D (suppress 83)":       ROOT / "outputs/opus47_crutch_D/raw.pt",
    "Mode E (suppress 113)":      ROOT / "outputs/opus47_crutch_E/raw.pt",
    "Mode F (suppress 143)":      ROOT / "outputs/opus47_crutch_F/raw.pt",
    "Mode G (suppress 173)":      ROOT / "outputs/opus47_crutch_G/raw.pt",
}


def main():
    rows = []
    for label, path in KNOWN_MODES.items():
        if path is None or not path.exists():
            continue
        d = torch.load(path, weights_only=False)
        sup_dict = d.get("suppress_dict", {})
        n_sup = sum(len(v) for v in sup_dict.values())
        n_layers_with_sup = sum(1 for v in sup_dict.values() if v)

        per_qa = d.get("per_qa_loss_curves", {})
        if per_qa:
            avg_init = sum(lc[0][0] for lc in per_qa.values()) / len(per_qa)
            avg_final = sum(lc[-1][0] for lc in per_qa.values()) / len(per_qa)
        else:
            # older format (single-QA grad_probe) — derive from history
            hist = d.get("history", [])
            if hist:
                avg_init = hist[0].get("plain_loss",
                                       hist[0].get("loss", float("nan")))
                avg_final = hist[-1].get("plain_loss",
                                         hist[-1].get("loss", float("nan")))
            else:
                avg_init = avg_final = float("nan")

        cum_grad = d.get("cum_grad", {})
        total_grad = sum(cum_grad.values())

        cum_neuron = d.get("cum_neuron_down", {})
        layer_max = {}
        for L, t in cum_neuron.items():
            sup = sup_dict.get(L, set())
            if isinstance(sup, list):
                sup = set(sup)
            tt = t.clone()
            for s in sup:
                tt[s] = 0
            layer_max[L] = tt.max().item()

        rows.append({
            "label": label, "path": path,
            "n_sup": n_sup,
            "n_layers_with_sup": n_layers_with_sup,
            "avg_init": avg_init,
            "avg_final": avg_final,
            "delta": avg_init - avg_final,
            "total_grad": total_grad,
            "L27_max_post_sup": layer_max.get(27, float("nan")),
            "L34_max_post_sup": layer_max.get(34, float("nan")),
        })

    if not rows:
        print("No mode results found.")
        return

    print("=" * 90)
    print(f"{'mode':<35} {'n_sup':>7} {'init':>7} {'final':>7} {'Δ':>7} {'L27.max':>8} {'L34.max':>8}")
    print("=" * 90)
    for r in rows:
        print(f"{r['label']:<35} {r['n_sup']:>7} "
              f"{r['avg_init']:>7.3f} {r['avg_final']:>7.3f} "
              f"{r['delta']:>7.3f} {r['L27_max_post_sup']:>8.3f} "
              f"{r['L34_max_post_sup']:>8.3f}")
    print("=" * 90)

    # check for sweet spot signal: where does Δ start dropping?
    if len(rows) >= 3:
        print("\n[sweet-spot check] looking for the round where Δloss starts shrinking")
        for i in range(1, len(rows)):
            prev = rows[i-1]
            curr = rows[i]
            change = curr["delta"] - prev["delta"]
            note = ""
            if change > 0.05:
                note = "  ↑ still gaining"
            elif change < -0.1:
                note = "  ↓ DROPPED — possible sweet spot exhausted"
            else:
                note = "  ≈ similar"
            print(f"  {prev['label']:<30} → {curr['label']:<30}  "
                  f"Δchange={change:+.3f}{note}")


if __name__ == "__main__":
    main()
