"""
Cross-QA neuron consistency analysis.

Reads all outputs/qa*/diff.pt and computes per-neuron statistics across QAs:
  - mean and std of |diff| across QAs
  - sign consistency (sign across QAs; close to ±1 = stable direction)
  - top-K appearances (how many QAs have this neuron in top-K of |diff|)

Reports neurons that are stable across all QAs vs those that fluctuate,
and breaks down "thinking-off" (model name has no Extended) vs "thinking-on".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
OUT_DIR = ROOT / "outputs"


def load_all():
    qa_dirs = sorted([d for d in OUT_DIR.iterdir() if d.is_dir() and d.name.startswith("qa") and (d / "diff.pt").exists()])
    runs = []
    for d in qa_dirs:
        data = torch.load(d / "diff.pt", weights_only=False)
        # tolerate older diff.pt (qa01 was saved without qa_id/model fields)
        qa_id = data.get("qa_id", d.name)
        model = data.get("model", "claude-opus4.6")  # default
        runs.append({
            "qa_id": qa_id,
            "model": model,
            "thinking": "Extended" in (model or ""),
            "diff": data["diff"],          # list of per-layer tensors
            "stats_a": data.get("stats_a"),
            "stats_b": data.get("stats_b"),
            "sizes": data["sizes"],
            "n_layers": data["n_layers"],
            "answer_span_len": data.get("answer_span_len"),
        })
    return runs


def stack_diffs(runs):
    """Returns per-layer stacked tensor of shape [n_runs, neurons_in_layer]."""
    n_layers = runs[0]["n_layers"]
    sizes = runs[0]["sizes"]
    out = []
    for li in range(n_layers):
        rows = [r["diff"][li] for r in runs]
        out.append(torch.stack(rows, dim=0))  # [n_runs, sz]
    return out


def per_neuron_stats(stacked):
    """For each (li,n): mean |diff|, std |diff|, sign_consistency."""
    abs_d_list = []
    sign_list = []
    for layer_t in stacked:  # [R, sz]
        abs_t = layer_t.abs()
        abs_d_list.append(abs_t.mean(dim=0))             # [sz]
        sign_list.append(layer_t.sign().mean(dim=0))     # [sz], in [-1,1]
    return abs_d_list, sign_list


def top_k_appearances(stacked, K=100):
    """For each (li,n): count of runs in which this neuron is in top-K of |diff|
    *across all neurons of all layers* (global top-K per run)."""
    n_runs = stacked[0].shape[0]
    n_layers = len(stacked)
    sizes = [t.shape[1] for t in stacked]
    counts = [torch.zeros(sz, dtype=torch.int32) for sz in sizes]

    for r in range(n_runs):
        # gather all |diff| values across layers for this run
        flat_vals = []
        flat_index = []  # (li, n)
        for li in range(n_layers):
            row = stacked[li][r].abs()
            flat_vals.append(row)
            flat_index.extend([(li, n) for n in range(sizes[li])])
        flat = torch.cat(flat_vals)
        # top-K indices in the flat tensor
        topk = torch.topk(flat, K).indices.tolist()
        for fi in topk:
            li, n = flat_index[fi]
            counts[li][n] += 1
    return counts


def render_top_neurons(label, ranking_pairs, mean_abs, sign_cons, runs, top=25):
    """ranking_pairs: list of (priority, li, n). Print details."""
    print(f"\n=== {label} (top {top}) ===")
    print(f"{'rank':>4} {'layer#nrn':>11} {'mean|d|':>8} {'sign':>6}  per-QA diffs")
    for k, (_, li, n) in enumerate(ranking_pairs[:top]):
        m = mean_abs[li][n].item()
        s = sign_cons[li][n].item()
        per_qa = []
        for r in runs:
            v = r["diff"][li][n].item()
            per_qa.append(f"{v:+.2f}")
        print(f"  {k+1:>2}  L{li:02d}#{n:<5d}  {m:.3f}  {s:+.2f}  [{', '.join(per_qa)}]")


def main():
    runs = load_all()
    print(f"[load] {len(runs)} QAs loaded")
    for r in runs:
        flag = "🧠" if r["thinking"] else "  "
        print(f"  {flag} {r['qa_id']}  model={r['model']}  span={r['answer_span_len']}")

    stacked = stack_diffs(runs)
    mean_abs, sign_cons = per_neuron_stats(stacked)

    # --- Ranking 1: by mean |diff| across all QAs ---
    pairs_mean = []
    for li, t in enumerate(mean_abs):
        for n in range(t.shape[0]):
            pairs_mean.append((t[n].item(), li, n))
    pairs_mean.sort(reverse=True)
    render_top_neurons("Top by mean |diff| across all QAs", pairs_mean, mean_abs, sign_cons, runs, top=30)

    # --- Ranking 2: top-K appearances (K=100) ---
    counts = top_k_appearances(stacked, K=100)
    pairs_app = []
    for li, t in enumerate(counts):
        for n in range(t.shape[0]):
            c = int(t[n])
            if c >= 1:
                pairs_app.append((c, li, n))
    pairs_app.sort(reverse=True)
    print(f"\n=== Top-100 |diff| appearances across {len(runs)} QAs ===")
    print(f"{'rank':>4} {'layer#nrn':>11} {'appearances':>12} {'mean|d|':>8} {'sign':>6}  per-QA diffs")
    for k, (c, li, n) in enumerate(pairs_app[:30]):
        m = mean_abs[li][n].item()
        s = sign_cons[li][n].item()
        per_qa = [f"{r['diff'][li][n].item():+.2f}" for r in runs]
        print(f"  {k+1:>2}  L{li:02d}#{n:<5d}  {c:>9}/{len(runs)}  {m:.3f}  {s:+.2f}  [{', '.join(per_qa)}]")

    # --- Ranking 3: L15+ persistent (within-QA active rate >=0.3) AND high mean across QAs ---
    L15_pairs = []
    for li in range(15, runs[0]["n_layers"]):
        for n in range(runs[0]["sizes"][li]):
            m = mean_abs[li][n].item()
            s = sign_cons[li][n].item()
            # require at least one QA where this neuron has >=0.3 active rate in either condition
            persistent = False
            for r in runs:
                if r["stats_a"] is None or r["stats_b"] is None:
                    persistent = True; break
                ra = r["stats_a"]["active_rate"][li][n].item()
                rb = r["stats_b"]["active_rate"][li][n].item()
                if max(ra, rb) >= 0.3:
                    persistent = True; break
            if persistent and m >= 0.15:
                L15_pairs.append((m * abs(s), li, n))  # priority weights consistency
    L15_pairs.sort(reverse=True)
    render_top_neurons("L15+ stable & persistent (priority = mean|d| × |sign|)",
                       L15_pairs, mean_abs, sign_cons, runs, top=30)

    # --- Group split: thinking-off vs thinking-on ---
    off_idx = [i for i, r in enumerate(runs) if not r["thinking"]]
    on_idx = [i for i, r in enumerate(runs) if r["thinking"]]
    if off_idx and on_idx:
        print(f"\n=== Thinking split: off={len(off_idx)} runs, on={len(on_idx)} runs ===")
        for li in range(runs[0]["n_layers"]):
            t = stacked[li]  # [R, sz]
            off_mean = t[off_idx].mean(dim=0).abs() if off_idx else None
            on_mean = t[on_idx].mean(dim=0).abs() if on_idx else None
            on_minus_off = (on_mean - off_mean) if (off_mean is not None and on_mean is not None) else None
            if on_minus_off is None:
                continue
            top3 = on_minus_off.topk(3)
            if li < 15:
                continue
            items = [f"#{int(top3.indices[k])}({on_minus_off[top3.indices[k]].item():+.2f})" for k in range(3)]
            print(f"  L{li:02d}  on-off top-3: {' '.join(items)}")

    # --- Save artifact for downstream clamp ---
    out = ROOT / "outputs" / "cross_qa_summary.pt"
    torch.save(
        {
            "runs": [{"qa_id": r["qa_id"], "model": r["model"], "thinking": r["thinking"]} for r in runs],
            "mean_abs": mean_abs,
            "sign_cons": sign_cons,
            "top_k_counts": counts,
            "stacked_diffs": stacked,
        },
        out,
    )
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
