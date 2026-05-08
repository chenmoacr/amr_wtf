"""
Focused reader for multi_qa_intent_sft results, with side-by-side
comparison to single-QA intent_sft and grad_probe.
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
MULTI_PT  = ROOT / "outputs" / "opus47_multi_qa_intent" / "raw.pt"
INTENT_PT = ROOT / "outputs" / "opus47_intent_sft" / "raw.pt"
PROBE_PT  = ROOT / "outputs" / "opus47_grad_probe" / "raw.pt"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def module_totals(cum_grad):
    out = {m: 0.0 for m in TARGET_MODULES}
    for (L, m), v in cum_grad.items():
        out[m] += v
    return out


def layer_totals(cum_grad, n_layers=35):
    return [(L, sum(cum_grad.get((L, m), 0.0) for m in TARGET_MODULES))
            for L in range(n_layers)]


def main():
    data = torch.load(MULTI_PT, weights_only=False)
    cfg = data["config"]
    qas = data["qas_meta"]
    history = data["history"]
    per_qa = data["per_qa_loss_curves"]
    cum_grad = data["cum_grad"]
    cum_neuron = data["cum_neuron_down"]

    print("=" * 70)
    print("  Multi-QA Intent SFT — focused diagnostic")
    print("=" * 70)
    print(f"  weights: concl={cfg['W_CONCL']}  filler={cfg['W_FILLER']}  "
          f"glue={cfg['W_GLUE']}")
    print(f"  epochs={cfg['N_EPOCHS']}  total_steps={cfg['N_EPOCHS']*5}  "
          f"LR={cfg['LR']}  LoRA r={cfg['LORA_R']}")

    # ---- per-QA bucket sizes ----
    print(f"\n[per-QA buckets]")
    print(f"  qa  | n_target | glue | concl | filler | concl%")
    for qa in qas:
        n = qa["n_target"]
        gl, cn, fl = qa["n_glue"], qa["n_concl"], qa["n_filler"]
        print(f"  QA0{qa['qa_id']} |   {n:>4}   | {gl:>4} | {cn:>5} | {fl:>5}  | "
              f"{cn/n*100:5.1f}%")

    # ---- per-QA loss convergence ----
    print(f"\n[per-QA convergence] (plain CE mean)")
    print(f"  qa   | epoch1 | epoch2 | epoch3 | epoch4 |  Δ")
    for qid, lc in per_qa.items():
        plains = [x[0] for x in lc]
        d = plains[0] - plains[-1] if len(plains) >= 2 else 0
        cells = "  ".join(f"{p:.3f}" for p in plains)
        print(f"  QA0{qid}  | {cells}  | {d:+.3f}")

    # ---- cross-QA reduction patterns ----
    # is each QA's loss reducing similarly? if so, model is learning shared patterns
    avg_init = sum(lc[0][0] for lc in per_qa.values()) / 5
    avg_final = sum(lc[-1][0] for lc in per_qa.values()) / 5
    print(f"\n  Average across 5 QAs: init={avg_init:.3f}  final={avg_final:.3f}  "
          f"Δ={avg_init-avg_final:+.3f}")

    # ---- compare with single-QA runs ----
    print(f"\n[vs single-QA runs]")
    if INTENT_PT.exists():
        sing = torch.load(INTENT_PT, weights_only=False)
        s_init = sing["history"][0]["plain_loss"]
        s_final = sing["history"][-1]["plain_loss"]
        print(f"  single-QA intent_sft : init={s_init:.3f}  final={s_final:.3f}  "
              f"Δ={s_init-s_final:+.3f}")
    if PROBE_PT.exists():
        prb = torch.load(PROBE_PT, weights_only=False)
        p_init = prb["history"][0]["loss"]
        p_final = prb["history"][-1]["loss"]
        print(f"  single-QA grad_probe : init={p_init:.3f}  final={p_final:.3f}  "
              f"Δ={p_init-p_final:+.3f}")
    print(f"  multi-QA (5)          : init={avg_init:.3f}  final={avg_final:.3f}  "
          f"Δ={avg_init-avg_final:+.3f}")

    # ---- module totals comparison ----
    print(f"\n[cum-grad module totals — 3-way]")
    multi_m = module_totals(cum_grad)
    sing_m = module_totals(sing["cum_grad"]) if INTENT_PT.exists() else None
    prb_m = module_totals(prb["cum_grad"]) if PROBE_PT.exists() else None
    print(f"  module     | multi(5)  | single_intent | probe")
    for m in TARGET_MODULES:
        v_multi = multi_m[m]
        v_sing = sing_m[m] if sing_m else 0
        v_prb = prb_m[m] if prb_m else 0
        print(f"  {m:<10} | {v_multi:>9.2f} | {v_sing:>13.2f} | {v_prb:>9.2f}")

    # ---- layer totals top-10 ----
    multi_layer = sorted(layer_totals(cum_grad), key=lambda kv: kv[1], reverse=True)
    print(f"\n[top-15 layers in multi-QA]")
    if INTENT_PT.exists():
        sing_l = dict(layer_totals(sing["cum_grad"]))
    else:
        sing_l = {}
    print(f"  rank | layer | multi.total | single.total | ratio (m/s)")
    for rk, (L, v) in enumerate(multi_layer[:15], 1):
        s = sing_l.get(L, 0)
        ratio = v / max(s, 1e-9)
        print(f"  {rk:>3}  |  L{L:>2}  |  {v:>9.2f}  |  {s:>9.2f}  |  {ratio:.2f}×")

    # ---- per-neuron max comparison ----
    print(f"\n[per-neuron max — selected layers]")
    print(f"  layer | multi.max | single.max | probe.max")
    sing_n = sing["cum_neuron_down"] if INTENT_PT.exists() else {}
    prb_n = prb["cum_neuron_down"] if PROBE_PT.exists() else {}
    for L in [11, 12, 13, 24, 26, 27, 28, 31, 32, 33, 34]:
        if L not in cum_neuron:
            continue
        m_max = cum_neuron[L].max().item()
        s_max = sing_n[L].max().item() if L in sing_n else 0
        p_max = prb_n[L].max().item() if L in prb_n else 0
        print(f"  L{L:>2}   |  {m_max:>8.4f} |  {s_max:>9.4f} |  {p_max:>9.4f}")

    # ---- top neurons in selected layers (with cross-method agreement) ----
    NEURONS_JSON = ROOT / "chat" / "neurons.json"
    inventory = {}
    if NEURONS_JSON.exists():
        import json
        nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
        inventory = {(n["layer"], n["index"]): n for n in nj["known_neurons"]}

    for L in [27, 34]:
        if L not in cum_neuron:
            continue
        t = cum_neuron[L]
        topk = t.topk(15)
        print(f"\n[L{L} top-15 in multi-QA — vs single-QA rank]")
        # build single-QA ranking for cross-reference
        if L in sing_n:
            sing_rank = (-sing_n[L]).argsort().tolist()
            sing_rank_pos = {n: r+1 for r, n in enumerate(sing_rank[:300])}
        else:
            sing_rank_pos = {}
        for i, (val, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
            tag = ""
            key = (L, idx)
            if key in inventory:
                inv = inventory[key]
                tag = (f"  [INV {inv.get('tier','?')[:8]}, "
                       f"gain={inv.get('default_gain',0):+.1f}]")
            sing_r = sing_rank_pos.get(idx, "—")
            sing_r_str = f"sing#{sing_r}" if isinstance(sing_r, int) else "sing#—"
            print(f"  {i+1:>2}. L{L}#{idx:<5}  cum={val:.4f}  ({sing_r_str}){tag}")

    # ---- inventory match summary ----
    total_matches = 0
    per_tier = defaultdict(int)
    by_layer = defaultdict(int)
    for L in range(35):
        if L not in cum_neuron:
            continue
        t = cum_neuron[L]
        top_idx = set(t.topk(50).indices.tolist())
        for (li, ni), inv in inventory.items():
            if li == L and ni in top_idx:
                total_matches += 1
                per_tier[inv.get("tier", "?")] += 1
                by_layer[L] += 1
    print(f"\n[inventory matches in top-50 per layer]")
    print(f"  total: {total_matches}")
    for tier, c in sorted(per_tier.items(), key=lambda kv: -kv[1]):
        print(f"    {tier:<15}: {c}")
    print(f"  layers with matches:")
    for L, c in sorted(by_layer.items()):
        print(f"    L{L:>2}: {c}")

    # ---- cross-QA gradient consistency (which neurons fire across all 5 QAs?) ----
    # Check: among per-step grads, how many neurons are top-20 in ALL 5 QAs?
    # Build per-QA cum_neuron from history
    per_qa_neuron = defaultdict(dict)  # qa_id -> {L: tensor}
    for h in history:
        # reconstruct: we don't have per-step per_neuron in history
        # only have grad_norms per (layer, module)
        pass
    # Actually we don't have it from the saved data, so skip this analysis.
    # Instead, we trust that the cross-QA averaging effect is in the cum_grad totals.

    print("\n[done]")


if __name__ == "__main__":
    main()
