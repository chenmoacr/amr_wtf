"""
Focused reader: pull only the diagnostic numbers from intent_sft and
compare side-by-side with grad_probe. Prints to stdout (under ~200 lines).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
PROBE_PT  = ROOT / "outputs" / "opus47_grad_probe" / "raw.pt"
INTENT_PT = ROOT / "outputs" / "opus47_intent_sft"  / "raw.pt"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def cum_grad_layer_totals(cum_grad, n_layers=35):
    return [(L, sum(cum_grad.get((L, m), 0.0) for m in TARGET_MODULES))
            for L in range(n_layers)]


def cum_grad_module_totals(cum_grad):
    out = {m: 0.0 for m in TARGET_MODULES}
    for (L, m), v in cum_grad.items():
        out[m] += v
    return out


def main():
    if not INTENT_PT.exists():
        print(f"[fatal] {INTENT_PT} missing")
        return
    intent = torch.load(INTENT_PT, weights_only=False)
    probe  = torch.load(PROBE_PT,  weights_only=False) if PROBE_PT.exists() else None

    cfg = intent["config"]
    target_strs = intent["target_strs"]
    scores      = intent["scores"]
    final_w     = intent["final_weights"]
    intent_idx  = intent["intent_idx"]
    mixed_idx   = intent["mixed_idx"]
    know_idx    = intent["knowledge_idx"]
    glue_pos    = set(intent["glue_positions"])
    history     = intent["history"]
    cum_grad    = intent["cum_grad"]
    cum_neuron  = intent["cum_neuron_down"]

    n_target = intent["n_target"]
    n_glue_active = sum(1 for i in glue_pos if final_w[i] > 0)

    print("=" * 70)
    print("  QA01 intent_sft — focused diagnostic")
    print("=" * 70)
    print(f"  thresholds: INTENT>{cfg['INTENT_THRESH']}  "
          f"MIXED>={cfg['MIXED_THRESH']}  glue_boost=×{cfg['GLUE_BOOST']}")
    print(f"  steps={cfg['N_STEPS']}  LR={cfg['LR']}  LoRA r={cfg['LORA_R']}")

    # ---- bucket sizes ----
    print(f"\n[buckets]  n_target={n_target}")
    print(f"  intent    (w=1.0): {len(intent_idx):>4}  "
          f"({100*len(intent_idx)/n_target:.1f}%)")
    print(f"  mixed     (w=0.4): {len(mixed_idx):>4}  "
          f"({100*len(mixed_idx)/n_target:.1f}%)")
    print(f"  knowledge (w=0  ): {len(know_idx):>4}  "
          f"({100*len(know_idx)/n_target:.1f}%)  ← masked")
    print(f"  glue boosted     : {n_glue_active:>4}  "
          f"(of {len(glue_pos)} glue-mapped)")

    # ---- top 20 intent tokens by score ----
    print(f"\n[top intent tokens by Δloss]")
    intent_sorted = sorted(intent_idx, key=lambda i: scores[i], reverse=True)[:20]
    for i in intent_sorted:
        ctx = "".join(target_strs[max(0, i-3):i+1])
        glue_flag = "★glue" if i in glue_pos else ""
        print(f"  pos={i:>4}  Δ={scores[i]:6.2f}  w={final_w[i]:.2f}  "
              f"tok={target_strs[i]!r:>10}  ctx={ctx[-15:]!r}  {glue_flag}")

    # ---- bottom 20 knowledge tokens by score ----
    print(f"\n[bottom knowledge tokens (masked)]")
    know_sorted = sorted(know_idx, key=lambda i: scores[i])[:20]
    for i in know_sorted:
        ctx = "".join(target_strs[max(0, i-3):i+1])
        print(f"  pos={i:>4}  Δ={scores[i]:6.2f}  "
              f"tok={target_strs[i]!r:>10}  ctx={ctx[-15:]!r}")

    # ---- training curve ----
    print(f"\n[training curve]")
    print(f"  step | weighted_loss | plain_loss")
    for h in history:
        print(f"  {h['step']+1:>3}  | {h['weighted_loss']:>13.4f} | "
              f"{h['plain_loss']:>10.4f}")

    # intent-bucket-only convergence
    init_per_tok = history[0]["per_tok_loss"]
    final_per_tok = history[-1]["per_tok_loss"]
    intent_init = sum(init_per_tok[i] for i in intent_idx) / max(1, len(intent_idx))
    intent_final = sum(final_per_tok[i] for i in intent_idx) / max(1, len(intent_idx))
    know_init = sum(init_per_tok[i] for i in know_idx) / max(1, len(know_idx))
    know_final = sum(final_per_tok[i] for i in know_idx) / max(1, len(know_idx))
    print(f"\n  intent-bucket  : init={intent_init:.3f} → final={intent_final:.3f}  "
          f"Δ={intent_init-intent_final:+.3f}")
    print(f"  knowledge-bucket: init={know_init:.3f} → final={know_final:.3f}  "
          f"Δ={know_init-know_final:+.3f}  (these were masked, so should not move)")

    # ---- compare vs probe ----
    if probe is not None:
        ph = probe["history"]
        probe_step1 = ph[0]["loss"]
        probe_step20 = ph[-1]["loss"]
        probe_module = cum_grad_module_totals({k: v for k, v in
                                                {(li, m): val for (li, m), val
                                                 in {**probe["cum_grad"]}.items()}.items()})
        probe_layer = cum_grad_layer_totals(probe["cum_grad"])

        print(f"\n[vs grad_probe]")
        print(f"  probe  : loss step1={probe_step1:.4f}  step20={probe_step20:.4f}  "
              f"Δ={probe_step1-probe_step20:.4f}")
        print(f"  intent : weighted step1={history[0]['weighted_loss']:.4f}  "
              f"step20={history[-1]['weighted_loss']:.4f}  "
              f"Δ={history[0]['weighted_loss']-history[-1]['weighted_loss']:.4f}")

    # ---- module-type totals ----
    intent_module = cum_grad_module_totals(cum_grad)
    print(f"\n[cumulative grad — module totals]")
    print(f"  module     | intent run  | probe run   | ratio (i/p)")
    if probe is not None:
        probe_module_run = cum_grad_module_totals(probe["cum_grad"])
    for m in TARGET_MODULES:
        v = intent_module[m]
        if probe is not None:
            p = probe_module_run[m]
            r = v / max(p, 1e-9)
            print(f"  {m:<10} | {v:>10.2f}  | {p:>10.2f}  | {r:.2f}×")
        else:
            print(f"  {m:<10} | {v:>10.2f}")

    # ---- layer ranking ----
    layer_tot = cum_grad_layer_totals(cum_grad)
    layer_tot_sorted = sorted(layer_tot, key=lambda kv: kv[1], reverse=True)
    print(f"\n[top-10 layers by total cum-grad]")
    if probe is not None:
        probe_layer = dict(cum_grad_layer_totals(probe["cum_grad"]))
        print(f"  rank  layer  intent_total  probe_total  ratio")
        for rk, (L, v) in enumerate(layer_tot_sorted[:10], 1):
            p = probe_layer.get(L, 0)
            print(f"  {rk:>3}   L{L:>2}    {v:>9.2f}    {p:>9.2f}   "
                  f"{v/max(p,1e-9):.2f}×")
    else:
        for rk, (L, v) in enumerate(layer_tot_sorted[:10], 1):
            print(f"  {rk:>3}   L{L:>2}    total={v:.2f}")

    # ---- per-neuron max comparison (L27, L34) ----
    if probe is not None:
        probe_neuron = probe["cum_neuron_down"]
        print(f"\n[per-neuron max — L27 / L34 / L26 / L32]")
        print(f"  layer | intent.max  probe.max")
        for L in [26, 27, 28, 31, 32, 33, 34]:
            i_max = cum_neuron[L].max().item() if L in cum_neuron else 0
            p_max = probe_neuron[L].max().item() if L in probe_neuron else 0
            print(f"  L{L:>2}   | {i_max:>10.4f}   {p_max:>10.4f}")

    # ---- L27 top-10 in intent run with inventory check ----
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
        print(f"\n[L{L} top-15 in intent run]")
        for i, (val, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
            tag = ""
            key = (L, idx)
            if key in inventory:
                inv = inventory[key]
                tag = f"  [INV {inv.get('tier','?')[:8]}, gain={inv.get('default_gain',0):+.1f}]"
            print(f"  {i+1:>2}. L{L}#{idx:<5}  cum={val:.4f}{tag}")

    # ---- inventory match counts ----
    total_matches = 0
    matches_by_tier = {}
    for L in range(35):
        if L not in cum_neuron:
            continue
        t = cum_neuron[L]
        top_idx = set(t.topk(50).indices.tolist())
        for (li, ni), inv in inventory.items():
            if li == L and ni in top_idx:
                total_matches += 1
                tier = inv.get("tier", "?")
                matches_by_tier[tier] = matches_by_tier.get(tier, 0) + 1
    print(f"\n[inventory match counts in top-50 per layer]")
    print(f"  total: {total_matches}")
    for tier, c in sorted(matches_by_tier.items(), key=lambda kv: -kv[1]):
        print(f"    {tier:<15}: {c}")

    print("\n[done]")


if __name__ == "__main__":
    main()
