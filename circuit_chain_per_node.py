"""
Per-node ablation of the 7-node chain — find the stop-signal carrier.

Logic:
  Run GB01-B forward (full sequence including the trailing <turn|> EOS).
  At the position predicting <turn|>, record the model's logit / prob for
  <turn|>. This is the "do I want to stop now?" signal.

  For each chain node (L, n) in isolation:
    install ablate hook on layers[L].mlp.down_proj
    forward
    measure logit[<turn|>] at the prediction position
    delta vs baseline = how much that single neuron contributed to the
                        decision to emit EOS

  The node whose ablation drops the EOS logit / prob the most is the
  empirical "stop signal carrier" within the chain.

  We also test the full 7-node simultaneous ablate as a sanity check
  (must drop EOS logit substantially, otherwise the chain framing is
  itself wrong).

Cost: 9 forward passes (8 ablate + 1 baseline), ~40 seconds total.
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

from circuit_attr_GB01 import build_sequence

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
OUT_DIR = ROOT / "outputs" / "circuit_attr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"

CHAIN = [
    ( 9,  357, "L09#357 driver→L12"),
    ( 9,  420, "L09#420 suppressor→L12"),
    ( 9, 6039, "L09#6039 FFI"),
    (11, 2089, "L11#2089 mid amp"),
    (12, 4638, "L12#4638 mid suppressor"),
    (24, 5330, "L24#5330 mid hub"),
    (33, 9054, "L33#9054 terminal"),
]


def make_ablate_hook(neuron_idx):
    def pre(module, inputs):
        x = inputs[0].clone()
        x[..., neuron_idx] = 0
        return (x,) + inputs[1:]
    return pre


def install_ablate(layers, nodes):
    """nodes: list of (li, ni). returns handles."""
    handles = []
    for li, ni in nodes:
        h = layers[li].mlp.down_proj.register_forward_pre_hook(
            make_ablate_hook(ni)
        )
        handles.append(h)
    return handles


def remove_handles(handles):
    for h in handles:
        h.remove()


def forward_eos_signal(model, full_ids, eot_id, predict_pos):
    """Single forward; return (logit, prob, top1_id, top1_prob) at predict_pos."""
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        out = model(input_ids=full_ids.unsqueeze(0).to(DEVICE), use_cache=False)
    logits = out.logits[0, predict_pos]   # [vocab]
    eot_logit = float(logits[eot_id].item())
    probs = F.softmax(logits.float(), dim=-1)
    eot_prob = float(probs[eot_id].item())
    top1_id = int(probs.argmax().item())
    top1_prob = float(probs[top1_id].item())
    return eot_logit, eot_prob, top1_id, top1_prob


def main():
    print(f"[load] {MODEL_PATH}  (BF16)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    model.eval()
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    layers = model.model.language_model.layers
    print(f"  layers={len(layers)}, alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)

    soc = tokenizer.convert_tokens_to_ids("<|channel>")
    eoc = tokenizer.convert_tokens_to_ids("<channel|>")
    eot = tokenizer.convert_tokens_to_ids("<turn|>")
    print(f"[tok] eot=<turn|> id={eot}")

    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    item_b = data["B"][0]
    full_b, P_b = build_sequence(tokenizer, item_b, soc, eoc, eot)
    print(f"[seq-B] T={full_b.shape[0]} (prompt={P_b}) last_token={int(full_b[-1])}")
    assert int(full_b[-1]) == eot, "expected last token = <turn|>"
    predict_pos = full_b.shape[0] - 2  # position whose logits predict the final <turn|>

    # decode the last few tokens for context
    tail_ids = full_b[-8:].tolist()
    print(f"[seq-B] tail tokens: {tail_ids}")
    print(f"[seq-B] tail decoded: {tokenizer.decode(tail_ids, skip_special_tokens=False)!r}")

    # ---- baseline ----
    t0 = time.time()
    base_logit, base_prob, base_top1, base_top1_prob = forward_eos_signal(
        model, full_b, eot, predict_pos
    )
    base_top1_tok = tokenizer.decode([base_top1], skip_special_tokens=False)
    print(f"\n[baseline] predict_pos={predict_pos}  ({time.time()-t0:.1f}s)")
    print(f"  EOS  logit={base_logit:+8.3f}  prob={base_prob:.4f}")
    print(f"  TOP1 id={base_top1} ({base_top1_tok!r}) prob={base_top1_prob:.4f}")

    # ---- single-node ablations ----
    rows = []
    print(f"\n--- single-node ablations ---")
    for li, ni, label in CHAIN:
        torch.cuda.empty_cache(); gc.collect()
        handles = install_ablate(layers, [(li, ni)])
        try:
            t0 = time.time()
            log, prob, top1, top1p = forward_eos_signal(model, full_b, eot, predict_pos)
        finally:
            remove_handles(handles)
        delta_logit = log - base_logit
        rel_prob = (prob - base_prob) / base_prob if base_prob > 1e-9 else 0.0
        top1_tok = tokenizer.decode([top1], skip_special_tokens=False)
        print(f"  L{li:02d}#{ni:<5d}  EOS logit={log:+8.3f} (Δ={delta_logit:+7.3f})  "
              f"prob={prob:.4f} ({rel_prob*100:+6.1f}%)  "
              f"TOP1=#{top1}={top1_tok!r}@{top1p:.3f}  ({time.time()-t0:.1f}s)", flush=True)
        rows.append({
            "layer": li, "neuron": ni, "label": label,
            "eos_logit": log, "eos_prob": prob,
            "delta_logit": delta_logit, "rel_prob": rel_prob,
            "top1_id": top1, "top1_token": top1_tok, "top1_prob": top1p,
        })

    # ---- full-chain ablation (sanity) ----
    print(f"\n--- full chain ablation ---")
    handles = install_ablate(layers, [(li, ni) for li, ni, _ in CHAIN])
    try:
        t0 = time.time()
        log, prob, top1, top1p = forward_eos_signal(model, full_b, eot, predict_pos)
    finally:
        remove_handles(handles)
    full_delta = log - base_logit
    full_rel = (prob - base_prob) / base_prob if base_prob > 1e-9 else 0.0
    top1_tok = tokenizer.decode([top1], skip_special_tokens=False)
    print(f"  FULL CHAIN  EOS logit={log:+8.3f} (Δ={full_delta:+7.3f})  "
          f"prob={prob:.4f} ({full_rel*100:+6.1f}%)  "
          f"TOP1=#{top1}={top1_tok!r}@{top1p:.3f}  ({time.time()-t0:.1f}s)")

    # ---- summary ----
    print("\n" + "=" * 80)
    print(f"  STOP SIGNAL CARRIER RANKING  (baseline EOS prob = {base_prob:.4f})")
    print("=" * 80)
    rows_sorted = sorted(rows, key=lambda r: r["delta_logit"])
    print(f"  {'L#nrn':>10}  {'logit Δ':>9}  {'prob shift':>11}  TOP1")
    for r in rows_sorted:
        print(f"  L{r['layer']:02d}#{r['neuron']:<5d}  {r['delta_logit']:+9.3f}  "
              f"{r['rel_prob']*100:+10.1f}%  #{r['top1_id']}={r['top1_token']!r}@{r['top1_prob']:.3f}")
    print(f"  {'FULL':>10}    {full_delta:+9.3f}  {full_rel*100:+10.1f}%   ablation of all 7 simultaneously")
    print("=" * 80)
    print("\nInterpretation:")
    print("  Most negative logit Δ  ⇒  this neuron most strongly carries the EOS decision.")
    print("  Single-node Δ summed > full-chain Δ  ⇒  redundancy / superposition.")
    print("  Single-node Δ summed < full-chain Δ  ⇒  synergistic / non-linear chain.")

    out = {
        "baseline": {
            "eos_logit": base_logit, "eos_prob": base_prob,
            "top1_id": base_top1, "top1_prob": base_top1_prob,
        },
        "single_node": rows,
        "full_chain": {
            "eos_logit": log, "eos_prob": prob,
            "delta_logit": full_delta, "rel_prob": full_rel,
        },
    }
    torch.save(out, OUT_DIR / "circuit_chain_per_node.pt")
    print(f"\n[save] {OUT_DIR / 'circuit_chain_per_node.pt'}")


if __name__ == "__main__":
    main()
