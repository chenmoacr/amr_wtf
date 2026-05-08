"""
QA01 hidden state replacement at condensation layers.

Goal:
  At each sentence-final decision position t-1 (the token right before 「。」
  in the Opus output), replace the hidden state output of layer L with the
  unembed direction of Opus's actual next token (scaled to keep the same
  L2-norm).

  We only inject in the "condensation" layers (L24-L35), since the logit
  lens already showed L0-L23 are high-entropy "computation" layers where
  injection would be overwritten by the residual rectification of layers
  24+.

  Then forward through the rest of the model and read the final-layer
  logit at the same decision position. The question is whether
    P(opus_target_token | injected) > P(opus_target_token | baseline)
  and whether top-1 of the final layer becomes opus_target_token.

  This directly answers: "if we tell mid-late layers what we want, does the
  end-layer obey?"

Setup:
  - Teacher-forced sequence: chat_template(user) + opus_output (no thought)
  - 4 selected positions covering different sentence types
  - 3 candidate layers: L24, L30, L34

Output:
  outputs/opus47_steer/
    steer_results.pt      raw records
    steer_summary.txt     human-readable per-position table
    run.log
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "claudeopusQA01.json"
LENS_PT = ROOT / "outputs" / "opus47_logit_lens" / "sentence_candidates.pt"
OUT_DIR = ROOT / "outputs" / "opus47_steer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
TEST_LAYERS = [24, 30, 34]   # condensation-zone layers only
SELECTED_LENS_INDICES = [6, 13, 15, 26]   # sent #7, #14, #16, #27 from logit_lens results
TOP_K = 5


def make_replace_hook(position, direction_unit):
    """Replace output of a transformer block at `position` with
    direction_unit * (original_norm at position). Keep magnitude, change
    direction."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None
        if position < h.shape[1]:
            orig_norm = h[0, position, :].float().norm()
            new_state = direction_unit.to(h.dtype).to(h.device) * orig_norm.to(h.dtype)
            h = h.clone()
            h[0, position, :] = new_state
        if rest is not None:
            return (h,) + rest
        return h
    return hook


def topk_at(logits_pos, k):
    probs = F.softmax(logits_pos.float(), dim=-1)
    top = probs.topk(k)
    return top.indices.tolist(), top.values.tolist()


def main():
    print(f"[load] {MODEL_PATH}  (BF16)", flush=True)
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
    unembed_w = model.model.language_model.embed_tokens.weight  # tied
    print(f"  layers={len(layers)}  alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)

    # ---- build sequence ----
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    msgs = [
        {"role": "user", "content": qa["input"]},
        {"role": "assistant", "content": qa["output"]},
    ]
    enc = tokenizer.apply_chat_template(
        msgs, tokenize=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    )
    full_ids = enc["input_ids"]
    T = full_ids.shape[1]
    print(f"[seq] T={T}", flush=True)

    # ---- pick selected positions from earlier logit-lens results ----
    lens_data = torch.load(LENS_PT, weights_only=False)
    lens_results = lens_data["results"]
    selected = []
    for si in SELECTED_LENS_INDICES:
        if si >= len(lens_results):
            continue
        r = lens_results[si]
        selected.append({
            "lens_idx": si,
            "decision_pos": r["decision_pos"],
            "period_pos": r["period_pos"],
            "context_tail": r["context_tail"][-30:],
            "actual_period_token": r["actual_period_token"],
            "decision_token": tokenizer.decode([int(full_ids[0, r["decision_pos"]])],
                                                skip_special_tokens=False),
            "target_id": int(full_ids[0, r["period_pos"]]),
        })
    print(f"[selected] {len(selected)} positions", flush=True)
    for s in selected:
        print(f"  decision_pos={s['decision_pos']:>5d}  '{s['decision_token']}'  "
              f"→  target id={s['target_id']:>6d}  '{s['actual_period_token']}'  "
              f"(...{s['context_tail']!r})", flush=True)

    # ---- baseline forward ----
    print("\n[baseline] running clean forward (no injection)...", flush=True)
    t0 = time.time()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out_base = model(input_ids=full_ids.to(DEVICE), use_cache=False)
    base_logits = out_base.logits.detach()  # keep on GPU for indexing
    print(f"  done in {time.time()-t0:.1f}s  peak={torch.cuda.max_memory_allocated(0)/1e9:.2f}GB",
          flush=True)
    torch.cuda.reset_peak_memory_stats(0)

    # baseline top-5 + target prob at each decision position
    for s in selected:
        dp = s["decision_pos"]
        ids, probs = topk_at(base_logits[0, dp], TOP_K)
        toks = [tokenizer.decode([i], skip_special_tokens=False) for i in ids]
        target_prob = F.softmax(base_logits[0, dp].float(), dim=-1)[s["target_id"]].item()
        target_in_top5 = s["target_id"] in ids
        s["baseline_top5"] = list(zip(toks, probs))
        s["baseline_target_prob"] = target_prob
        s["baseline_target_in_top5"] = target_in_top5
    del out_base, base_logits
    torch.cuda.empty_cache()

    # ---- steering experiments ----
    print("\n[steer] testing layer-wise replacement...", flush=True)
    records = []
    for s in selected:
        target_id = s["target_id"]
        target_dir = unembed_w[target_id].detach().float()
        target_unit = (target_dir / target_dir.norm()).to(unembed_w.dtype).to(DEVICE)

        for L in TEST_LAYERS:
            torch.cuda.empty_cache()
            handle = layers[L].register_forward_hook(
                make_replace_hook(s["decision_pos"], target_unit)
            )
            try:
                t0 = time.time()
                with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    out = model(input_ids=full_ids.to(DEVICE), use_cache=False)
                logits = out.logits[0, s["decision_pos"]].detach().float()
                ids_, probs = topk_at(logits, TOP_K)
                toks = [tokenizer.decode([i], skip_special_tokens=False) for i in ids_]
                target_prob = F.softmax(logits, dim=-1)[target_id].item()
                target_in_top5 = target_id in ids_
                target_top1 = (ids_[0] == target_id)
                dt = time.time() - t0
                peak = torch.cuda.max_memory_allocated(0) / 1e9
                rec = {
                    "lens_idx": s["lens_idx"],
                    "context_tail": s["context_tail"],
                    "decision_pos": s["decision_pos"],
                    "decision_token": s["decision_token"],
                    "target_id": target_id,
                    "target_token": s["actual_period_token"],
                    "layer": L,
                    "top5": list(zip(toks, probs)),
                    "target_prob": target_prob,
                    "target_in_top5": target_in_top5,
                    "target_top1": target_top1,
                    "elapsed_s": dt,
                }
                records.append(rec)
                arrow = "★" if target_top1 else (" " if target_in_top5 else "✗")
                print(f"  {arrow} L{L:>2}  '{s['decision_token']}'  "
                      f"target={s['actual_period_token']!r}  "
                      f"P_target = {s['baseline_target_prob']:.3f} → {target_prob:.3f}  "
                      f"top-1 → {toks[0]!r}@{probs[0]:.3f}  ({dt:.1f}s, peak {peak:.2f}GB)",
                      flush=True)
                torch.cuda.reset_peak_memory_stats(0)
                del out, logits
            finally:
                handle.remove()
            torch.cuda.empty_cache()

    # ---- save ----
    torch.save({
        "test_layers": TEST_LAYERS,
        "selected": selected,
        "records": records,
    }, OUT_DIR / "steer_results.pt")

    # ---- write txt summary ----
    with open(OUT_DIR / "steer_summary.txt", "w", encoding="utf-8") as f:
        f.write("================ QA01 Hidden State Replacement at Condensation Layers ================\n\n")
        f.write(f"layers tested : {TEST_LAYERS}\n")
        f.write(f"replacement   : h[0, decision_pos, :] := unembed[opus_target] / ||unembed|| * ||h_orig||\n")
        f.write(f"               (unit direction of target token, magnitude preserved)\n\n")

        f.write(f"\n{'='*78}\n")
        f.write(f"BASELINE (no injection): per-position final-layer top-{TOP_K}\n")
        f.write(f"{'='*78}\n")
        for s in selected:
            f.write(f"\n--- ...{s['context_tail']!r} → target '{s['actual_period_token']}' ---\n")
            f.write(f"  decision token       : '{s['decision_token']}'\n")
            f.write(f"  baseline P(target)   : {s['baseline_target_prob']:.4f}\n")
            f.write(f"  baseline target_in_top5: {s['baseline_target_in_top5']}\n")
            f.write(f"  baseline top-{TOP_K}        : ")
            f.write(" | ".join(f"{t!r}@{p:.3f}" for t, p in s["baseline_top5"]))
            f.write("\n")

        f.write(f"\n\n{'='*78}\n")
        f.write(f"AFTER REPLACEMENT at L24, L30, L34\n")
        f.write(f"{'='*78}\n")
        for s in selected:
            f.write(f"\n--- ...{s['context_tail']!r} → target '{s['actual_period_token']}' ---\n")
            f.write(f"  decision token : '{s['decision_token']}'\n")
            f.write(f"  baseline P(target) = {s['baseline_target_prob']:.4f}\n")
            this_pos_records = [r for r in records if r["lens_idx"] == s["lens_idx"]]
            f.write(f"\n  L  | P(target) | top-1 (token : prob)              | target ↑/↓ | top1=target?\n")
            f.write(f"  ---+-----------+-----------------------------------+-----------+----------\n")
            for r in this_pos_records:
                base_p = s["baseline_target_prob"]
                delta = r["target_prob"] - base_p
                arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "·")
                top1_match = "★ YES" if r["target_top1"] else "no"
                top1_tok, top1_prob = r["top5"][0]
                f.write(f"  L{r['layer']:>2} | {r['target_prob']:>7.4f}   | "
                        f"{top1_tok!r:>12s}@{top1_prob:.3f}                 | "
                        f"{arrow} {abs(delta):.3f}    | {top1_match}\n")
            # Show full top-5 for the strongest layer for this position
            best_rec = max(this_pos_records, key=lambda r: r["target_prob"])
            f.write(f"\n  best layer = L{best_rec['layer']}; full top-{TOP_K}:\n    ")
            f.write(" | ".join(f"{t!r}@{p:.3f}" for t, p in best_rec["top5"]))
            f.write("\n")

        f.write(f"\n\n{'='*78}\n")
        f.write(f"GLOBAL (across {len(selected)} positions × {len(TEST_LAYERS)} layers)\n")
        f.write(f"{'='*78}\n")
        for L in TEST_LAYERS:
            this_l = [r for r in records if r["layer"] == L]
            n_top1 = sum(r["target_top1"] for r in this_l)
            n_in5 = sum(r["target_in_top5"] for r in this_l)
            mean_pt = sum(r["target_prob"] for r in this_l) / max(len(this_l), 1)
            f.write(f"  L{L:>2}: target_top1 = {n_top1}/{len(this_l)},  "
                    f"target_in_top5 = {n_in5}/{len(this_l)},  mean P(target) = {mean_pt:.3f}\n")
        # Compare to baseline
        n_base_top1 = 0
        n_base_in5 = sum(s["baseline_target_in_top5"] for s in selected)
        mean_base = sum(s["baseline_target_prob"] for s in selected) / max(len(selected), 1)
        f.write(f"  baseline: target_top1 = (see baseline rows),  "
                f"target_in_top5 = {n_base_in5}/{len(selected)},  mean P(target) = {mean_base:.3f}\n")

    print(f"\n[save] {OUT_DIR / 'steer_results.pt'}", flush=True)
    print(f"[save] {OUT_DIR / 'steer_summary.txt'}", flush=True)


if __name__ == "__main__":
    main()
