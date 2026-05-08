"""
QA01 logit lens at sentence-final positions.

Teacher-forced sequence:
  user turn = QA01 input
  model turn = QA01 Opus output (NO thought segment)

Logic:
  We feed Gemma 4 E2B the full (prompt + Opus answer) sequence as a single
  forward pass. For every token whose decoded form contains a Chinese full
  stop "。", we treat the *previous* position as the "decision position":
  it is the place whose hidden state is responsible for predicting the
  period token. At that position, we apply logit lens to every transformer
  layer:

    logits_L[t] = final_norm( hidden_states[L][t] ) @ unembed^T

  Top-5 of softmax(logits_L[t]) tells us, per layer, what tokens the model
  considers most likely. We compare these layer-by-layer top-5 to see:
    - early layer candidates (lexical / syntactic)
    - mid layer candidates (semantic)
    - late layer candidates (final decision)
    - whether sentence-end is well-predicted from layer X onwards

Output: outputs/opus47_logit_lens/sentence_candidates.txt + .pt
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
OUT_DIR = ROOT / "outputs" / "opus47_logit_lens"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
TOP_K = 5
LOGIT_SOFTCAP = 30.0   # Gemma 4 final_logit_softcapping


def get_unembed_and_norm(model):
    """Return (final_norm_module, unembed_weight) for Gemma4."""
    lm = model.model.language_model
    final_norm = lm.norm           # RMSNorm
    embed = lm.embed_tokens.weight # tied, so this is also unembed
    return final_norm, embed


def logit_lens_topk(hidden_state_2d, final_norm, unembed_w, k=TOP_K, softcap=LOGIT_SOFTCAP):
    """hidden_state_2d: [T, hidden].  final_norm: nn.Module.  unembed_w: [vocab, hidden].
    Returns top-k indices and probs at each row."""
    h = final_norm(hidden_state_2d)        # [T, hidden]
    logits = h.float() @ unembed_w.float().T  # [T, vocab]
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    probs = F.softmax(logits, dim=-1)
    top = probs.topk(k, dim=-1)
    return top.indices, top.values


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
    n_layers = len(layers)
    print(f"  layers={n_layers}, alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)

    final_norm, unembed_w = get_unembed_and_norm(model)
    print(f"[unembed] shape={tuple(unembed_w.shape)}  (tied with embed_tokens)", flush=True)

    # ---- build teacher-forced sequence ----
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = qa["input"]
    opus_text = qa["output"]
    print(f"[input] user={len(user_text)}c  opus={len(opus_text)}c", flush=True)

    # No thought segment: directly user → assistant content
    msgs_full = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": opus_text},
    ]
    enc_full = tokenizer.apply_chat_template(
        msgs_full, tokenize=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    )
    full_ids = enc_full["input_ids"].to(DEVICE)
    T = full_ids.shape[1]

    # Locate where the assistant content starts (need it to scope period search
    # to the model output side, not user side).
    msgs_user_only = [{"role": "user", "content": user_text}]
    enc_user = tokenizer.apply_chat_template(
        msgs_user_only, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    assistant_start = enc_user["input_ids"].shape[1]
    print(f"[seq] T={T}  assistant_start={assistant_start}  "
          f"user_len_in_seq={assistant_start}  assistant_len={T - assistant_start}", flush=True)

    # ---- find Chinese-period positions in assistant region ----
    # A period position is t such that decode(full_ids[t]) contains "。".
    # The "decision position" predicting that period is t-1.
    period_positions = []
    for t in range(assistant_start, T):
        tok_str = tokenizer.decode([int(full_ids[0, t])], skip_special_tokens=False)
        if "。" in tok_str:
            period_positions.append(t)
    print(f"[periods] {len(period_positions)} sentence-end tokens in assistant region", flush=True)

    if not period_positions:
        print("[fatal] no Chinese period tokens found", flush=True)
        return

    # ---- single forward to get all hidden states ----
    print(f"\n[forward] running teacher-forced forward (T={T}) ...", flush=True)
    t0 = time.time()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model(input_ids=full_ids, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states  # tuple of length n_layers+1
    print(f"  done in {time.time()-t0:.1f}s, hidden_states tuple len={len(hidden_states)}",
          flush=True)
    print(f"  hidden_states[0].shape={tuple(hidden_states[0].shape)}  "
          f"(embedding output)", flush=True)
    print(f"  hidden_states[-1].shape={tuple(hidden_states[-1].shape)}  "
          f"(after layer {n_layers-1})", flush=True)

    # del large tensors we don't need
    del out
    torch.cuda.empty_cache()

    # ---- logit lens for each period decision position ----
    # We want, at every period position t, the layer-wise top-K predictions
    # produced by hidden_states[L][:, t-1, :] for L in [0, n_layers].
    # Layer 0 = embedding output (no transformer block applied yet).
    # Layer n_layers = output of the last transformer block (= input to final_norm).
    decision_positions = [t - 1 for t in period_positions]

    # Stack hidden states at decision positions: shape [n_layers+1, n_pos, hidden]
    n_pos = len(decision_positions)
    n_layer_outputs = len(hidden_states)
    print(f"\n[lens] applying logit lens to {n_pos} decision positions × "
          f"{n_layer_outputs} layer outputs", flush=True)

    results = []
    for pi, (period_pos, decision_pos) in enumerate(zip(period_positions, decision_positions)):
        # Context = previous 30 tokens for human reading
        ctx_lo = max(0, decision_pos - 30)
        ctx = tokenizer.decode(full_ids[0, ctx_lo:decision_pos + 1].tolist(),
                               skip_special_tokens=False)
        actual_period_tok = tokenizer.decode(
            [int(full_ids[0, period_pos])], skip_special_tokens=False
        )

        # Per-layer top-K
        layer_top = []
        for L in range(n_layer_outputs):
            h = hidden_states[L][0, decision_pos:decision_pos + 1, :]  # [1, hidden]
            top_idx, top_val = logit_lens_topk(h, final_norm, unembed_w, k=TOP_K)
            top_idx = top_idx[0].cpu().tolist()
            top_val = top_val[0].cpu().tolist()
            top_tok = [tokenizer.decode([i], skip_special_tokens=False) for i in top_idx]
            layer_top.append({"layer": L, "tokens": top_tok, "probs": top_val,
                              "ids": top_idx})
        results.append({
            "period_pos": period_pos,
            "decision_pos": decision_pos,
            "context_tail": ctx[-60:],   # last 60 chars before period
            "actual_period_token": actual_period_tok,
            "layer_top": layer_top,
        })
        if (pi + 1) % 5 == 0 or pi == n_pos - 1:
            print(f"    [{pi+1}/{n_pos}] pos={period_pos}  context_tail={ctx[-30:]!r}",
                  flush=True)

    # ---- save ----
    pt_path = OUT_DIR / "sentence_candidates.pt"
    torch.save({
        "n_layers": n_layers,
        "n_layer_outputs": n_layer_outputs,
        "n_periods": n_pos,
        "results": results,
    }, pt_path)
    print(f"\n[save] {pt_path}", flush=True)

    # human-readable txt
    txt_path = OUT_DIR / "sentence_candidates.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"================ QA01 logit lens — sentence-final positions ================\n")
        f.write(f"model         : {MODEL_PATH}\n")
        f.write(f"sequence_T    : {T}\n")
        f.write(f"assistant_T   : {T - assistant_start}\n")
        f.write(f"layers        : {n_layers}\n")
        f.write(f"layer outputs : {n_layer_outputs} (= embedding + {n_layers} blocks)\n")
        f.write(f"top_k         : {TOP_K}\n")
        f.write(f"period count  : {n_pos}\n\n")

        for pi, r in enumerate(results):
            f.write(f"\n{'='*78}\n")
            f.write(f"#{pi+1}  period_pos={r['period_pos']}  decision_pos={r['decision_pos']}\n")
            f.write(f"context tail: ...{r['context_tail']}\n")
            f.write(f"actual period token: {r['actual_period_token']!r}\n")
            f.write(f"\nLayer | top-{TOP_K} (token : prob)\n")
            f.write(f"------+" + "-" * 70 + "\n")
            for lt in r["layer_top"]:
                cells = " | ".join(
                    f"{t!r}@{p:.3f}" for t, p in zip(lt["tokens"], lt["probs"])
                )
                f.write(f"  L{lt['layer']:>2}  | {cells}\n")
        f.write("\n\n================ summary ================\n")
        # Per-layer "fraction of period predictions that have 。 as top-1"
        period_id_set = set()
        for r in results:
            decoded = r["actual_period_token"]
            # Map to id if pure period
            if decoded == "。":
                # find the id directly
                tok_id = tokenizer.encode("。", add_special_tokens=False)
                if len(tok_id) == 1:
                    period_id_set.add(tok_id[0])
        # Only meaningful if 。 is a single token
        if period_id_set:
            ptid = next(iter(period_id_set))
            f.write(f"\n'。' single-token id: {ptid}\n")
            f.write(f"Fraction of decision positions where layer L's top-1 is '。':\n")
            for L in range(n_layer_outputs):
                hits = sum(1 for r in results if r["layer_top"][L]["ids"][0] == ptid)
                f.write(f"  L{L:>2}: {hits}/{n_pos} = {hits/n_pos*100:5.1f}%\n")
    print(f"[save] {txt_path}", flush=True)
    print(f"[done] {n_pos} period positions, "
          f"{n_layer_outputs} layer outputs each top-{TOP_K}", flush=True)


if __name__ == "__main__":
    main()
