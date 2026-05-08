"""
QA01 1-shot SFT gradient probe.

Setup:
  - Single sample: (user_prompt, opus_answer) — same QA01 we've been using
  - LoRA r=16 on q/k/v/o/gate/up/down_proj across all 35 layers
  - 20 SFT steps, LR=1e-4, AdamW + cosine
  - Mask loss on prompt, compute CE only on Opus tokens

What we log per step:
  - Total CE loss
  - Per-token loss vector (over Opus tokens) — see which positions are
    hardest to learn / which improve most
  - Per-(layer, module) LoRA gradient norm — heatmap of where the model is
    "trying hardest"
  - Per-neuron gradient on down_proj.lora_A — columnwise norm gives a
    proxy for "how much the task wants to recruit neuron n"

What we save:
  outputs/opus47_grad_probe/
    raw.pt           full per-step history (loss, per-tok loss, grad norms,
                     per-neuron down_proj grads)
    summary.txt
      - per-step loss table
      - per-(layer, module) cumulative gradient heatmap
      - per-layer top-K neurons by cumulative gradient
      - amr_wtf inventory cross-reference (which known neurons hit top-K)
      - per-token loss reduction (initial vs final): which Opus tokens were
        the hardest, which improved most
    run.log
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "claudeopusQA01.json"
NEURONS_JSON = ROOT / "chat" / "neurons.json"
OUT_DIR = ROOT / "outputs" / "opus47_grad_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
N_STEPS = 20
LR = 1e-4
LORA_R = 8
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
TOP_K_NEURONS = 50
# Memory safety: truncate user prompt to keep total seq under this many tokens.
# QA01 raw prompt is 5832 tokens — too long for 12GB card with backward graph.
# We keep the END of the story (most relevant for analysis) plus the
# trailing "阅读和评价虚构故事" instruction.
MAX_PROMPT_TOKENS = 1800


def parse_lora_param_name(name: str):
    """Parse '...layers.27.mlp.down_proj.lora_A.default.weight' →
    (layer=27, module='down_proj', ab='A')."""
    parts = name.split(".")
    layer_idx = None
    if "layers" in parts:
        i = parts.index("layers")
        try:
            layer_idx = int(parts[i + 1])
        except (ValueError, IndexError):
            pass
    module = None
    for m in TARGET_MODULES:
        if m in parts:
            module = m
            break
    ab = None
    if "lora_A" in name:
        ab = "A"
    elif "lora_B" in name:
        ab = "B"
    return layer_idx, module, ab


def main():
    print(f"[load] {MODEL_PATH}  (BF16)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)

    # apply LoRA FIRST (must wrap before enabling checkpointing, otherwise
    # the checkpointing flag may not propagate through the PEFT wrapper)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable={n_trainable/1e6:.2f}M / total={n_total/1e9:.2f}B "
          f"({n_trainable/n_total*100:.3f}%)", flush=True)

    # NOW enable gradient checkpointing on the PEFT-wrapped model.
    # In transformers 5.3+ with PEFT, the outer call doesn't always propagate
    # to the inner text_model — we have to do this at multiple levels.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # explicit enable on inner text_model (belt-and-suspenders)
    inner_text_model = model.base_model.model.model.language_model
    inner_text_model.config.use_cache = False
    if hasattr(inner_text_model, "gradient_checkpointing_enable"):
        inner_text_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    # last resort: directly set the flag on every decoder layer
    for layer in inner_text_model.layers:
        if hasattr(layer, "gradient_checkpointing"):
            layer.gradient_checkpointing = True

    # CRITICAL: training mode required for checkpointing to actually fire
    model.train()

    # sanity report
    print(f"[gc] outer.is_gradient_checkpointing="
          f"{getattr(model, 'is_gradient_checkpointing', None)}", flush=True)
    print(f"[gc] inner_text.is_gradient_checkpointing="
          f"{getattr(inner_text_model, 'is_gradient_checkpointing', None)}", flush=True)
    print(f"[gc] layer[0].gradient_checkpointing="
          f"{getattr(inner_text_model.layers[0], 'gradient_checkpointing', None)}",
          flush=True)
    print(f"[gc] model.training={model.training}  "
          f"use_cache={model.config.use_cache}", flush=True)

    # locate layers list
    base = model.base_model.model.model.language_model
    n_layers = len(base.layers)
    intermediate = base.layers[0].mlp.down_proj.in_features
    print(f"[arch] layers={n_layers}  intermediate={intermediate}", flush=True)

    # ---- prepare data ----
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text_raw = qa["input"]
    opus_answer = qa["output"]

    # ---- truncate user_text if too long ----
    # First tokenize the raw user_text to measure
    msgs_user_full = [{"role": "user", "content": user_text_raw}]
    enc_full = tokenizer.apply_chat_template(
        msgs_user_full, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    full_prompt_len = enc_full["input_ids"].shape[1]

    if full_prompt_len > MAX_PROMPT_TOKENS:
        # Strategy: chop characters off the FRONT of the story until prompt
        # fits, but always keep the trailing instruction "阅读和评价虚构故事".
        # We binary-search a char-level truncation.
        instr_marker = "阅读和评价虚构故事"
        if instr_marker in user_text_raw:
            instr_idx = user_text_raw.rindex(instr_marker)
            story_part = user_text_raw[:instr_idx]
            tail_part = user_text_raw[instr_idx:]
        else:
            story_part = user_text_raw
            tail_part = ""

        lo, hi = 0, len(story_part)
        keep_chars = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = story_part[-mid:] + tail_part
            test_msgs = [{"role": "user", "content": candidate}]
            tlen = tokenizer.apply_chat_template(
                test_msgs, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True, enable_thinking=False,
            )["input_ids"].shape[1]
            if tlen <= MAX_PROMPT_TOKENS:
                lo = mid
                keep_chars = mid
            else:
                hi = mid - 1
        user_text = story_part[-keep_chars:] + tail_part
        print(f"[truncate] full prompt was {full_prompt_len} tokens > "
              f"{MAX_PROMPT_TOKENS}; truncated story to last {keep_chars} chars "
              f"(of {len(story_part)})", flush=True)
    else:
        user_text = user_text_raw
        print(f"[truncate] full prompt {full_prompt_len} ≤ {MAX_PROMPT_TOKENS}, "
              f"no truncation needed", flush=True)

    msgs_user = [{"role": "user", "content": user_text}]
    user_enc = tokenizer.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    assistant_start = user_enc["input_ids"].shape[1]

    msgs_full = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": opus_answer},
    ]
    enc = tokenizer.apply_chat_template(
        msgs_full, tokenize=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    T = input_ids.shape[1]
    n_target = T - assistant_start
    print(f"[data] T={T}  prompt={assistant_start}  target={n_target} (Opus tokens)",
          flush=True)

    # We compute loss MANUALLY (not via labels=) using logits_to_keep=K to
    # avoid materializing full [1, T, V] logits.
    # K = n_target + 1 covers the predictions for input_ids[assistant_start..T-1]
    # (the +1 because we shift by 1 for next-token prediction).
    K_LOGITS = n_target + 1

    # save target token ids for later inspection
    target_ids = input_ids[0, assistant_start:].cpu().tolist()
    target_strs = [tokenizer.decode([t], skip_special_tokens=False) for t in target_ids]

    # ---- optimizer ----
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=N_STEPS, eta_min=1e-6,
    )

    # ---- training loop ----
    history = []
    print(f"\n[train] {N_STEPS} steps  LR={LR}  LoRA r={LORA_R}", flush=True)

    # Probe whether HF supports logits_to_keep / num_logits_to_keep.
    # We try it once before the training loop.
    logits_kw = None
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                _probe = model(input_ids=input_ids, use_cache=False,
                               **{kw: K_LOGITS})
            if _probe.logits.shape[1] == K_LOGITS:
                logits_kw = kw
                print(f"[probe] using forward kwarg {kw!r} → logits shape "
                      f"{tuple(_probe.logits.shape)}", flush=True)
                del _probe
                torch.cuda.empty_cache()
                break
            del _probe
        except (TypeError, ValueError):
            continue
    if logits_kw is None:
        print("[probe] WARN: model does not support logits_to_keep; "
              "falling back to full logits (may OOM)", flush=True)

    for step in range(N_STEPS):
        t0 = time.time()
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            forward_kwargs = {"input_ids": input_ids, "use_cache": False}
            if logits_kw is not None:
                forward_kwargs[logits_kw] = K_LOGITS
            out = model(**forward_kwargs)

        # logits has shape [1, K, V] if sliced, else [1, T, V].
        # logits[i] predicts token at position [T - K + i + 1] when sliced,
        # or [i+1] when full.
        # We want predictions for tokens [assistant_start..T-1] (n_target tokens).
        if logits_kw is not None:
            # sliced — first K-1 predict tokens [assistant_start..T-1]
            shift_logits = out.logits[:, :-1, :]   # [1, n_target, V]
        else:
            # full — slice manually
            shift_logits = out.logits[:, assistant_start - 1:-1, :]  # [1, n_target, V]
        shift_labels = input_ids[:, assistant_start:]                 # [1, n_target]

        # CE in float — only on n_target=745 positions, fits in memory
        ce_per_tok = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        )
        loss = ce_per_tok.mean()
        per_tok_loss = ce_per_tok.detach().cpu().tolist()

        # free the (still-large) logits tensor before backward
        del out, shift_logits
        torch.cuda.empty_cache()

        loss.backward()

        # collect grad norms by (layer, module), and per-neuron for down_proj
        grad_norms = defaultdict(float)
        per_neuron_down = {}
        for n, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            layer_idx, module, ab = parse_lora_param_name(n)
            if layer_idx is None or module is None or ab is None:
                continue
            gn = p.grad.float().norm().item()
            grad_norms[(layer_idx, module)] += gn

            if module == "down_proj" and ab == "A":
                # A: [r, intermediate]. Column n = how much we want to
                # change the input pattern reaching neuron n.
                col_norms = p.grad.float().norm(dim=0).cpu()  # [intermediate]
                per_neuron_down[layer_idx] = col_norms

        history.append({
            "step": step,
            "loss": loss.item(),
            "per_tok_loss": per_tok_loss,
            "grad_norms": dict(grad_norms),
            "per_neuron_down": per_neuron_down,
        })

        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        peak = torch.cuda.max_memory_allocated(0) / 1e9
        torch.cuda.reset_peak_memory_stats(0)
        n_pt = len(per_tok_loss)
        pt_mean = sum(per_tok_loss) / max(1, n_pt)
        pt_max = max(per_tok_loss) if per_tok_loss else 0
        print(f"  step {step+1:>2}/{N_STEPS}  loss={loss.item():.4f}  "
              f"per_tok mean={pt_mean:.3f} max={pt_max:.3f}  "
              f"{time.time()-t0:.1f}s  peak={peak:.2f}GB", flush=True)

    # ---- aggregate ----
    print("\n[aggregate]", flush=True)

    # cumulative per-(layer, module) grad
    cum_grad = defaultdict(float)
    for h in history:
        for k, v in h["grad_norms"].items():
            cum_grad[k] += v

    # cumulative per-neuron grad for down_proj
    cum_neuron_down = {}  # {layer: tensor[intermediate]}
    for h in history:
        for L, t in h["per_neuron_down"].items():
            if L not in cum_neuron_down:
                cum_neuron_down[L] = torch.zeros_like(t)
            cum_neuron_down[L] += t

    # ---- save raw ----
    torch.save({
        "config": {
            "N_STEPS": N_STEPS, "LR": LR, "LORA_R": LORA_R,
            "LORA_ALPHA": LORA_ALPHA, "TARGET_MODULES": TARGET_MODULES,
        },
        "T": T,
        "assistant_start": assistant_start,
        "n_target": n_target,
        "target_ids": target_ids,
        "target_strs": target_strs,
        "history": history,
        "cum_grad": dict(cum_grad),
        "cum_neuron_down": cum_neuron_down,
    }, OUT_DIR / "raw.pt")
    print(f"  saved raw.pt", flush=True)

    # ---- inventory cross-ref ----
    inventory = {}
    if NEURONS_JSON.exists():
        nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
        inventory = {(n["layer"], n["index"]): n for n in nj["known_neurons"]}

    # ---- write summary ----
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("================ QA01 1-shot SFT Gradient Probe ================\n")
        f.write(f"Model:        {MODEL_PATH}\n")
        f.write(f"Sample:       QA01 (prompt + Opus answer)\n")
        f.write(f"Setup:        LoRA r={LORA_R} α={LORA_ALPHA}, "
                f"target={TARGET_MODULES}\n")
        f.write(f"Steps:        {N_STEPS}  LR={LR}  optimizer=AdamW (cosine)\n")
        f.write(f"Sequence:     T={T}  prompt={assistant_start}  target={n_target}\n\n")

        # ---- per-step loss ----
        f.write("================ Per-step loss ================\n")
        f.write(f"  step   loss     per_tok_mean  per_tok_max\n")
        for h in history:
            ptl = h["per_tok_loss"]
            f.write(f"  {h['step']+1:>3}    {h['loss']:.4f}   "
                    f"{sum(ptl)/max(1,len(ptl)):.4f}        "
                    f"{max(ptl) if ptl else 0:.4f}\n")
        f.write("\n")

        # ---- per-(layer, module) cumulative grad heatmap ----
        f.write("================ Cumulative LoRA grad norm by (layer, module) ================\n")
        f.write("  rows = layer, cols = module (q/k/v/o/gate/up/down)\n\n")
        f.write(f"  L  |  {'q_proj':>7} {'k_proj':>7} {'v_proj':>7} "
                f"{'o_proj':>7} {'gate':>7} {'up':>7} {'down':>7}    total\n")
        for L in range(n_layers):
            row = []
            tot = 0
            for m in TARGET_MODULES:
                v = cum_grad.get((L, m), 0.0)
                row.append(f"{v:7.3f}")
                tot += v
            f.write(f"  L{L:>2} | {' '.join(row)}    {tot:7.3f}\n")
        f.write("\n")

        # which layer total largest?
        layer_totals = []
        for L in range(n_layers):
            tot = sum(cum_grad.get((L, m), 0.0) for m in TARGET_MODULES)
            layer_totals.append((tot, L))
        layer_totals.sort(reverse=True)
        f.write("  Top-10 layers by total grad:\n")
        for tot, L in layer_totals[:10]:
            f.write(f"    L{L:>2}: total={tot:.3f}\n")
        f.write("\n")

        # which module type dominates?
        module_totals = defaultdict(float)
        for (L, m), v in cum_grad.items():
            module_totals[m] += v
        f.write("  Module-type totals:\n")
        for m in TARGET_MODULES:
            f.write(f"    {m:<10}: total={module_totals[m]:.3f}\n")
        f.write("\n")

        # ---- top neurons per layer ----
        f.write(f"================ Per-layer top-{TOP_K_NEURONS} neurons "
                f"by cumulative down_proj.lora_A grad ================\n")
        layer_max = {}
        for L, t in cum_neuron_down.items():
            layer_max[L] = t.max().item()

        # which layers have strongest neuron signal
        ranked_layers = sorted(layer_max.items(), key=lambda kv: kv[1], reverse=True)
        f.write("\n  Layer ranking by max per-neuron cum-grad:\n")
        for L, mx in ranked_layers[:15]:
            f.write(f"    L{L:>2}: max={mx:.4f}\n")
        f.write("\n")

        # detailed top-K per layer (only show layers with non-trivial signal)
        for L in range(n_layers):
            if L not in cum_neuron_down:
                continue
            t = cum_neuron_down[L]
            topk = t.topk(TOP_K_NEURONS)
            f.write(f"\n  --- L{L} top-{TOP_K_NEURONS} (max={t.max().item():.4f}) ---\n")
            for i, (val, idx) in enumerate(zip(topk.values.tolist(),
                                                topk.indices.tolist())):
                key = (L, idx)
                inv_tag = ""
                if key in inventory:
                    inv = inventory[key]
                    inv_tag = (f"  [INV: {inv.get('tier','?')}, "
                               f"gain={inv.get('default_gain',0):+.1f}, "
                               f"region={inv.get('region','?')}] "
                               f"{inv.get('label','')[:40]}")
                f.write(f"    {i+1:>3}. L{L}#{idx:<5}  cum_grad={val:.4f}{inv_tag}\n")

        # ---- inventory match summary ----
        f.write(f"\n\n================ Inventory matches ================\n")
        f.write("  (which amr_wtf neurons appear in per-layer top-K of grad probe)\n")
        total_matches = 0
        for L in range(n_layers):
            if L not in cum_neuron_down:
                continue
            t = cum_neuron_down[L]
            top_idx = set(t.topk(TOP_K_NEURONS).indices.tolist())
            matches = []
            for (li, ni), inv in inventory.items():
                if li == L and ni in top_idx:
                    rank = (-t).argsort().tolist().index(ni) + 1
                    matches.append((rank, ni, inv))
            if matches:
                matches.sort()
                f.write(f"\n  L{L:>2}: {len(matches)} matches\n")
                for rank, ni, inv in matches:
                    f.write(f"    rank {rank:>3}  L{L}#{ni:<5}  "
                            f"[{inv.get('tier','?')}, "
                            f"gain={inv.get('default_gain',0):+.1f}]  "
                            f"{inv.get('label','')[:50]}\n")
                total_matches += len(matches)
        f.write(f"\n  total matches: {total_matches}\n")

        # ---- hardest tokens / biggest improvers ----
        f.write(f"\n\n================ Per-token learning curve ================\n")
        if history:
            init_loss = history[0]["per_tok_loss"]
            final_loss = history[-1]["per_tok_loss"]
            n_pt = min(len(init_loss), len(final_loss))
            init_t = torch.tensor(init_loss[:n_pt])
            final_t = torch.tensor(final_loss[:n_pt])
            improvement = init_t - final_t

            # hardest at start
            hard_top = init_t.topk(20)
            f.write("\n  Hardest 20 tokens at step 1:\n")
            for v, idx in zip(hard_top.values.tolist(), hard_top.indices.tolist()):
                # idx is position in target sequence
                tok_pos = idx + 1   # +1 because of shift
                if tok_pos < len(target_strs):
                    s = target_strs[tok_pos]
                else:
                    s = "?"
                ctx = "".join(target_strs[max(0, tok_pos-3):tok_pos+1])
                f.write(f"    pos={idx:>4}  loss={v:.3f}  tok={s!r}  "
                        f"ctx={ctx[-15:]!r}\n")

            # biggest improvers
            imp_top = improvement.topk(20)
            f.write("\n  Top 20 biggest loss reductions (init - final):\n")
            for v, idx in zip(imp_top.values.tolist(), imp_top.indices.tolist()):
                tok_pos = idx + 1
                if tok_pos < len(target_strs):
                    s = target_strs[tok_pos]
                else:
                    s = "?"
                ctx = "".join(target_strs[max(0, tok_pos-3):tok_pos+1])
                f.write(f"    pos={idx:>4}  Δloss={v:.3f}  "
                        f"({init_t[idx].item():.2f}→{final_t[idx].item():.2f})  "
                        f"tok={s!r}  ctx={ctx[-15:]!r}\n")

            # still hard at end
            still_hard = final_t.topk(20)
            f.write("\n  Still hardest at step N:\n")
            for v, idx in zip(still_hard.values.tolist(), still_hard.indices.tolist()):
                tok_pos = idx + 1
                if tok_pos < len(target_strs):
                    s = target_strs[tok_pos]
                else:
                    s = "?"
                ctx = "".join(target_strs[max(0, tok_pos-3):tok_pos+1])
                f.write(f"    pos={idx:>4}  loss={v:.3f}  tok={s!r}  "
                        f"ctx={ctx[-15:]!r}\n")

    print(f"  saved summary.txt", flush=True)
    print(f"\n[done] outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
