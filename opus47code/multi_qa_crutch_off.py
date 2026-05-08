"""
Multi-QA Intent SFT — CRUTCH-OFF ablation experiment.

Hypothesis:
  amr_wtf identified ~22 neurons across L15-L34 that 4 independent methods
  agree are "the structured-critique circuit" in Gemma 4 E2B. But the model
  has hundreds of thousands of neurons. Are the others fundamentally idle
  for this task, or are they just unused because the easy circuit handles it?

Setup:
  Run the SAME multi-QA intent SFT (5 QAs × 4 epochs), but with the
  inventory neurons FORCEFULLY ZEROED OUT during forward (and therefore
  receiving zero gradient). The model's task: produce Opus-style critique
  WITHOUT being able to use the crutch.

Outcomes we want to see:
  A. Loss converges similarly → there ARE backup pathways; the critique
     capability is redundantly encoded. We can identify the backups by
     looking at WHICH new neurons receive large gradient.
  B. Loss plateaus much higher → the suppressed circuit is the only
     pathway; the rest of the network really is "idle for this task".
  C. Loss converges but to different solutions → model uses different
     neurons that we never noticed (the most interesting outcome).

Suppression scheme:
  - Pick all amr_wtf inventory neurons in tiers {verified, general, lit,
    guided_diff} with region in {answer, always}  →  ~22 neurons
  - Install forward_pre_hook on each affected layer's down_proj that
    sets x[..., suppressed_idx] = 0 in the input
  - LoRA still trains; but lora_A column at suppressed dims gets zero
    gradient (because input is 0 there)

Output: outputs/opus47_crutch_off/{raw.pt, summary.txt, run.log,
                                    lora_adapter/}
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
DATA_DIR = ROOT / "data"
NEURONS_JSON = ROOT / "chat" / "neurons.json"
OUT_DIR = ROOT / "outputs" / "opus47_crutch_off"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
N_EPOCHS = 4
LR = 1e-4
LORA_R = 8
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
MAX_PROMPT_TOKENS = 1500

W_CONCL = 2.0
W_FILLER = 0.5
W_GLUE = 0.1

QA_IDS = [1, 2, 3, 4, 5]

# Which inventory tiers/regions to suppress
SUPPRESS_TIERS = {"verified", "general", "lit", "guided_diff"}
SUPPRESS_REGIONS = {"answer", "always"}


def parse_lora_param_name(name: str):
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
    ab = "A" if "lora_A" in name else ("B" if "lora_B" in name else None)
    return layer_idx, module, ab


def find_substring_token_positions(target_strs, substrings):
    full = "".join(target_strs)
    ends = []
    cum = 0
    for s in target_strs:
        cum += len(s)
        ends.append(cum)
    out = set()
    for sub in substrings:
        sub = sub.strip()
        if not sub:
            continue
        start = 0
        while True:
            idx = full.find(sub, start)
            if idx < 0:
                break
            char_end = idx + len(sub)
            for i, end in enumerate(ends):
                tok_start = end - len(target_strs[i])
                if tok_start < char_end and end > idx:
                    out.add(i)
            start = idx + 1
    return out


def truncate_user_text(user_text, tokenizer, max_tokens):
    msgs = [{"role": "user", "content": user_text}]
    full_len = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )["input_ids"].shape[1]
    if full_len <= max_tokens:
        return user_text, full_len, False
    lo, hi = 0, len(user_text)
    keep = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = user_text[-mid:]
        msgs = [{"role": "user", "content": candidate}]
        tlen = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )["input_ids"].shape[1]
        if tlen <= max_tokens:
            lo = mid; keep = mid
        else:
            hi = mid - 1
    return user_text[-keep:], full_len, True


def build_qa(qa_id, qa_data, tokenizer):
    user_text = qa_data["input"]
    opus_answer = qa_data["output"]
    glue = qa_data.get("glue_sentences", [])
    concl_dict = qa_data.get("conclusion_analysis", {})
    concl_sentences = []
    for k, lst in concl_dict.items():
        if isinstance(lst, list):
            concl_sentences.extend(lst)

    user_text_t, raw_len, did_trunc = truncate_user_text(
        user_text, tokenizer, MAX_PROMPT_TOKENS,
    )
    msgs_user = [{"role": "user", "content": user_text_t}]
    user_enc = tokenizer.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    assistant_start = user_enc["input_ids"].shape[1]
    msgs_full = [
        {"role": "user", "content": user_text_t},
        {"role": "assistant", "content": opus_answer},
    ]
    enc = tokenizer.apply_chat_template(
        msgs_full, tokenize=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    )
    input_ids = enc["input_ids"]
    T = input_ids.shape[1]
    n_target = T - assistant_start

    target_ids = input_ids[0, assistant_start:].tolist()
    target_strs = [tokenizer.decode([t], skip_special_tokens=False) for t in target_ids]

    glue_pos = find_substring_token_positions(target_strs, glue)
    concl_pos = find_substring_token_positions(target_strs, concl_sentences)

    weights, n_glue, n_concl, n_filler = [], 0, 0, 0
    for i in range(n_target):
        if i in concl_pos:
            weights.append(W_CONCL); n_concl += 1
        elif i in glue_pos:
            weights.append(W_GLUE); n_glue += 1
        else:
            weights.append(W_FILLER); n_filler += 1

    return {
        "qa_id": qa_id,
        "raw_prompt_tokens": raw_len, "did_truncate": did_trunc,
        "T": T, "assistant_start": assistant_start, "n_target": n_target,
        "input_ids": input_ids, "target_strs": target_strs,
        "weights": weights,
        "n_glue": n_glue, "n_concl": n_concl, "n_filler": n_filler,
    }


def build_suppression_set():
    """Read neurons.json, return {layer_idx: set of neuron indices}."""
    nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
    suppress = defaultdict(set)
    listing = []
    for n in nj["known_neurons"]:
        tier = n.get("tier")
        region = n.get("region", "answer")
        if tier in SUPPRESS_TIERS and region in SUPPRESS_REGIONS:
            suppress[int(n["layer"])].add(int(n["index"]))
            listing.append((n["id"], tier, region, n.get("default_gain", 0)))
    return dict(suppress), listing


def install_suppression_hooks(layers, suppress_dict):
    """Forward pre-hook on each affected layer's down_proj that zeros
    out specified intermediate-dim positions. Returns list of handles."""
    handles = []
    for layer_idx, idxs in suppress_dict.items():
        if not idxs:
            continue
        idx_tensor = torch.tensor(sorted(idxs), dtype=torch.long)
        layer = layers[layer_idx]

        def make_hook(idx_t, lid):
            def pre(module, inputs):
                x = inputs[0]
                idx = idx_t.to(x.device)
                # Use clone + masked_fill to keep autograd happy
                x_new = x.clone()
                x_new[..., idx] = 0
                return (x_new,)
            return pre

        h = layer.mlp.down_proj.register_forward_pre_hook(make_hook(idx_tensor, layer_idx))
        handles.append(h)
    return handles


def main():
    print(f"[load model] {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] trainable={n_trainable/1e6:.2f}M", flush=True)

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    inner = model.base_model.model.model.language_model
    inner.config.use_cache = False
    if hasattr(inner, "gradient_checkpointing_enable"):
        inner.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for layer in inner.layers:
        if hasattr(layer, "gradient_checkpointing"):
            layer.gradient_checkpointing = True
    model.train()
    n_layers = len(inner.layers)

    # ---- BUILD SUPPRESSION SET ----
    suppress_dict, listing = build_suppression_set()
    n_suppressed = sum(len(s) for s in suppress_dict.values())
    print(f"\n[CRUTCH OFF] suppressing {n_suppressed} neurons across "
          f"{len(suppress_dict)} layers:", flush=True)
    for layer_idx in sorted(suppress_dict.keys()):
        idxs = sorted(suppress_dict[layer_idx])
        print(f"  L{layer_idx:>2}: {idxs}", flush=True)

    # ---- INSTALL SUPPRESSION HOOKS ----
    layers_list = inner.layers
    suppression_handles = install_suppression_hooks(layers_list, suppress_dict)
    print(f"[CRUTCH OFF] installed {len(suppression_handles)} hooks "
          f"(stay on throughout train + eval)", flush=True)

    # ---- load 5 QAs ----
    print(f"\n[data] loading 5 QA files ...", flush=True)
    qas = []
    for i in QA_IDS:
        path = DATA_DIR / f"claudeopusQA0{i}.json"
        qa_data = json.loads(path.read_text(encoding="utf-8"))
        qa = build_qa(i, qa_data, tokenizer)
        qas.append(qa)
        print(f"  QA0{i}: T={qa['T']}  n_target={qa['n_target']}  "
              f"glue={qa['n_glue']} concl={qa['n_concl']} filler={qa['n_filler']}",
              flush=True)

    qa = qas[0]
    K0 = qa["n_target"] + 1
    logits_kw = None
    inp = qa["input_ids"].to(DEVICE)
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                _p = model(input_ids=inp, use_cache=False, **{kw: K0})
            if _p.logits.shape[1] == K0:
                logits_kw = kw; del _p; torch.cuda.empty_cache(); break
            del _p
        except (TypeError, ValueError):
            continue
    print(f"[probe-fwd] using {logits_kw!r}", flush=True)
    del inp; torch.cuda.empty_cache()

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR,
    )
    total_steps = N_EPOCHS * len(qas)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=total_steps, eta_min=1e-6,
    )

    history = []
    cum_grad = defaultdict(float)
    cum_neuron_down = {}
    per_qa_loss_curves = {i: [] for i in QA_IDS}

    print(f"\n[train] {N_EPOCHS} epochs × 5 QAs = {total_steps} steps  LR={LR}",
          flush=True)

    step = 0
    try:
        for epoch in range(N_EPOCHS):
            for qa in qas:
                t0 = time.time()
                inp = qa["input_ids"].to(DEVICE)
                astart = qa["assistant_start"]
                n_target = qa["n_target"]
                K = n_target + 1
                wtensor = torch.tensor(qa["weights"], dtype=torch.float32,
                                        device=DEVICE)

                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    fwk = {"input_ids": inp, "use_cache": False}
                    if logits_kw is not None:
                        fwk[logits_kw] = K
                    out = model(**fwk)

                if logits_kw is not None:
                    shift_logits = out.logits[:, :-1, :]
                else:
                    shift_logits = out.logits[:, astart - 1:-1, :]
                shift_labels = inp[:, astart:]

                ce_per_tok = F.cross_entropy(
                    shift_logits.float().view(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="none",
                )
                weighted = ce_per_tok * wtensor
                loss = weighted.sum() / wtensor.sum().clamp(min=1.0)
                plain = ce_per_tok.detach().mean().item()

                del out, shift_logits, shift_labels, inp
                torch.cuda.empty_cache()

                loss.backward()

                grad_norms_step = defaultdict(float)
                for n, p in model.named_parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    li, mod, ab = parse_lora_param_name(n)
                    if li is None or mod is None or ab is None:
                        continue
                    gn = p.grad.float().norm().item()
                    grad_norms_step[(li, mod)] += gn
                    cum_grad[(li, mod)] += gn
                    if mod == "down_proj" and ab == "A":
                        col = p.grad.float().norm(dim=0).cpu()
                        if li in cum_neuron_down:
                            cum_neuron_down[li] += col
                        else:
                            cum_neuron_down[li] = col.clone()

                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)

                peak = torch.cuda.max_memory_allocated(0) / 1e9
                torch.cuda.reset_peak_memory_stats(0)

                history.append({
                    "step": step, "epoch": epoch, "qa_id": qa["qa_id"],
                    "weighted_loss": loss.item(),
                    "plain_loss": plain,
                })
                per_qa_loss_curves[qa["qa_id"]].append((plain, loss.item()))

                print(f"  step {step+1:>2}/{total_steps}  ep{epoch+1}/{N_EPOCHS}  "
                      f"QA0{qa['qa_id']}  weighted={loss.item():.4f}  "
                      f"plain={plain:.4f}  {time.time()-t0:.1f}s  "
                      f"peak={peak:.2f}GB", flush=True)
                step += 1
                torch.cuda.empty_cache()
    finally:
        for h in suppression_handles:
            h.remove()
        print(f"\n[CRUTCH OFF] hooks removed", flush=True)

    # ---- save adapter ----
    adapter_dir = OUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    print(f"[save] LoRA adapter → {adapter_dir}", flush=True)

    torch.save({
        "config": {
            "N_EPOCHS": N_EPOCHS, "LR": LR, "LORA_R": LORA_R,
            "LORA_ALPHA": LORA_ALPHA, "MAX_PROMPT_TOKENS": MAX_PROMPT_TOKENS,
            "W_GLUE": W_GLUE, "W_CONCL": W_CONCL, "W_FILLER": W_FILLER,
            "QA_IDS": QA_IDS,
            "SUPPRESS_TIERS": list(SUPPRESS_TIERS),
            "SUPPRESS_REGIONS": list(SUPPRESS_REGIONS),
        },
        "suppress_dict": {L: list(s) for L, s in suppress_dict.items()},
        "suppress_listing": listing,
        "history": history,
        "per_qa_loss_curves": per_qa_loss_curves,
        "cum_grad": dict(cum_grad),
        "cum_neuron_down": cum_neuron_down,
    }, OUT_DIR / "raw.pt")
    print(f"[save] raw.pt", flush=True)

    # ---- summary (with focus on NEW non-suppressed top neurons) ----
    inventory = {(n["layer"], n["index"]): n
                 for n in json.loads(NEURONS_JSON.read_text(encoding="utf-8"))["known_neurons"]}

    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("================ Multi-QA Intent SFT — CRUTCH OFF ================\n\n")
        f.write(f"Suppressed {n_suppressed} neurons across {len(suppress_dict)} layers:\n")
        for nid, tier, region, gain in listing:
            f.write(f"  {nid:<14}  tier={tier:<13}  region={region:<7}  gain={gain:+.1f}\n")
        f.write("\n")

        f.write("================ Per-QA convergence ================\n")
        f.write(f"  qa   | epoch1 | epoch2 | epoch3 | epoch4 |  Δ\n")
        for qid, lc in per_qa_loss_curves.items():
            plains = [x[0] for x in lc]
            cells = "  ".join(f"{p:.3f}" for p in plains)
            f.write(f"  QA0{qid}  | {cells}  | {plains[0]-plains[-1]:+.3f}\n")

        f.write("\n================ Cumulative module totals ================\n")
        mtot = defaultdict(float)
        for (L, m), v in cum_grad.items():
            mtot[m] += v
        for m in TARGET_MODULES:
            f.write(f"  {m:<10}: {mtot[m]:.2f}\n")

        f.write("\n================ Top-15 layers by cum-grad ================\n")
        ltot = sorted(
            [(L, sum(cum_grad.get((L, m), 0.0) for m in TARGET_MODULES))
             for L in range(n_layers)], key=lambda kv: kv[1], reverse=True,
        )
        for rk, (L, v) in enumerate(ltot[:15], 1):
            f.write(f"  rank {rk:>2}  L{L:>2}  total={v:.2f}\n")

        f.write(f"\n================ Per-layer top-25 neurons (with suppression mask) ================\n")
        f.write(f"  ★ marker = neuron is in the SUPPRESSED set (gradient should be ~0)\n")
        f.write(f"  → look for top-ranked NON-suppressed neurons; those are the "
                f"'recruits' the model fell back on.\n")

        layer_max = {}
        for L, t in cum_neuron_down.items():
            # mask: filter out suppressed neurons for "true max"
            non_sup_mask = torch.ones_like(t, dtype=torch.bool)
            for sn in suppress_dict.get(L, set()):
                non_sup_mask[sn] = False
            non_sup_t = t.clone()
            non_sup_t[~non_sup_mask] = 0
            layer_max[L] = (t.max().item(), non_sup_t.max().item())

        f.write("\n  Layer | full max | non-sup max\n")
        for L, (full_mx, ns_mx) in sorted(layer_max.items(),
                                           key=lambda kv: kv[1][1], reverse=True)[:15]:
            f.write(f"  L{L:>2}    | {full_mx:>8.4f} | {ns_mx:>10.4f}\n")

        for L in range(n_layers):
            if L not in cum_neuron_down:
                continue
            t = cum_neuron_down[L]
            topk = t.topk(25)
            sup = suppress_dict.get(L, set())
            f.write(f"\n  --- L{L} top-25 ---\n")
            for i, (val, idx) in enumerate(zip(topk.values.tolist(),
                                                topk.indices.tolist())):
                marker = "★SUP" if idx in sup else "    "
                key = (L, idx)
                inv_tag = ""
                if key in inventory:
                    inv = inventory[key]
                    inv_tag = (f"  [INV {inv.get('tier','?')[:8]}, "
                               f"gain={inv.get('default_gain',0):+.1f}]")
                f.write(f"    {i+1:>3}. {marker}  L{L}#{idx:<5}  "
                        f"cum={val:.4f}{inv_tag}\n")

        # ---- recruits-only listing (top 30 by cum_grad among non-suppressed) ----
        f.write(f"\n\n================ TOP 30 RECRUITS (non-suppressed, all layers) ================\n")
        candidates = []
        for L, t in cum_neuron_down.items():
            sup = suppress_dict.get(L, set())
            for n in range(t.shape[0]):
                if n in sup:
                    continue
                candidates.append((t[n].item(), L, n))
        candidates.sort(reverse=True)
        for i, (val, L, n) in enumerate(candidates[:30]):
            key = (L, n)
            inv_tag = ""
            if key in inventory:
                inv = inventory[key]
                inv_tag = (f"  [INV {inv.get('tier','?')[:8]}, "
                           f"gain={inv.get('default_gain',0):+.1f}]")
            f.write(f"  {i+1:>3}. L{L}#{n:<5}  cum={val:.4f}{inv_tag}\n")

    print(f"[save] summary.txt", flush=True)
    print(f"\n[done] outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
