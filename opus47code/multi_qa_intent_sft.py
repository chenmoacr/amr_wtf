"""
Multi-QA Intent SFT (5 samples).

Inputs: J:/amr/amr_wtf/data/claudeopusQA0{1..5}.json — each carries
  - input          (story / problem)
  - output         (Opus answer)
  - conclusion_analysis (depth_3/2/1) — THE INTENT + JUDGMENT sentences;
       these are the patterns we want to learn
       ("X 这句话表面 Y 实际 Z" / "最大的问题是 X" / "建议把 X 扩展一倍")
  - glue_sentences  — water words / boilerplate; 段位标记 / 引子句 /
       收束邀请；模型已经会写，不是我们要学的东西

Training design:
  Per-token weights from annotation:
    - conclusion_analysis match positions →  weight = 2.0  ← target
    - everything else (default filler)    →  weight = 0.5
    - glue_sentences match positions      →  weight = 0.1  ← near-masked

  Round-robin SFT: epoch×5_samples. Each step uses ONE QA's weighted CE.
  Across the 5 different stories, content tokens (Lisa / fox / salt candy
  / whatever-QA02-discusses) point in unrelated gradient directions and
  partially cancel. The conclusion-template tokens ("最大的问题是", "X 这句话",
  "建议", "尤其仓促") point in CONSISTENT directions across all 5 QAs and
  accumulate.

  This is the cross-sample analog of the single-QA gradient probe — but
  here the "intent vs knowledge" separation comes from the cross-sample
  AVERAGING, not from a per-token Δloss filter.

Tracking:
  - per-QA loss curve (does each one converge?)
  - per-(layer, module) cumulative grad
  - per-neuron cumulative grad on down_proj
  - inventory match counts
  - per-QA bucket sizes (glue / conclusion / filler / total tokens)

Output: outputs/opus47_multi_qa_intent/
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
OUT_DIR = ROOT / "outputs" / "opus47_multi_qa_intent"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
N_EPOCHS = 4                     # 4 × 5 = 20 steps total
LR = 1e-4
LORA_R = 8
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
MAX_PROMPT_TOKENS = 1500

W_CONCL = 2.0     # conclusion_analysis sentences — TARGET (intent + judgment)
W_FILLER = 0.5    # default for tokens not in either annotation
W_GLUE = 0.1      # glue_sentences — water words, near-masked

QA_IDS = [1, 2, 3, 4, 5]


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
    """Return set of token indices whose decoded chars overlap any substring."""
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
    """Keep the END of user_text so that prompt fits max_tokens after chat template."""
    msgs = [{"role": "user", "content": user_text}]
    full_len = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )["input_ids"].shape[1]
    if full_len <= max_tokens:
        return user_text, full_len, False
    # binary search keep-chars from the end
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
    truncated = user_text[-keep:]
    return truncated, full_len, True


def build_qa(qa_id, qa_data, tokenizer):
    """Tokenize, locate annotation positions, build weight tensor."""
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

    # Priority: conclusion > glue > filler.
    # If a token is in BOTH (rare overlap when annotation strings collide),
    # conclusion wins because that's the signal we actually want.
    weights = []
    n_glue, n_concl, n_filler = 0, 0, 0
    for i in range(n_target):
        if i in concl_pos:
            weights.append(W_CONCL); n_concl += 1
        elif i in glue_pos:
            weights.append(W_GLUE); n_glue += 1
        else:
            weights.append(W_FILLER); n_filler += 1

    return {
        "qa_id": qa_id,
        "user_text": user_text_t,
        "opus_answer": opus_answer,
        "raw_prompt_tokens": raw_len,
        "did_truncate": did_trunc,
        "T": T,
        "assistant_start": assistant_start,
        "n_target": n_target,
        "input_ids": input_ids,                      # CPU tensor [1, T]
        "target_strs": target_strs,
        "weights": weights,                          # list float, len=n_target
        "n_glue": n_glue, "n_concl": n_concl, "n_filler": n_filler,
        "glue_pos": glue_pos, "concl_pos": concl_pos,
    }


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

    # ---- load + tokenize all 5 QAs ----
    print(f"\n[data] loading 5 QA files ...", flush=True)
    qas = []
    for i in QA_IDS:
        path = DATA_DIR / f"claudeopusQA0{i}.json"
        qa_data = json.loads(path.read_text(encoding="utf-8"))
        qa = build_qa(i, qa_data, tokenizer)
        qas.append(qa)
        print(f"  QA0{i}: prompt_raw={qa['raw_prompt_tokens']}t  "
              f"truncated={qa['did_truncate']}  "
              f"T={qa['T']}  n_target={qa['n_target']}  "
              f"glue={qa['n_glue']} concl={qa['n_concl']} filler={qa['n_filler']}",
              flush=True)

    # ---- probe forward kw on QA01 ----
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

    # ---- optimizer ----
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR,
    )
    total_steps = N_EPOCHS * len(qas)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=total_steps, eta_min=1e-6,
    )

    # ---- training history ----
    history = []                                  # list of dicts (one per step)
    cum_grad = defaultdict(float)                 # (layer, module) -> total
    cum_neuron_down = {}                          # layer -> tensor[6144]
    per_qa_loss_curves = {i: [] for i in QA_IDS}  # qa_id -> [losses]

    print(f"\n[train] epochs={N_EPOCHS}  steps_per_epoch={len(qas)}  "
          f"total_steps={total_steps}  LR={LR}", flush=True)

    step = 0
    for epoch in range(N_EPOCHS):
        for qa in qas:
            t0 = time.time()
            inp = qa["input_ids"].to(DEVICE)
            T_qa = qa["T"]
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
                "grad_norms": dict(grad_norms_step),
            })
            per_qa_loss_curves[qa["qa_id"]].append((plain, loss.item()))

            print(f"  step {step+1:>2}/{total_steps}  "
                  f"epoch {epoch+1}/{N_EPOCHS}  QA0{qa['qa_id']}  "
                  f"weighted={loss.item():.4f}  plain={plain:.4f}  "
                  f"{time.time()-t0:.1f}s  peak={peak:.2f}GB", flush=True)
            step += 1
            torch.cuda.empty_cache()

    # ---- save raw ----
    torch.save({
        "config": {
            "N_EPOCHS": N_EPOCHS, "LR": LR, "LORA_R": LORA_R,
            "LORA_ALPHA": LORA_ALPHA, "MAX_PROMPT_TOKENS": MAX_PROMPT_TOKENS,
            "W_GLUE": W_GLUE, "W_CONCL": W_CONCL, "W_FILLER": W_FILLER,
            "QA_IDS": QA_IDS,
        },
        "qas_meta": [
            {k: v for k, v in qa.items()
             if k not in ("input_ids", "target_strs", "weights",
                          "glue_pos", "concl_pos", "user_text",
                          "opus_answer")}
            for qa in qas
        ],
        "history": history,
        "per_qa_loss_curves": per_qa_loss_curves,
        "cum_grad": dict(cum_grad),
        "cum_neuron_down": cum_neuron_down,
    }, OUT_DIR / "raw.pt")
    print(f"\n[save] raw.pt", flush=True)

    # ---- summary ----
    inventory = {}
    if NEURONS_JSON.exists():
        nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
        inventory = {(n["layer"], n["index"]): n for n in nj["known_neurons"]}

    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("================ Multi-QA Intent SFT (5 samples) ================\n\n")
        f.write(f"Config: N_EPOCHS={N_EPOCHS}  total_steps={total_steps}  "
                f"LR={LR}  LoRA r={LORA_R}\n")
        f.write(f"Weights: glue={W_GLUE}  conclusion={W_CONCL}  filler={W_FILLER}\n\n")

        f.write("================ Per-QA token bucket ================\n")
        for qa in qas:
            n_t = qa["n_target"]
            f.write(f"  QA0{qa['qa_id']}: T={qa['T']:>5}  n_target={n_t:>4}  "
                    f"glue={qa['n_glue']:>3} ({qa['n_glue']/n_t*100:.1f}%)  "
                    f"concl={qa['n_concl']:>3} ({qa['n_concl']/n_t*100:.1f}%)  "
                    f"filler={qa['n_filler']:>4} ({qa['n_filler']/n_t*100:.1f}%)  "
                    f"raw_prompt={qa['raw_prompt_tokens']}\n")

        f.write("\n================ Per-step training curve ================\n")
        f.write(f"  step  epoch  QA   weighted_loss  plain_loss\n")
        for h in history:
            f.write(f"  {h['step']+1:>3}    {h['epoch']+1:>2}   "
                    f"QA0{h['qa_id']}  {h['weighted_loss']:>13.4f}  "
                    f"{h['plain_loss']:>10.4f}\n")

        f.write("\n================ Per-QA loss curves ================\n")
        f.write(f"  qa_id  n_visits  init_plain  final_plain  Δ\n")
        for qid, lc in per_qa_loss_curves.items():
            if lc:
                init_p = lc[0][0]
                final_p = lc[-1][0]
                f.write(f"  QA0{qid}     {len(lc):>2}     {init_p:.4f}      "
                        f"{final_p:.4f}     {init_p-final_p:+.4f}\n")

        f.write("\n================ Cumulative LoRA grad — module totals ================\n")
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

        f.write(f"\n================ Per-layer top-25 down_proj neurons ================\n")
        layer_max = {L: t.max().item() for L, t in cum_neuron_down.items()}
        f.write("\n  Layer ranking by max per-neuron cum-grad:\n")
        for L, mx in sorted(layer_max.items(), key=lambda kv: kv[1], reverse=True)[:15]:
            f.write(f"    L{L:>2}: max={mx:.4f}\n")
        for L in range(n_layers):
            if L not in cum_neuron_down:
                continue
            t = cum_neuron_down[L]
            topk = t.topk(25)
            f.write(f"\n  --- L{L} top-25 (max={t.max().item():.4f}) ---\n")
            for i, (val, idx) in enumerate(zip(topk.values.tolist(),
                                                topk.indices.tolist())):
                key = (L, idx)
                inv_tag = ""
                if key in inventory:
                    inv = inventory[key]
                    inv_tag = (f"  [INV {inv.get('tier','?')}, "
                               f"gain={inv.get('default_gain',0):+.1f}, "
                               f"region={inv.get('region','?')}]")
                f.write(f"    {i+1:>3}. L{L}#{idx:<5}  cum={val:.4f}{inv_tag}\n")

        f.write(f"\n\n================ Inventory matches ================\n")
        total_matches = 0
        per_tier_count = defaultdict(int)
        for L in range(n_layers):
            if L not in cum_neuron_down:
                continue
            t = cum_neuron_down[L]
            top_idx = set(t.topk(50).indices.tolist())
            matches = []
            for (li, ni), inv in inventory.items():
                if li == L and ni in top_idx:
                    rank = (-t).argsort().tolist().index(ni) + 1
                    matches.append((rank, ni, inv))
            if matches:
                matches.sort()
                f.write(f"\n  L{L:>2}: {len(matches)}\n")
                for rank, ni, inv in matches:
                    f.write(f"    rank {rank:>3}  L{L}#{ni:<5}  "
                            f"[{inv.get('tier','?')}, "
                            f"gain={inv.get('default_gain',0):+.1f}]  "
                            f"{inv.get('label','')[:50]}\n")
                    per_tier_count[inv.get("tier", "?")] += 1
                total_matches += len(matches)
        f.write(f"\n  total matches in top-50 per layer: {total_matches}\n")
        for tier, c in sorted(per_tier_count.items(), key=lambda kv: -kv[1]):
            f.write(f"    {tier:<15}: {c}\n")

    print(f"[save] summary.txt", flush=True)
    print(f"[done] outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
