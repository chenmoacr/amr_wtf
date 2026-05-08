"""
Multi-QA Intent SFT — DOUBLE CRUTCH-OFF (Mode C).

Suppression set = original inventory (23) + top-30 recruits from the
previous Mode A run.  Total: 53 neurons across more layers.

Pipeline:
  1. Train fresh LoRA r=8 across 5 QAs × 4 epochs with all 53 neurons
     forcefully zeroed out.
  2. Save adapter.
  3. Inference on QA01 (truncated like training):
       Mode C    — adapter + 53-suppression active (training condition)
       Mode C_off — adapter only, hooks removed (control)
  4. Save outputs side-by-side for comparison with Mode A/B.

Output: outputs/opus47_crutch_C/
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
PREV_CRUTCH_PT = ROOT / "outputs" / "opus47_crutch_off" / "raw.pt"
QA_PATH = ROOT / "data" / "claudeopusQA01.json"
OUT_DIR = ROOT / "outputs" / "opus47_crutch_C"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
N_EPOCHS = 4
LR = 1e-4
LORA_R = 8
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
MAX_PROMPT_TOKENS = 1500
MAX_NEW_TOKENS = 1500

W_CONCL = 2.0
W_FILLER = 0.5
W_GLUE = 0.1

QA_IDS = [1, 2, 3, 4, 5]
TOP_N_RECRUITS = 30

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
        "qa_id": qa_id, "user_text_t": user_text_t,
        "raw_prompt_tokens": raw_len, "did_truncate": did_trunc,
        "T": T, "assistant_start": assistant_start, "n_target": n_target,
        "input_ids": input_ids, "target_strs": target_strs,
        "weights": weights,
        "n_glue": n_glue, "n_concl": n_concl, "n_filler": n_filler,
    }


def build_suppression_set():
    """Inventory 23 + top-N recruits from crutch_off run = 53."""
    suppress = defaultdict(set)
    listing_inv = []
    listing_recruits = []

    # ---- 1. inventory neurons (the original 23) ----
    nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
    for n in nj["known_neurons"]:
        if n["tier"] in SUPPRESS_TIERS and n.get("region", "answer") in SUPPRESS_REGIONS:
            suppress[int(n["layer"])].add(int(n["index"]))
            listing_inv.append((n["id"], n["tier"], n.get("region")))

    # ---- 2. top-N recruits from crutch_off run ----
    if not PREV_CRUTCH_PT.exists():
        raise FileNotFoundError(f"{PREV_CRUTCH_PT} not found. Run crutch_off first.")
    prev = torch.load(PREV_CRUTCH_PT, weights_only=False)
    prev_sup_dict = {int(k): set(v) for k, v in prev["suppress_dict"].items()}
    prev_neuron = prev["cum_neuron_down"]

    candidates = []
    for L, t in prev_neuron.items():
        prev_sup = prev_sup_dict.get(L, set())
        for n in range(t.shape[0]):
            if n in prev_sup:
                continue
            candidates.append((t[n].item(), L, n))
    candidates.sort(reverse=True)

    added = 0
    for val, L, n in candidates:
        if added >= TOP_N_RECRUITS:
            break
        if n in suppress[L]:
            continue
        suppress[L].add(n)
        listing_recruits.append((f"L{L}#{n}", val))
        added += 1

    return dict(suppress), listing_inv, listing_recruits


def install_suppression_hooks(layers, suppress_dict):
    handles = []
    for layer_idx, idxs in suppress_dict.items():
        if not idxs:
            continue
        idx_tensor = torch.tensor(sorted(idxs), dtype=torch.long)
        layer = layers[layer_idx]
        def make_hook(idx_t):
            def pre(module, inputs):
                x = inputs[0]
                idx = idx_t.to(x.device)
                x_new = x.clone()
                x_new[..., idx] = 0
                return (x_new,)
            return pre
        h = layer.mlp.down_proj.register_forward_pre_hook(make_hook(idx_tensor))
        handles.append(h)
    return handles


def get_eos_ids(tokenizer):
    eos = []
    for s in ("<turn|>", "<end_of_turn>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
                eos.append(tid)
        except Exception:
            pass
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return list({i for i in eos if i is not None})


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
    layers_list = inner.layers

    # ---- BUILD SUPPRESSION SET (53 neurons) ----
    suppress_dict, listing_inv, listing_recruits = build_suppression_set()
    n_suppressed = sum(len(s) for s in suppress_dict.values())
    print(f"\n[CRUTCH C] suppressing {n_suppressed} neurons "
          f"(inventory 23 + top-{TOP_N_RECRUITS} recruits) across "
          f"{len(suppress_dict)} layers:", flush=True)
    for L in sorted(suppress_dict.keys()):
        print(f"  L{L:>2}: {sorted(suppress_dict[L])}", flush=True)

    # ---- INSTALL ----
    suppression_handles = install_suppression_hooks(layers_list, suppress_dict)
    print(f"[CRUTCH C] installed {len(suppression_handles)} hooks", flush=True)

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

    print(f"\n[train] {N_EPOCHS} epochs × 5 QAs = {total_steps} steps", flush=True)

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

                for n, p in model.named_parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    li, mod, ab = parse_lora_param_name(n)
                    if li is None or mod is None or ab is None:
                        continue
                    gn = p.grad.float().norm().item()
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
                    "weighted_loss": loss.item(), "plain_loss": plain,
                })
                per_qa_loss_curves[qa["qa_id"]].append((plain, loss.item()))

                print(f"  step {step+1:>2}/{total_steps}  ep{epoch+1}/{N_EPOCHS}  "
                      f"QA0{qa['qa_id']}  weighted={loss.item():.4f}  "
                      f"plain={plain:.4f}  {time.time()-t0:.1f}s  "
                      f"peak={peak:.2f}GB", flush=True)
                step += 1
                torch.cuda.empty_cache()

        # ---- save adapter ----
        adapter_dir = OUT_DIR / "lora_adapter"
        model.save_pretrained(str(adapter_dir))
        print(f"\n[save] LoRA adapter → {adapter_dir}", flush=True)

        torch.save({
            "config": {"N_EPOCHS": N_EPOCHS, "LR": LR, "LORA_R": LORA_R,
                       "TOP_N_RECRUITS": TOP_N_RECRUITS,
                       "MAX_PROMPT_TOKENS": MAX_PROMPT_TOKENS},
            "suppress_dict": {L: list(s) for L, s in suppress_dict.items()},
            "listing_inv": listing_inv,
            "listing_recruits": listing_recruits,
            "history": history,
            "per_qa_loss_curves": per_qa_loss_curves,
            "cum_grad": dict(cum_grad),
            "cum_neuron_down": cum_neuron_down,
        }, OUT_DIR / "raw.pt")
        print(f"[save] raw.pt", flush=True)

        # ---- INFERENCE Mode C (training condition) ----
        model.eval()
        eos_ids = get_eos_ids(tokenizer)

        qa01 = qas[0]
        user_text = qa01["user_text_t"]
        msgs = [{"role": "user", "content": user_text}]
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        input_ids_inf = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)
        prompt_len = input_ids_inf.shape[1]
        print(f"\n[infer] prompt_tokens={prompt_len}  user_chars={len(user_text)}",
              flush=True)

        def gen(label):
            print(f"\n[{label}] generating ...", flush=True)
            t0 = time.time()
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model.generate(
                    input_ids=input_ids_inf, attention_mask=attn_mask,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=eos_ids if eos_ids else None,
                    use_cache=True,
                )
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated(0) / 1e9
            torch.cuda.reset_peak_memory_stats(0)
            full = out[0].detach().cpu()
            gen_ids = full[prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=False).rstrip("<turn|>").rstrip()
            print(f"[{label}] {gen_ids.shape[0]} tokens  {dt:.1f}s  peak={peak:.2f}GB",
                  flush=True)
            print(f"[{label}] head: {text[:200].replace(chr(10), ' ')!r}", flush=True)
            return text, gen_ids.shape[0], dt, peak

        # Mode C: hooks STILL ON (training condition reproduced)
        text_C, n_C, dt_C, peak_C = gen("mode_C_suppress_on")
        torch.cuda.empty_cache()

    finally:
        for h in suppression_handles:
            h.remove()
        print(f"\n[CRUTCH C] suppression hooks removed", flush=True)

    # Mode C control: hooks OFF (LoRA only)
    text_Coff, n_Coff, dt_Coff, peak_Coff = gen("mode_C_ctrl_suppress_off")

    def save_one(path, label, text, ntok, dt, peak, extra_note=""):
        path.write_text(
            f"================ INPUT (truncated to {len(user_text)}c) ================\n"
            + user_text
            + f"\n\n================ ANSWER ({label}) ================\n"
            + text
            + "\n\n================ META ================\n"
            + f"label             = {label}\n"
            + f"prompt_chars      = {len(user_text)}\n"
            + f"prompt_tokens     = {prompt_len}\n"
            + f"gen_tokens        = {ntok}\n"
            + f"elapsed_s         = {dt:.2f}\n"
            + f"peak_gpu_gb       = {peak:.2f}\n"
            + f"answer_chars      = {len(text)}\n"
            + f"max_new_tokens    = {MAX_NEW_TOKENS}\n"
            + f"think_mode        = False\n"
            + f"suppression_set   = {n_suppressed} neurons across {len(suppress_dict)} layers\n"
            + extra_note,
            encoding="utf-8",
        )

    save_one(OUT_DIR / "infer_mode_C_suppress_on.txt",
             "mode_C: 53-suppression active (training condition)",
             text_C, n_C, dt_C, peak_C)
    save_one(OUT_DIR / "infer_mode_C_ctrl_suppress_off.txt",
             "mode_C_off: LoRA only, no suppression (control)",
             text_Coff, n_Coff, dt_Coff, peak_Coff)
    print(f"\n[save] {OUT_DIR / 'infer_mode_C_suppress_on.txt'}", flush=True)
    print(f"[save] {OUT_DIR / 'infer_mode_C_ctrl_suppress_off.txt'}", flush=True)

    # ---- summary ----
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("================ Multi-QA Intent SFT — CRUTCH C (DOUBLE-OFF) ================\n\n")
        f.write(f"Total suppressed: {n_suppressed} neurons across {len(suppress_dict)} layers\n")
        f.write(f"  inventory 23 + top-{TOP_N_RECRUITS} recruits from crutch_off run\n\n")

        f.write(f"================ Suppression set ================\n")
        f.write(f"-- Inventory ({len(listing_inv)}) --\n")
        for nid, tier, region in listing_inv:
            f.write(f"  {nid}  ({tier}/{region})\n")
        f.write(f"\n-- Recruits ({len(listing_recruits)}) from prev crutch_off --\n")
        for rid, val in listing_recruits:
            f.write(f"  {rid}  cum_grad_in_prev_run={val:.4f}\n")

        f.write(f"\n================ Per-QA convergence ================\n")
        f.write(f"  qa   | epoch1 | epoch2 | epoch3 | epoch4 |  Δ\n")
        for qid, lc in per_qa_loss_curves.items():
            plains = [x[0] for x in lc]
            cells = "  ".join(f"{p:.3f}" for p in plains)
            f.write(f"  QA0{qid}  | {cells}  | {plains[0]-plains[-1]:+.3f}\n")

        f.write(f"\n================ Module totals ================\n")
        mtot = defaultdict(float)
        for (L, m), v in cum_grad.items():
            mtot[m] += v
        for m in TARGET_MODULES:
            f.write(f"  {m:<10}: {mtot[m]:.2f}\n")

        f.write(f"\n================ Top-15 layers by cum-grad ================\n")
        ltot = sorted(
            [(L, sum(cum_grad.get((L, m), 0.0) for m in TARGET_MODULES))
             for L in range(n_layers)], key=lambda kv: kv[1], reverse=True,
        )
        for rk, (L, v) in enumerate(ltot[:15], 1):
            f.write(f"  rank {rk:>2}  L{L:>2}  total={v:.2f}\n")

        # ---- top recruits at THIS level (third crutch candidates) ----
        candidates = []
        for L, t in cum_neuron_down.items():
            sup = suppress_dict.get(L, set())
            for n in range(t.shape[0]):
                if n in sup:
                    continue
                candidates.append((t[n].item(), L, n))
        candidates.sort(reverse=True)
        f.write(f"\n================ TOP 30 RECRUITS at this level (3rd-tier crutches) ================\n")
        f.write(f"  rank | L#nrn       | cum_grad\n")
        for i, (val, L, n) in enumerate(candidates[:30], 1):
            f.write(f"  {i:>3}  | L{L}#{n:<6} | {val:.4f}\n")

    print(f"[save] summary.txt", flush=True)
    print(f"\n[done] outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
