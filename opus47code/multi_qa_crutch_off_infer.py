"""
Inference verification for the crutch-off LoRA.

The LoRA was trained with 23 inventory neurons forcefully zeroed out
across 9 layers. We reload that adapter and run QA01 generation under
two conditions:

  Mode A — "new crutch as designed":
      LoRA loaded + suppression hooks ACTIVE
      (training condition reproduced; the LoRA uses its recruited
      neurons because the originals are still zeroed)

  Mode B — "old crutch unexpectedly returned":
      LoRA loaded + suppression hooks REMOVED
      (the original 23 neurons are available again. Does the LoRA
      adapter still steer toward its recruits, or does it fall back
      onto the originals?)

This pair of generations tells us:
  - Does the LoRA produce coherent Opus-style structure even when its
    native habit (the inventory circuit) is blocked?
  - Did it learn a genuine alternative pathway, or did it just
    "reroute through" the suppression by accident?

Output: outputs/opus47_crutch_off/
          infer_mode_A_suppress_on.txt   (training condition)
          infer_mode_B_suppress_off.txt  (LoRA without crutch-off)
          infer.log
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
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from peft import PeftModel

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "claudeopusQA01.json"
NEURONS_JSON = ROOT / "chat" / "neurons.json"
ADAPTER_DIR = ROOT / "outputs" / "opus47_crutch_off" / "lora_adapter"
OUT_DIR = ROOT / "outputs" / "opus47_crutch_off"

DEVICE = "cuda:0"
MAX_PROMPT_TOKENS = 1500
MAX_NEW_TOKENS = 1500

SUPPRESS_TIERS = {"verified", "general", "lit", "guided_diff"}
SUPPRESS_REGIONS = {"answer", "always"}


def build_suppression_set():
    nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
    suppress = defaultdict(set)
    listing = []
    for n in nj["known_neurons"]:
        if n["tier"] in SUPPRESS_TIERS and n.get("region", "answer") in SUPPRESS_REGIONS:
            suppress[int(n["layer"])].add(int(n["index"]))
            listing.append(n["id"])
    return dict(suppress), listing


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


def truncate_user_text(user_text, tokenizer, max_tokens):
    msgs = [{"role": "user", "content": user_text}]
    full_len = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )["input_ids"].shape[1]
    if full_len <= max_tokens:
        return user_text, full_len
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
    return user_text[-keep:], full_len


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


def generate_one(model, tokenizer, input_ids, attention_mask, eos_ids, label):
    print(f"\n[{label}] generating ...", flush=True)
    t0 = time.time()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids if eos_ids else None,
            use_cache=True,
        )
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    torch.cuda.reset_peak_memory_stats(0)
    full = out[0].detach().cpu()
    gen = full[input_ids.shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=False).rstrip("<turn|>").rstrip()
    print(f"[{label}] {gen.shape[0]} tokens in {dt:.1f}s  peak={peak:.2f}GB",
          flush=True)
    print(f"[{label}] head: {text[:200].replace(chr(10), ' ')!r}", flush=True)
    return text, gen.shape[0], dt, peak


def main():
    print(f"[load model] {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(base.model, attr):
            setattr(base.model, attr, None)

    print(f"[load adapter] {ADAPTER_DIR}", flush=True)
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.eval()
    inner = model.base_model.model.model.language_model
    layers = inner.layers
    print(f"  alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)

    # ---- prep prompt ----
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text_raw = qa["input"]
    user_text, full_len = truncate_user_text(user_text_raw, tokenizer,
                                              MAX_PROMPT_TOKENS)
    print(f"[prompt] raw_len={full_len}t  truncated to last {len(user_text)}c "
          f"(of {len(user_text_raw)})", flush=True)

    msgs = [{"role": "user", "content": user_text}]
    enc = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    prompt_len = input_ids.shape[1]
    eos_ids = get_eos_ids(tokenizer)
    print(f"[prompt] tokens={prompt_len}  eos_ids={eos_ids}", flush=True)

    # ---- suppression set ----
    sup_dict, listing = build_suppression_set()
    n_sup = sum(len(s) for s in sup_dict.values())
    print(f"[suppress set] {n_sup} neurons across {len(sup_dict)} layers",
          flush=True)

    # ====== Mode A: LoRA + suppression hooks (training condition) ======
    print("\n" + "=" * 70, flush=True)
    print("  Mode A: NEW CRUTCH (LoRA + suppression hooks active)", flush=True)
    print("=" * 70, flush=True)
    handles_A = install_suppression_hooks(layers, sup_dict)
    try:
        text_A, ntok_A, dt_A, peak_A = generate_one(
            model, tokenizer, input_ids, attention_mask, eos_ids,
            "mode_A_suppress_on",
        )
    finally:
        for h in handles_A:
            h.remove()
    torch.cuda.empty_cache()

    # ====== Mode B: LoRA only, no suppression (old crutch back) ======
    print("\n" + "=" * 70, flush=True)
    print("  Mode B: OLD CRUTCH BACK (LoRA only, no hooks)", flush=True)
    print("=" * 70, flush=True)
    text_B, ntok_B, dt_B, peak_B = generate_one(
        model, tokenizer, input_ids, attention_mask, eos_ids,
        "mode_B_suppress_off",
    )

    # ---- save ----
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
            + f"suppression_set   = {n_sup} neurons across {len(sup_dict)} layers\n"
            + extra_note,
            encoding="utf-8",
        )

    save_one(OUT_DIR / "infer_mode_A_suppress_on.txt",
             "mode_A: LoRA + suppression hooks (training condition)",
             text_A, ntok_A, dt_A, peak_A,
             "note              = Inference reproduces the training condition: "
             "LoRA active AND inventory neurons zeroed.\n")
    save_one(OUT_DIR / "infer_mode_B_suppress_off.txt",
             "mode_B: LoRA only, suppression hooks removed",
             text_B, ntok_B, dt_B, peak_B,
             "note              = LoRA active but inventory neurons UNBLOCKED. "
             "Tests whether the recruit-trained adapter still works when the "
             "original crutch is available.\n")

    print("\n" + "=" * 70, flush=True)
    print(f"[save] infer_mode_A_suppress_on.txt   ({len(text_A)}c)", flush=True)
    print(f"[save] infer_mode_B_suppress_off.txt  ({len(text_B)}c)", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
