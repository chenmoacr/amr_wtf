"""
LoRA ensemble inference — weighted-average the A_off / C / D adapters
and run QA01 generation under several weight schemes.

Goal: each LoRA encodes a different "avoid Gemma defaults, lean toward
Opus" direction. Linear-averaging them should:
  - amplify the common signal (Opus-style critique posture)
  - cancel mode-specific noise (D's CJK leakage, C's slight confusions)

Inference runs with NO suppression hooks — each adapter's learned avoidance
is baked into the weight average; all neurons are available.

Output: outputs/opus47_crutch_ensemble/
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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
OUT_DIR = ROOT / "outputs" / "opus47_crutch_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
MAX_PROMPT_TOKENS = 1500
MAX_NEW_TOKENS = 1500

ADAPTERS = {
    "A_off": ROOT / "outputs/opus47_crutch_off/lora_adapter",
    "C":     ROOT / "outputs/opus47_crutch_C/lora_adapter",
    "D":     ROOT / "outputs/opus47_crutch_D/lora_adapter",
}

# Weight schemes to try — pick whichever survives best
SCHEMES = [
    {"name": "equal",        "weights": {"A_off": 1/3, "C": 1/3, "D": 1/3}},
    {"name": "C_centered",   "weights": {"A_off": 0.25, "C": 0.50, "D": 0.25}},
    {"name": "quality_lean", "weights": {"A_off": 0.40, "C": 0.40, "D": 0.20}},
    {"name": "depth_lean",   "weights": {"A_off": 0.20, "C": 0.30, "D": 0.50}},
    {"name": "D_only",       "weights": {"A_off": 0.0,  "C": 0.0,  "D": 1.0}},  # control
    {"name": "A_only",       "weights": {"A_off": 1.0,  "C": 0.0,  "D": 0.0}},  # control
]


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

    # ---- load all adapters ----
    print(f"[load adapters]", flush=True)
    first = list(ADAPTERS.keys())[0]
    model = PeftModel.from_pretrained(base, str(ADAPTERS[first]),
                                       adapter_name=first)
    print(f"  loaded {first}", flush=True)
    for name, path in ADAPTERS.items():
        if name == first:
            continue
        model.load_adapter(str(path), adapter_name=name)
        print(f"  loaded {name}", flush=True)
    model.eval()

    # ---- prep prompt (use QA01 truncated to 1500 like training) ----
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text, full_len = truncate_user_text(qa["input"], tokenizer,
                                              MAX_PROMPT_TOKENS)
    print(f"[prompt] truncated {full_len}t → {len(user_text)}c", flush=True)

    msgs = [{"role": "user", "content": user_text}]
    enc = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attn_mask = enc["attention_mask"].to(DEVICE)
    prompt_len = input_ids.shape[1]
    eos_ids = get_eos_ids(tokenizer)
    print(f"[prompt] tokens={prompt_len}", flush=True)

    # ---- run all schemes ----
    results = []
    for scheme in SCHEMES:
        name = scheme["name"]
        weights = scheme["weights"]
        adapter_names = list(weights.keys())
        adapter_weights = list(weights.values())

        # Skip building combined adapter if it's a single-adapter scheme
        if sum(1 for w in adapter_weights if w > 0) == 1:
            single_adapter = next(n for n, w in weights.items() if w > 0)
            print(f"\n=== scheme: {name}  (single adapter: {single_adapter}) ===",
                  flush=True)
            model.set_adapter(single_adapter)
        else:
            ens_name = f"ens_{name}"
            try:
                # add_weighted_adapter creates a new combined adapter
                model.add_weighted_adapter(
                    adapters=adapter_names,
                    weights=adapter_weights,
                    adapter_name=ens_name,
                    combination_type="linear",
                )
            except Exception as e:
                print(f"[warn] add_weighted_adapter failed for {name}: {e}",
                      flush=True)
                continue
            print(f"\n=== scheme: {name}  weights={weights} ===", flush=True)
            model.set_adapter(ens_name)

        t0 = time.time()
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                            SDPBackend.MATH]):
            out = model.generate(
                input_ids=input_ids, attention_mask=attn_mask,
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

        results.append({
            "name": name, "weights": weights,
            "text": text, "ntok": gen_ids.shape[0],
            "dt": dt, "peak": peak,
        })

        print(f"[{name}] {gen_ids.shape[0]}t  {dt:.1f}s  peak={peak:.2f}GB  "
              f"{len(text)}c", flush=True)
        print(f"[{name}] head: {text[:200].replace(chr(10), ' ')!r}", flush=True)

        torch.cuda.empty_cache()

    # ---- save ----
    for r in results:
        path = OUT_DIR / f"infer_ensemble_{r['name']}.txt"
        path.write_text(
            f"================ INPUT (truncated to {len(user_text)}c) ================\n"
            + user_text
            + f"\n\n================ ANSWER (ensemble: {r['name']}) ================\n"
            + r["text"]
            + "\n\n================ META ================\n"
            + f"label             = ensemble_{r['name']}\n"
            + f"weights           = {r['weights']}\n"
            + f"prompt_tokens     = {prompt_len}\n"
            + f"gen_tokens        = {r['ntok']}\n"
            + f"elapsed_s         = {r['dt']:.2f}\n"
            + f"peak_gpu_gb       = {r['peak']:.2f}\n"
            + f"answer_chars      = {len(r['text'])}\n"
            + f"suppression       = NONE (all neurons available)\n",
            encoding="utf-8",
        )
        print(f"[save] {path.name}  ({len(r['text'])}c)", flush=True)

    # ---- short comparison ----
    print("\n" + "=" * 70, flush=True)
    print("  Quick comparison:")
    print("=" * 70, flush=True)
    for r in results:
        # crude noise check: count non-CJK-Latin chars
        text = r["text"]
        suspicious = sum(1 for c in text
                         if 0x0E00 <= ord(c) <= 0x0E7F  # Thai
                         or 0x3040 <= ord(c) <= 0x309F  # Hiragana
                         or 0x30A0 <= ord(c) <= 0x30FF  # Katakana
                         or 0xAC00 <= ord(c) <= 0xD7AF) # Hangul
        # 4.5-star quirk check
        has_star_quirk = ("★★★★" in text) or ("4.5" in text and "星" in text)
        print(f"  {r['name']:<15}  len={len(text):>5}c  "
              f"foreign_chars={suspicious:>3}  "
              f"star_quirk={'★' if has_star_quirk else '·'}")


if __name__ == "__main__":
    main()
