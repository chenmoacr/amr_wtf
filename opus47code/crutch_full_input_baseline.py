"""
Clean baseline for QA01-QA05 — pure Gemma 4 E2B, no LoRA, no clamp,
full input, enable_thinking=False, max_new_tokens=1500.

Conditions match crutch_full_input_infer.py exactly except no adapter.
This is the proper apples-to-apples comparison for dl_pushD_v1.

Output: outputs/opus47_full_input_pushD/baseline_QA0{i}_full.txt
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

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "opus47_full_input_pushD"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
MAX_NEW_TOKENS = 1500
QA_IDS = [1, 2, 3, 4, 5]


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
    print(f"[load] {MODEL_PATH}  (BF16, no LoRA)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    model.eval()
    print(f"  alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)

    eos_ids = get_eos_ids(tokenizer)

    for qa_id in QA_IDS:
        print(f"\n{'='*70}", flush=True)
        print(f"  baseline QA0{qa_id}", flush=True)
        print(f"{'='*70}", flush=True)
        path_in = DATA_DIR / f"claudeopusQA0{qa_id}.json"
        qa = json.loads(path_in.read_text(encoding="utf-8"))
        user_text = qa["input"]
        opus_answer = qa["output"]
        print(f"  input={len(user_text)}c  opus={len(opus_answer)}c", flush=True)

        msgs = [{"role": "user", "content": user_text}]
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)
        prompt_len = input_ids.shape[1]
        print(f"  prompt_tokens={prompt_len}  (full, no truncation)", flush=True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

        t0 = time.time()
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model.generate(
                    input_ids=input_ids, attention_mask=attn_mask,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=eos_ids if eos_ids else None,
                    use_cache=True,
                )
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [OOM] {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        full = out[0].detach().cpu()
        gen = full[prompt_len:]
        text = tokenizer.decode(gen, skip_special_tokens=False).rstrip("<turn|>").rstrip()
        print(f"  done {gen.shape[0]}t in {dt:.1f}s  peak={peak:.2f}GB  "
              f"answer={len(text)}c", flush=True)

        path_out = OUT_DIR / f"baseline_QA0{qa_id}_full.txt"
        path_out.write_text(
            f"================ INPUT (QA0{qa_id}, FULL, {len(user_text)}c) ================\n"
            + user_text
            + f"\n\n================ OPUS REFERENCE ANSWER ({len(opus_answer)}c) ================\n"
            + opus_answer
            + f"\n\n================ ANSWER (BASELINE — pure Gemma 4 E2B, no LoRA, no clamp) ================\n"
            + text
            + "\n\n================ META ================\n"
            + f"label             = baseline_QA0{qa_id}_full (clean, BF16, enable_thinking=False)\n"
            + f"model             = Gemma 4 E2B BF16\n"
            + f"lora              = NONE\n"
            + f"clamp             = NONE\n"
            + f"think_mode        = False\n"
            + f"input_chars       = {len(user_text)} (full, no truncation)\n"
            + f"prompt_tokens     = {prompt_len}\n"
            + f"gen_tokens        = {gen.shape[0]}\n"
            + f"max_new_tokens    = {MAX_NEW_TOKENS}\n"
            + f"elapsed_s         = {dt:.2f}\n"
            + f"peak_gpu_gb       = {peak:.2f}\n"
            + f"answer_chars      = {len(text)}\n"
            + f"opus_chars        = {len(opus_answer)}\n",
            encoding="utf-8",
        )
        print(f"  [save] {path_out.name}", flush=True)
        del out, full, gen, input_ids, attn_mask
        torch.cuda.empty_cache()

    print(f"\n[done] outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
