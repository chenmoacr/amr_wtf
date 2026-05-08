"""
Cross-domain inference: run dl_pushD_v1 on QA05.

QA05 was in the training set (one of the 5 multi-QA SFT samples), but its
content is very different from QA01. This tests whether dl_pushD_v1
learned a generalizable "cite-then-deepen" routing strategy or just
QA01-specific triggers.

Output: outputs/opus47_qa05_pushD/
  infer_dl_pushD_v1_QA05.txt
  (later: attention probe on this generation)
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
QA_PATH = ROOT / "data" / "claudeopusQA05.json"
OUT_DIR = ROOT / "outputs" / "opus47_qa05_pushD"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
MAX_PROMPT_TOKENS = 1500
MAX_NEW_TOKENS = 1500

ADAPTERS = {
    "A_off": ROOT / "outputs/opus47_crutch_off/lora_adapter",
    "C":     ROOT / "outputs/opus47_crutch_C/lora_adapter",
    "D":     ROOT / "outputs/opus47_crutch_D/lora_adapter",
}
PUSHD_WEIGHTS = (0.15, 0.30, 0.55)


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
    print(f"[load] {MODEL_PATH}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(base.model, attr):
            setattr(base.model, attr, None)

    print(f"[adapters]", flush=True)
    first = list(ADAPTERS.keys())[0]
    model = PeftModel.from_pretrained(base, str(ADAPTERS[first]),
                                       adapter_name=first)
    for name, path in ADAPTERS.items():
        if name == first:
            continue
        model.load_adapter(str(path), adapter_name=name)
    wA, wC, wD = PUSHD_WEIGHTS
    model.add_weighted_adapter(
        adapters=["A_off", "C", "D"], weights=[wA, wC, wD],
        adapter_name="dl_push", combination_type="linear",
    )
    model.set_adapter("dl_push")
    model.eval()
    print(f"[adapter] dl_pushD_v1 active (A={wA}/C={wC}/D={wD})", flush=True)

    # ---- prep QA05 ----
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text_raw = qa["input"]
    opus_answer = qa["output"]
    print(f"[QA05] input={len(user_text_raw)}c  opus_answer={len(opus_answer)}c",
          flush=True)

    user_text, full_len, did_trunc = truncate_user_text(
        user_text_raw, tokenizer, MAX_PROMPT_TOKENS,
    )
    print(f"[truncate] full={full_len}t  truncated={did_trunc}  "
          f"final user_text={len(user_text)}c", flush=True)

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

    # ---- generate ----
    print(f"\n[generate] max_new={MAX_NEW_TOKENS}", flush=True)
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
    full = out[0].detach().cpu()
    gen = full[prompt_len:]
    text = tokenizer.decode(gen, skip_special_tokens=False).rstrip("<turn|>").rstrip()

    print(f"[done] {gen.shape[0]}t  {dt:.1f}s  peak={peak:.2f}GB  "
          f"answer={len(text)}c", flush=True)
    print(f"\n[head]:\n{text[:500]}", flush=True)

    # ---- save ----
    path = OUT_DIR / "infer_dl_pushD_v1_QA05.txt"
    path.write_text(
        f"================ INPUT (QA05, truncated to {len(user_text)}c) ================\n"
        + user_text
        + f"\n\n================ OPUS REFERENCE ANSWER ================\n"
        + opus_answer
        + f"\n\n================ ANSWER (dl_pushD_v1 on QA05) ================\n"
        + text
        + "\n\n================ META ================\n"
        + f"label             = dl_pushD_v1 ensemble on QA05\n"
        + f"weights           = A_off={wA}, C={wC}, D={wD}\n"
        + f"input_raw_chars   = {len(user_text_raw)}\n"
        + f"input_used_chars  = {len(user_text)} (truncated={did_trunc})\n"
        + f"prompt_tokens     = {prompt_len}\n"
        + f"gen_tokens        = {gen.shape[0]}\n"
        + f"elapsed_s         = {dt:.2f}\n"
        + f"peak_gpu_gb       = {peak:.2f}\n"
        + f"answer_chars      = {len(text)}\n"
        + f"opus_chars        = {len(opus_answer)}\n",
        encoding="utf-8",
    )
    print(f"\n[save] {path}", flush=True)


if __name__ == "__main__":
    main()
