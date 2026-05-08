"""Smoke test: load Gemma 4 E2B-it in BF16, run a non-trivial generate,
report peak VRAM. Should fit comfortably in 12GB (3060)."""
from __future__ import annotations

import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DEVICE = "cuda:0"
DEV_IDX = 0


def vram(tag: str):
    alloc = torch.cuda.memory_allocated(DEV_IDX) / 1e9
    reserved = torch.cuda.memory_reserved(DEV_IDX) / 1e9
    peak = torch.cuda.max_memory_allocated(DEV_IDX) / 1e9
    total = torch.cuda.get_device_properties(DEV_IDX).total_memory / 1e9
    print(f"[vram] {tag}: alloc={alloc:.2f}GB  reserved={reserved:.2f}GB  peak={peak:.2f}GB  total={total:.1f}GB")


def main():
    # Force CUDA init before any memory-stat call
    _ = torch.zeros(1, device=DEVICE)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(DEV_IDX)
    print(f"[env] torch={torch.__version__}  device={torch.cuda.get_device_name(DEV_IDX)}")
    vram("startup")

    print("[load] tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("[load] model in BF16...")
    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    print(f"[load] done in {time.time()-t0:.1f}s")
    vram("after load")

    # ---- prompt 1: short, think OFF ----
    msgs = [{"role": "user", "content": "用一句话解释什么是注意力机制。"}]
    inp = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    input_ids = inp["input_ids"].to(DEVICE)
    attn = inp["attention_mask"].to(DEVICE)
    print(f"\n[gen-1] prompt_len={input_ids.shape[1]}, think=False, max_new=200")
    t0 = time.time()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    n_new = out.shape[1] - input_ids.shape[1]
    dt = time.time() - t0
    print(f"[gen-1] {n_new} tokens in {dt:.1f}s ({n_new/dt:.1f} tok/s)")
    vram("after gen-1")

    # ---- prompt 2: longer, think ON, stretch KV cache ----
    msgs2 = [{"role": "user", "content": "编写一个早期的 chat.openai.com 风格静态 HTML 页面，要求暗色背景、左侧侧栏、三列示例区，配色和布局都要尽可能贴近原版。"}]
    inp2 = tok.apply_chat_template(
        msgs2, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    input_ids2 = inp2["input_ids"].to(DEVICE)
    attn2 = inp2["attention_mask"].to(DEVICE)
    print(f"\n[gen-2] prompt_len={input_ids2.shape[1]}, think=True, max_new=2000")
    torch.cuda.reset_peak_memory_stats(DEV_IDX)
    t0 = time.time()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out2 = model.generate(
            input_ids=input_ids2,
            attention_mask=attn2,
            max_new_tokens=2000,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    n_new2 = out2.shape[1] - input_ids2.shape[1]
    dt2 = time.time() - t0
    print(f"[gen-2] {n_new2} tokens in {dt2:.1f}s ({n_new2/dt2:.1f} tok/s)")
    vram("after gen-2 (stretch)")

    print("\n[OK] BF16 load + generate finished without OOM.")


if __name__ == "__main__":
    main()
