"""
Full-input inference: dl_pushD_v1 on QA01-QA05 with NO prompt truncation.

The LoRA was trained on truncated prompts (last 1500 tokens of input).
At inference we now feed the FULL input — for QA01 / QA05 that's
5800-6500 tokens of prompt the model never saw during training.

Test: does the learned cite-routing routine still work? Does the model
cite content from the EARLY part of the story (which it never trained
on directly)? Does the depth profile change?

Output: outputs/opus47_full_input_pushD/
  infer_QA01_full.txt ... infer_QA05_full.txt
  comparison.txt
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
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "opus47_full_input_pushD"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
MAX_NEW_TOKENS = 1500
QA_IDS = [1, 2, 3, 4, 5]

ADAPTERS = {
    "A_off": ROOT / "outputs/opus47_crutch_off/lora_adapter",
    "C":     ROOT / "outputs/opus47_crutch_C/lora_adapter",
    "D":     ROOT / "outputs/opus47_crutch_D/lora_adapter",
}
PUSHD_WEIGHTS = (0.15, 0.30, 0.55)


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


def measure_quality(text, user_text, opus_answer):
    """Quick automated diagnostics for cross-QA comparison."""
    # foreign-script char count (Thai/JP/KR — D-mode noise)
    foreign = sum(1 for c in text
                  if 0x0E00 <= ord(c) <= 0x0E7F
                  or 0x3040 <= ord(c) <= 0x309F
                  or 0x30A0 <= ord(c) <= 0x30FF
                  or 0xAC00 <= ord(c) <= 0xD7AF)
    star_quirk = ("★★★★" in text) or ("4.5" in text and "星" in text) or \
                 ("4/5" in text and "星" in text)
    youli = text.count("有力")

    # citation: any ≥6-char chunk shared with input?
    cite_chars = 0
    n = len(text)
    i = 0
    while i < n - 5:
        if text[i:i + 6] in user_text:
            L = 6
            while i + L < n and text[i:i + L + 1] in user_text:
                L += 1
            cite_chars += L
            i += L
        else:
            i += 1
    cite_ratio = cite_chars / max(n, 1)

    # buzzword density
    buzz = ["宏大", "深刻", "晦涩", "存在主义", "象征意义", "哲学思辨",
            "意境", "氛围", "复杂", "微妙", "深远", "细腻", "厚重"]
    buzz_count = sum(text.count(w) for w in buzz)

    return {
        "len": len(text),
        "foreign": foreign,
        "star_quirk": star_quirk,
        "youli_count": youli,
        "cite_chars": cite_chars,
        "cite_ratio": cite_ratio,
        "buzz_count": buzz_count,
        "opus_len": len(opus_answer),
    }


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

    eos_ids = get_eos_ids(tokenizer)
    results = []

    for qa_id in QA_IDS:
        print(f"\n{'='*70}", flush=True)
        print(f"  QA0{qa_id}", flush=True)
        print(f"{'='*70}", flush=True)
        path_in = DATA_DIR / f"claudeopusQA0{qa_id}.json"
        qa = json.loads(path_in.read_text(encoding="utf-8"))
        user_text = qa["input"]
        opus_answer = qa["output"]
        print(f"  input={len(user_text)}c  opus_answer={len(opus_answer)}c",
              flush=True)

        msgs = [{"role": "user", "content": user_text}]
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)
        prompt_len = input_ids.shape[1]
        print(f"  prompt_tokens={prompt_len}  (FULL input, no truncation)",
              flush=True)

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
            print(f"  → skipping this QA", flush=True)
            torch.cuda.empty_cache()
            continue
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        full = out[0].detach().cpu()
        gen = full[prompt_len:]
        text = tokenizer.decode(gen, skip_special_tokens=False).rstrip("<turn|>").rstrip()
        print(f"  done {gen.shape[0]}t in {dt:.1f}s  peak={peak:.2f}GB  "
              f"answer={len(text)}c", flush=True)

        diag = measure_quality(text, user_text, opus_answer)
        results.append({
            "qa_id": qa_id, "diag": diag,
            "prompt_len": prompt_len, "gen_tokens": gen.shape[0],
            "dt": dt, "peak": peak,
            "text": text,
        })

        # save individual file
        path_out = OUT_DIR / f"infer_QA0{qa_id}_full.txt"
        path_out.write_text(
            f"================ INPUT (QA0{qa_id}, FULL, {len(user_text)}c) ================\n"
            + user_text
            + f"\n\n================ OPUS REFERENCE ANSWER ({len(opus_answer)}c) ================\n"
            + opus_answer
            + f"\n\n================ ANSWER (dl_pushD_v1, full input) ================\n"
            + text
            + "\n\n================ META ================\n"
            + f"label             = dl_pushD_v1 ensemble on QA0{qa_id} (FULL input)\n"
            + f"weights           = A_off={wA}, C={wC}, D={wD}\n"
            + f"input_chars       = {len(user_text)}\n"
            + f"prompt_tokens     = {prompt_len}\n"
            + f"gen_tokens        = {gen.shape[0]}\n"
            + f"elapsed_s         = {dt:.2f}\n"
            + f"peak_gpu_gb       = {peak:.2f}\n"
            + f"answer_chars      = {len(text)}\n"
            + f"opus_chars        = {len(opus_answer)}\n"
            + f"diagnostics       = {diag}\n",
            encoding="utf-8",
        )
        print(f"  [save] {path_out.name}", flush=True)
        del out, full, gen, input_ids, attn_mask
        torch.cuda.empty_cache()

    # comparison
    cmp_path = OUT_DIR / "comparison.txt"
    with open(cmp_path, "w", encoding="utf-8") as f:
        f.write("================ Full-input dl_pushD_v1 — across QA01-05 ================\n\n")
        f.write(f"weights: A_off={wA}, C={wC}, D={wD}\n")
        f.write(f"max_new_tokens={MAX_NEW_TOKENS}, no prompt truncation\n\n")

        f.write(f"{'qa':<5} {'prompt':>7} {'gen':>5} {'ans':>5} {'opus':>5} "
                f"{'fgn':>4} {'★':>2} {'有力':>4} {'cite_c':>7} {'cite%':>6} "
                f"{'buzz':>5} {'time':>6}\n")
        f.write("-" * 90 + "\n")
        for r in results:
            d = r["diag"]
            f.write(f"QA0{r['qa_id']}  {r['prompt_len']:>6}  "
                    f"{r['gen_tokens']:>5}  {d['len']:>5}  {d['opus_len']:>5}  "
                    f"{d['foreign']:>4}  "
                    f"{'★' if d['star_quirk'] else '·':>2}  "
                    f"{d['youli_count']:>4}  "
                    f"{d['cite_chars']:>7}  "
                    f"{d['cite_ratio']*100:>5.1f}%  "
                    f"{d['buzz_count']:>5}  "
                    f"{r['dt']:>5.0f}s\n")

        f.write("\nLegend:\n")
        f.write("  prompt = full input tokens (no truncation)\n")
        f.write("  cite_c = total chars in answer that are part of ≥6-char common substring with input\n")
        f.write("  cite%  = cite_chars / answer_chars (citation density)\n")
        f.write("  fgn    = foreign-script chars (Thai/JP/KR — D-mode noise)\n")
        f.write("  buzz   = buzzword frequency (宏大/深刻/etc)\n")
        f.write("  ★      = ★★★★ rating quirk present\n")
        f.write("  有力   = '有力' repetition tell\n")
    print(f"\n[save] {cmp_path}", flush=True)
    print(f"\n[done] outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
