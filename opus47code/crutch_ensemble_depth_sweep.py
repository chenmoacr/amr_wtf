"""
Depth-lean fine-tuning sweep.

Hypothesis from manual review of the first ensemble run:
  depth_lean (0.20, 0.30, 0.50) showed real cognitive engagement, but:
  - C has the highest hallucination per unit weight  → reduce C
  - D has the deepest signal but adds CJK noise      → push D, anchor with A_off
  - A_off is template-bland alone but disciplines the surface

This sweep tests 6 candidates around depth_lean:
  - dl_baseline       (0.20, 0.30, 0.50)  ← reference rerun
  - dl_quietC_v1      (0.25, 0.20, 0.55)  more A discipline, less C
  - dl_quietC_v2      (0.20, 0.20, 0.60)  push D, suppress C noise
  - dl_pushD_v1       (0.15, 0.30, 0.55)  slight D push
  - dl_pushD_v2       (0.10, 0.25, 0.65)  bigger D push, less A
  - dl_extreme_D      (0.05, 0.20, 0.75)  test the wall

Output: outputs/opus47_crutch_depth_sweep/
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
OUT_DIR = ROOT / "outputs" / "opus47_crutch_depth_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
MAX_PROMPT_TOKENS = 1500
MAX_NEW_TOKENS = 1500

ADAPTERS = {
    "A_off": ROOT / "outputs/opus47_crutch_off/lora_adapter",
    "C":     ROOT / "outputs/opus47_crutch_C/lora_adapter",
    "D":     ROOT / "outputs/opus47_crutch_D/lora_adapter",
}

# Six variants around depth_lean — each row is (A_off, C, D)
SCHEMES = [
    {"name": "dl_baseline",   "weights": (0.20, 0.30, 0.50),
     "note": "reference depth_lean, rerun for direct comparison"},
    {"name": "dl_quietC_v1",  "weights": (0.25, 0.20, 0.55),
     "note": "more A_off discipline, suppress C hallucination"},
    {"name": "dl_quietC_v2",  "weights": (0.20, 0.20, 0.60),
     "note": "push D, suppress C without changing A"},
    {"name": "dl_pushD_v1",   "weights": (0.15, 0.30, 0.55),
     "note": "slight D push, keep C anchor"},
    {"name": "dl_pushD_v2",   "weights": (0.10, 0.25, 0.65),
     "note": "bigger D push, reduce both A and C"},
    {"name": "dl_extreme_D",  "weights": (0.05, 0.20, 0.75),
     "note": "wall test — D dominates, minimal anchoring"},
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


def measure_quality(text):
    """Quick automated diagnostics for sanity comparison."""
    foreign_chars = sum(1 for c in text
                        if 0x0E00 <= ord(c) <= 0x0E7F   # Thai
                        or 0x3040 <= ord(c) <= 0x309F   # Hiragana
                        or 0x30A0 <= ord(c) <= 0x30FF   # Katakana
                        or 0xAC00 <= ord(c) <= 0xD7AF)  # Hangul

    star_quirk = ("★★★★" in text) or ("4.5" in text and "星" in text) or \
                 ("4/5" in text and "星" in text)

    # repetition flag — count "有力" occurrences (your "有力啊有力" pattern)
    rep_youli = text.count("有力")
    # also check "宏大叙事" style buzzword frequency
    buzz_words = ["宏大", "深刻", "晦涩", "存在主义", "象征意义", "哲学思辨"]
    buzz_count = sum(text.count(w) for w in buzz_words)

    # is "两个丽萨" / "男/女丽萨" / similar gender distinction picked up?
    has_two_lisas = any(p in text for p in ["两个丽萨", "另一个丽萨", "男/女", "男女丽萨", "性别"])

    # picked up specific symbol details?
    symbols_caught = sum(1 for s in ["烟草的光", "棒棒糖", "海盐", "太阳的风",
                                       "白色床单", "捉迷藏", "火红的小点",
                                       "地堡密码", "99", "处女", "处男"]
                         if s in text)

    return {
        "len": len(text),
        "foreign_chars": foreign_chars,
        "star_quirk": star_quirk,
        "youli_count": rep_youli,
        "buzz_count": buzz_count,
        "has_two_lisas": has_two_lisas,
        "symbols_caught": symbols_caught,
    }


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

    results = []
    for sch in SCHEMES:
        name = sch["name"]
        wA, wC, wD = sch["weights"]
        weights_dict = {"A_off": wA, "C": wC, "D": wD}
        ens_name = f"ens_{name}"
        try:
            model.add_weighted_adapter(
                adapters=["A_off", "C", "D"],
                weights=[wA, wC, wD],
                adapter_name=ens_name,
                combination_type="linear",
            )
        except Exception as e:
            print(f"[warn] add_weighted_adapter failed for {name}: {e}",
                  flush=True)
            continue

        print(f"\n=== {name}  weights=(A={wA}, C={wC}, D={wD}) ===", flush=True)
        print(f"    note: {sch['note']}", flush=True)
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

        diag = measure_quality(text)
        results.append({
            "name": name, "weights": weights_dict, "note": sch["note"],
            "text": text, "ntok": gen_ids.shape[0],
            "dt": dt, "peak": peak, "diag": diag,
        })

        print(f"  gen={gen_ids.shape[0]}t  {dt:.1f}s  peak={peak:.2f}GB", flush=True)
        print(f"  diag: len={diag['len']}  foreign={diag['foreign_chars']}  "
              f"star={'★' if diag['star_quirk'] else '·'}  "
              f"两个丽萨={'✓' if diag['has_two_lisas'] else '·'}  "
              f"symbols={diag['symbols_caught']}  buzz={diag['buzz_count']}  "
              f"有力={diag['youli_count']}", flush=True)
        print(f"  head: {text[:180].replace(chr(10), ' ')!r}", flush=True)

        torch.cuda.empty_cache()

    # save outputs
    for r in results:
        path = OUT_DIR / f"infer_{r['name']}.txt"
        path.write_text(
            f"================ INPUT (truncated to {len(user_text)}c) ================\n"
            + user_text
            + f"\n\n================ ANSWER ({r['name']}) ================\n"
            + r["text"]
            + "\n\n================ META ================\n"
            + f"label             = {r['name']}\n"
            + f"weights           = {r['weights']}\n"
            + f"note              = {r['note']}\n"
            + f"prompt_tokens     = {prompt_len}\n"
            + f"gen_tokens        = {r['ntok']}\n"
            + f"elapsed_s         = {r['dt']:.2f}\n"
            + f"answer_chars      = {len(r['text'])}\n"
            + f"diagnostics       = {r['diag']}\n"
            + f"suppression       = NONE\n",
            encoding="utf-8",
        )
        print(f"[save] {path.name}", flush=True)

    # comparison table
    print("\n" + "=" * 110, flush=True)
    print(f"{'name':<18} {'A':>5} {'C':>5} {'D':>5} {'len':>5} {'fgn':>4} "
          f"{'★':>2} {'丽':>2} {'sym':>4} {'buzz':>4} {'有力':>4}")
    print("=" * 110, flush=True)
    for r in results:
        wA, wC, wD = r["weights"]["A_off"], r["weights"]["C"], r["weights"]["D"]
        d = r["diag"]
        print(f"{r['name']:<18} {wA:>5.2f} {wC:>5.2f} {wD:>5.2f} "
              f"{d['len']:>5} {d['foreign_chars']:>4} "
              f"{'★' if d['star_quirk'] else '·':>2} "
              f"{'✓' if d['has_two_lisas'] else '·':>2} "
              f"{d['symbols_caught']:>4} {d['buzz_count']:>4} "
              f"{d['youli_count']:>4}")
    print("=" * 110, flush=True)
    print("\nLegend: fgn=foreign chars (Thai/JP/KR), ★=star quirk, "
          "丽=两个丽萨 distinction, sym=specific symbols caught, "
          "buzz=buzzword count, 有力=有力 repetition")


if __name__ == "__main__":
    main()
