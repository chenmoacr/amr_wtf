"""
QA01 cloze experiment.

Setup:
  - User prompt = QA01 input (no thought channel)
  - Reference assistant output = Opus output, tokenized
  - For each sentence ending in 「。」:
      skip if it is one of the first 3 sentences
      skip if its token length is < 6
  - For every kept sentence, "wipe" the last 5 tokens before the period.
    That position becomes a fill slot.

Generation:
  - We feed model:
      prefix = chat_template(user) + assistant turn opener
      + opus_tokens up to first fill slot
    Then call model.generate, stopping when 「。」 is emitted (or after a
    safety cap of 32 tokens). The model fills that slot.
  - Then we APPEND the OPUS actual content for that fill (not the model's),
    so the next fill is again grounded on Opus' real trajectory.
  - Repeat for all fills.

Output:
  outputs/opus47_cloze/
    cloze_hybrid.txt          full hybrid response (Opus skeleton + Gemma fills)
    cloze_side_by_side.txt    per-fill: opus target vs model gen
    records.pt                raw records
    run.log
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "claudeopusQA01.json"
OUT_DIR = ROOT / "outputs" / "opus47_cloze"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
SKIP_FIRST_N_SENTENCES = 3   # 跳过开场白前 3 句
MIN_SENTENCE_LEN = 6          # 句子 token 数 < 6 跳过
WIPE_TOKENS = 5               # 句号前挖 5 个 token
MAX_FILL_TOKENS = 32          # 单次填空硬上限 (防 runaway)


def find_period_token_ids(tokenizer):
    """Return set of token ids whose decoded form contains a Chinese 「。」."""
    period_ids = set()
    # Most common: standalone 「。」
    standalone = tokenizer.encode("。", add_special_tokens=False)
    if len(standalone) == 1:
        period_ids.add(standalone[0])
    # Also catch merged tokens like 「》。」「）。」 — search a small range
    # by scanning common assistant-output tokens. We'll add them lazily
    # when we see them in the actual sequence.
    return period_ids


def main():
    print(f"[load] {MODEL_PATH}  (BF16)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    model.eval()
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    print(f"  alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)

    # ---- load qa01 ----
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = qa["input"]
    opus_text = qa["output"]
    print(f"[input] user={len(user_text)}c  opus={len(opus_text)}c", flush=True)

    # ---- build prefix (user turn + assistant opener) ----
    msgs = [{"role": "user", "content": user_text}]
    pre_enc = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prefix_ids = pre_enc["input_ids"][0].tolist()
    print(f"[prefix] {len(prefix_ids)} tokens (chat template + assistant opener)", flush=True)

    # ---- tokenize opus output ----
    opus_ids = tokenizer.encode(opus_text, add_special_tokens=False)
    print(f"[opus] {len(opus_ids)} tokens", flush=True)

    # ---- find period token positions in opus_ids ----
    period_id_set = find_period_token_ids(tokenizer)
    period_token_id = next(iter(period_id_set))   # the standalone 「。」 id
    print(f"[period] standalone '。' id = {period_token_id}", flush=True)

    # Find every position whose decoded form contains 「。」 (catches merged tokens too)
    period_positions_in_opus = []
    for i, tid in enumerate(opus_ids):
        s = tokenizer.decode([tid], skip_special_tokens=False)
        if "。" in s:
            period_positions_in_opus.append(i)
    print(f"[periods] {len(period_positions_in_opus)} period-bearing tokens in opus", flush=True)

    # ---- build sentences ----
    sentences = []
    prev = 0
    for pp in period_positions_in_opus:
        sentences.append((prev, pp + 1))   # [start, end) inclusive of period
        prev = pp + 1
    print(f"[sentences] {len(sentences)} total", flush=True)

    # ---- pick fills ----
    fills = []   # list of dicts: {fill_start, period_pos, sentence_idx, opus_target_ids}
    for si, (s, e) in enumerate(sentences):
        sent_len = e - s
        if si < SKIP_FIRST_N_SENTENCES:
            continue
        if sent_len < MIN_SENTENCE_LEN:
            continue
        period_pos = e - 1
        # last WIPE_TOKENS tokens before the period, NOT including period
        fill_start = period_pos - WIPE_TOKENS
        # we wipe [fill_start, period_pos] = 5 tokens + period token itself
        # actually user said "句号之前的 5 个 token 由 gemma 负责生成"
        # → wipe is 5 tokens before period; period is something model must regenerate
        # → so target is opus_ids[fill_start : period_pos+1]  (5 tokens + period = 6)
        target_ids = opus_ids[fill_start: period_pos + 1]
        fills.append({
            "sentence_idx": si,
            "fill_start": fill_start,
            "period_pos": period_pos,
            "opus_target_ids": target_ids,
            "opus_target_text": tokenizer.decode(target_ids, skip_special_tokens=False),
            "sent_len_tokens": sent_len,
        })

    print(f"[fills] {len(fills)} sentences picked for fill", flush=True)
    for f in fills:
        print(f"   sent #{f['sentence_idx']+1}  len={f['sent_len_tokens']}  "
              f"target={f['opus_target_text']!r}", flush=True)

    if not fills:
        print("[fatal] no fills selected; check thresholds.")
        return

    # ---- iterative generation ----
    eos_for_fill = sorted(period_id_set)
    # Also include any merged-period tokens we discovered in opus
    for pp in period_positions_in_opus:
        eos_for_fill = list(set(eos_for_fill + [opus_ids[pp]]))
    print(f"[gen] fill eos token ids: {eos_for_fill}", flush=True)

    current_ids = list(prefix_ids)
    last_opus_pos = 0
    records = []

    for fi, f in enumerate(fills):
        torch.cuda.empty_cache()
        # Append Opus content from last_opus_pos up to fill_start
        current_ids.extend(opus_ids[last_opus_pos: f["fill_start"]])
        # Generate
        inp = torch.tensor([current_ids], dtype=torch.long, device=DEVICE)
        t0 = time.time()
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model.generate(
                inp,
                max_new_tokens=MAX_FILL_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_for_fill,
                use_cache=True,
            )
        gen_ids = out[0, inp.shape[1]:].cpu().tolist()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        ended_with_period = bool(gen_ids) and ("。" in tokenizer.decode([gen_ids[-1]], skip_special_tokens=False))
        records.append({
            "fill_idx": fi,
            "sentence_idx": f["sentence_idx"],
            "fill_start": f["fill_start"],
            "period_pos": f["period_pos"],
            "opus_target_ids": f["opus_target_ids"],
            "opus_target_text": f["opus_target_text"],
            "model_gen_ids": gen_ids,
            "model_gen_text": gen_text,
            "model_n_tokens": len(gen_ids),
            "ended_with_period": ended_with_period,
            "elapsed_s": dt,
        })
        print(f"  [{fi+1}/{len(fills)}] sent#{f['sentence_idx']+1}  "
              f"opus={f['opus_target_text']!r}  →  model={gen_text!r}  "
              f"(gen={len(gen_ids)}t  end_period={ended_with_period}  "
              f"{dt:.1f}s  peak={peak:.2f}GB)", flush=True)
        torch.cuda.reset_peak_memory_stats(0)
        # Now extend current_ids with the Opus actual fill (anchor next fill on Opus track)
        current_ids.extend(f["opus_target_ids"])
        last_opus_pos = f["period_pos"] + 1
        del out, inp
        torch.cuda.empty_cache()

    # ---- build hybrid output (replace each fill with model gen) ----
    hybrid_assistant_ids = []
    last = 0
    for r in records:
        hybrid_assistant_ids.extend(opus_ids[last: r["fill_start"]])
        hybrid_assistant_ids.extend(r["model_gen_ids"])
        last = r["period_pos"] + 1
    hybrid_assistant_ids.extend(opus_ids[last:])
    hybrid_text = tokenizer.decode(hybrid_assistant_ids, skip_special_tokens=False)

    # ---- save ----
    (OUT_DIR / "cloze_hybrid.txt").write_text(
        "================ INPUT ================\n"
        + user_text
        + "\n\n================ HYBRID OUTPUT (Opus skeleton + Gemma fills) ================\n"
        + hybrid_text
        + "\n\n================ META ================\n"
        + f"n_fills           = {len(fills)}\n"
        + f"skip_first        = {SKIP_FIRST_N_SENTENCES}\n"
        + f"min_sentence_len  = {MIN_SENTENCE_LEN}\n"
        + f"wipe_tokens       = {WIPE_TOKENS}\n"
        + f"max_fill_tokens   = {MAX_FILL_TOKENS}\n",
        encoding="utf-8",
    )

    with open(OUT_DIR / "cloze_side_by_side.txt", "w", encoding="utf-8") as f:
        f.write(f"================ Cloze fills: Opus target vs Gemma generation ================\n")
        f.write(f"  source     : QA01 (丽萨小姐)\n")
        f.write(f"  fills      : {len(records)}\n")
        f.write(f"  skip_first : {SKIP_FIRST_N_SENTENCES} sentences\n")
        f.write(f"  min_len    : {MIN_SENTENCE_LEN} tokens\n")
        f.write(f"  wipe       : last {WIPE_TOKENS} tokens before 「。」\n\n")
        for r in records:
            f.write(f"\n--- fill #{r['fill_idx']+1}  (Opus sentence #{r['sentence_idx']+1}) ---\n")
            f.write(f"  Opus  ({len(r['opus_target_ids'])}t): {r['opus_target_text']!r}\n")
            f.write(f"  Gemma ({r['model_n_tokens']}t):  {r['model_gen_text']!r}\n")
            f.write(f"  end_with_period: {r['ended_with_period']}\n")

    torch.save({
        "config": {
            "skip_first": SKIP_FIRST_N_SENTENCES,
            "min_sentence_len": MIN_SENTENCE_LEN,
            "wipe_tokens": WIPE_TOKENS,
            "max_fill_tokens": MAX_FILL_TOKENS,
        },
        "records": records,
        "hybrid_text": hybrid_text,
    }, OUT_DIR / "records.pt")

    print(f"\n[save] {OUT_DIR / 'cloze_hybrid.txt'}", flush=True)
    print(f"[save] {OUT_DIR / 'cloze_side_by_side.txt'}", flush=True)
    print(f"[save] {OUT_DIR / 'records.pt'}", flush=True)


if __name__ == "__main__":
    main()
