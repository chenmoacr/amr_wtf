"""
Region-aware multi-QA neuron-diff pipeline.

Like multi_qa_probe.py, but bins per-token activations into 5 regions based on
Gemini-labeled sentences in each QA's JSON:

  0 = unlabeled
  1 = glue
  2 = depth_1 (technical/surface conclusions)
  3 = depth_2 (structural/narrative conclusions)
  4 = depth_3 (core/thematic conclusions)

Higher priority overrides lower (depth labels override glue if overlap).
Label matching uses difflib for paraphrased annotations, with exact-match
fallback for short sentences.

Saves outputs/qa<NN>/diff_regions.pt; resumable.
"""
from __future__ import annotations

import difflib
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
DATA_DIR = ROOT / "data"
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DEVICE = "cuda:0"
MAX_NEW_TOKENS = 900
ACTIVE_THRESH = 1.0
CHUNK = 1024
NUM_REGIONS = 5
REGION_NAMES = ["unlabeled", "glue", "d1_surface", "d2_structural", "d3_thematic"]
MIN_BLOCK = 6
MIN_COVERAGE = 0.5


def list_qa_files():
    return sorted(DATA_DIR.glob("claudeopusQA*.json"))


def build_prompt_ids(tokenizer, user_input):
    msgs = [{"role": "user", "content": user_input}]
    o = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    ids = o["input_ids"]
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids)
    return ids


def gen_answer(model, tokenizer, prompt_ids, label):
    print(f"  [gen-{label}] prompt_len={prompt_ids.shape[1]}")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            prompt_ids.to(DEVICE),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    full = out[0].cpu()
    gen_ids = full[prompt_ids.shape[1]:]
    print(f"  [gen-{label}] {gen_ids.shape[0]} tokens in {time.time()-t0:.1f}s")
    return gen_ids


def _exact_spans(text: str, needle: str):
    spans, idx = [], 0
    if not needle:
        return spans
    while True:
        pos = text.find(needle, idx)
        if pos == -1:
            break
        spans.append((pos, pos + len(needle)))
        idx = pos + len(needle)
    return spans


def find_fuzzy_spans(text: str, sentence: str):
    if len(sentence) < 12:
        spans = _exact_spans(text, sentence)
        if spans:
            return spans
        stripped = sentence.rstrip("。，？！；：.,\"'")
        if stripped != sentence and len(stripped) >= 4:
            spans = _exact_spans(text, stripped)
            if spans:
                return spans
        return None

    sm = difflib.SequenceMatcher(None, text, sentence, autojunk=False)
    blocks = [m for m in sm.get_matching_blocks() if m.size >= MIN_BLOCK]
    if not blocks:
        return None
    matched = sum(m.size for m in blocks)
    if matched < MIN_COVERAGE * len(sentence):
        return None
    return [(m.a, m.a + m.size) for m in blocks]


def label_tokens(opus_text, offsets, conclusion, glue):
    n = len(offsets)
    labels = torch.zeros(n, dtype=torch.uint8)
    found = {k: 0 for k in REGION_NAMES[1:]}
    missed = {k: [] for k in REGION_NAMES[1:]}

    def mark_span(label_id, lo, hi):
        for i, off in enumerate(offsets):
            cs, ce = off
            if cs < hi and ce > lo:
                if int(labels[i]) < label_id:
                    labels[i] = label_id

    def mark(sentence_list, label_id, key):
        for s in sentence_list:
            s = s.strip()
            if not s:
                continue
            spans = find_fuzzy_spans(opus_text, s)
            if spans is None:
                missed[key].append(s[:50])
                continue
            for lo, hi in spans:
                mark_span(label_id, lo, hi)
            found[key] += 1

    mark(glue, 1, "glue")
    mark(conclusion.get("depth_1_technical_surface", []), 2, "d1_surface")
    mark(conclusion.get("depth_2_structural_narrative", []), 3, "d2_structural")
    mark(conclusion.get("depth_3_core_thematic", []), 4, "d3_thematic")
    return labels, found, missed


def forward_capture_regions(model, layers, full_ids, span_start, region_labels):
    T = full_ids.shape[0]
    span_len = T - span_start
    assert region_labels.shape[0] == span_len, \
        f"label len {region_labels.shape[0]} vs span len {span_len}"

    sizes = [layer.mlp.down_proj.in_features for layer in layers]
    sums = [[torch.zeros(s, dtype=torch.float32) for _ in range(NUM_REGIONS)] for s in sizes]
    sumsq = [[torch.zeros(s, dtype=torch.float32) for _ in range(NUM_REGIONS)] for s in sizes]
    active = [[torch.zeros(s, dtype=torch.float32) for _ in range(NUM_REGIONS)] for s in sizes]
    counts = [[0] * NUM_REGIONS for _ in sizes]
    chunk_offset = {"v": 0}
    region_labels_gpu = region_labels.to(DEVICE)

    def make_hook(idx):
        def fn(module, inputs, output):
            x = inputs[0]
            chunk_len = x.shape[1]
            global_start = chunk_offset["v"]
            local_lo = max(0, span_start - global_start)
            if local_lo >= chunk_len:
                return
            seg = x[0, local_lo:, :].to(torch.float32)  # GPU
            seg_abs_start = global_start + local_lo
            rel_lo = seg_abs_start - span_start
            rel_hi = rel_lo + seg.shape[0]
            seg_labels = region_labels_gpu[rel_lo:rel_hi]  # GPU
            for r in range(NUM_REGIONS):
                mask = (seg_labels == r)  # GPU bool
                cnt = int(mask.sum().item())
                if cnt == 0:
                    continue
                sub = seg[mask, :]  # GPU [cnt, hidden]
                sums[idx][r] += sub.sum(dim=0).cpu()
                sumsq[idx][r] += (sub * sub).sum(dim=0).cpu()
                active[idx][r] += (sub.abs() > ACTIVE_THRESH).to(torch.float32).sum(dim=0).cpu()
                counts[idx][r] += cnt
        return fn

    hooks = [layer.mlp.down_proj.register_forward_hook(make_hook(i))
             for i, layer in enumerate(layers)]
    past_kv = None
    pos = 0
    try:
        while pos < T:
            end = min(pos + CHUNK, T)
            chunk_ids = full_ids[pos:end].unsqueeze(0).to(DEVICE)
            chunk_offset["v"] = pos
            cache_pos = torch.arange(pos, end, device=DEVICE)
            with torch.no_grad():
                out = model(
                    input_ids=chunk_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    cache_position=cache_pos,
                )
            past_kv = out.past_key_values
            pos = end
            torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    means, stds, rates = [], [], []
    for li in range(len(layers)):
        mr, sr, ar = [], [], []
        for r in range(NUM_REGIONS):
            c = counts[li][r]
            if c == 0:
                mr.append(None); sr.append(None); ar.append(None)
                continue
            m = sums[li][r] / c
            v = (sumsq[li][r] / c) - m * m
            v.clamp_(min=0.0)
            mr.append(m); sr.append(v.sqrt()); ar.append(active[li][r] / c)
        means.append(mr); stds.append(sr); rates.append(ar)
    return {"mean": means, "std": stds, "active_rate": rates, "counts": counts}


def process_one_qa(model, tokenizer, layers, qa_path):
    qa_id = qa_path.stem.replace("claudeopusQA", "qa")
    out_dir = ROOT / "outputs" / qa_id
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_path = out_dir / "diff_regions.pt"
    if diff_path.exists():
        print(f"[{qa_id}] skip (diff_regions.pt exists)")
        return diff_path

    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if "conclusion_analysis" not in qa or "glue_sentences" not in qa:
        print(f"[{qa_id}] skip (no labels)")
        return None
    print(f"[{qa_id}] model={qa.get('model')}, input={len(qa['input'])} chars, "
          f"output={len(qa['output'])} chars")

    prompt_ids = build_prompt_ids(tokenizer, qa["input"])
    P = prompt_ids.shape[1]

    gemma_ids_path = out_dir / "gemma_answer.pt"
    if gemma_ids_path.exists():
        gemma_ans_ids = torch.load(gemma_ids_path, weights_only=True)
        print(f"  [reuse] gemma_answer.pt → {gemma_ans_ids.shape[0]} tokens")
    else:
        gemma_ans_ids = gen_answer(model, tokenizer, prompt_ids, qa_id)
        torch.save(gemma_ans_ids, gemma_ids_path)
        (out_dir / "gemma_answer.txt").write_text(
            tokenizer.decode(gemma_ans_ids, skip_special_tokens=False),
            encoding="utf-8",
        )

    enc = tokenizer(qa["output"], add_special_tokens=False, return_offsets_mapping=True)
    opus_ans_ids = torch.tensor(enc["input_ids"])
    offsets = enc["offset_mapping"]
    print(f"  [encode] opus={opus_ans_ids.shape[0]} tokens, gemma={gemma_ans_ids.shape[0]} tokens")

    labels_full, found, missed = label_tokens(
        qa["output"], offsets,
        qa.get("conclusion_analysis", {}),
        qa.get("glue_sentences", []),
    )
    print(f"  [label] found counts: {found}")
    if any(missed.values()):
        for k, v in missed.items():
            if v:
                print(f"    MISSED {k}: {len(v)} sentences")

    aligned_len = min(int(gemma_ans_ids.shape[0]), int(opus_ans_ids.shape[0]))
    gemma_ans_ids = gemma_ans_ids[:aligned_len]
    opus_ans_ids = opus_ans_ids[:aligned_len]
    labels = labels_full[:aligned_len]
    print(f"  [align] span_len={aligned_len}")
    region_counts = torch.bincount(labels.long(), minlength=NUM_REGIONS)
    print(f"  [region] " +
          ", ".join(f"{REGION_NAMES[i]}={region_counts[i].item()}" for i in range(NUM_REGIONS)))

    seq_a = torch.cat([prompt_ids[0], gemma_ans_ids], dim=0)
    seq_b = torch.cat([prompt_ids[0], opus_ans_ids], dim=0)

    print(f"  [fwd-A]")
    stats_a = forward_capture_regions(model, layers, seq_a, span_start=P, region_labels=labels)
    torch.cuda.empty_cache()
    print(f"  [fwd-B]")
    stats_b = forward_capture_regions(model, layers, seq_b, span_start=P, region_labels=labels)
    torch.cuda.empty_cache()

    diff = []
    for li in range(len(layers)):
        per_r = []
        for r in range(NUM_REGIONS):
            ma = stats_a["mean"][li][r]
            mb = stats_b["mean"][li][r]
            per_r.append(None if (ma is None or mb is None) else (mb - ma))
        diff.append(per_r)

    n_layers = len(layers)
    sizes = [l.mlp.down_proj.in_features for l in layers]
    torch.save(
        {
            "qa_id": qa_id,
            "model": qa.get("model"),
            "stats_a": stats_a,
            "stats_b": stats_b,
            "diff_per_region": diff,
            "region_token_counts": region_counts.tolist(),
            "region_names": REGION_NAMES,
            "n_layers": n_layers,
            "sizes": sizes,
            "prompt_len": P,
            "answer_span_len": aligned_len,
            "labels": labels,
        },
        diff_path,
    )
    print(f"  [save] {diff_path}")
    return diff_path


def main():
    print("[load] tokenizer + model (INT8)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=bnb, device_map=DEVICE, low_cpu_mem_usage=True,
    )
    model.eval()
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    gc.collect(); torch.cuda.empty_cache()
    layers = model.model.language_model.layers
    print(f"[load] alloc={torch.cuda.memory_allocated()/1e9:.2f}GB, layers={len(layers)}")

    qa_files = list_qa_files()
    print(f"[plan] processing {len(qa_files)} QAs:")
    for f in qa_files:
        print(f"  - {f.name}")

    for qa_path in qa_files:
        try:
            process_one_qa(model, tokenizer, layers, qa_path)
        except Exception as e:
            print(f"[err] {qa_path.name}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    main()
