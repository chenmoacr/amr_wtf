"""
QA02 (LeetCode 233 Number of Digit One) region-aware + plain differential.

Input JSON: gemini_3_flash_preview_code_qa_02.json
  - cot              : single paragraph -> all tokens label = T_glue (no fine-grained)
  - output           : C++ answer with conclusion_analysis + glue_sentences
  - conclusion_analysis.depth_{1,2,3}_*  -> A_d1 / A_d2 / A_d3
  - glue_sentences   -> A_glue
  - unmatched answer tokens (mostly raw C++ code lines) -> A_code  (NEW)

Outputs:
  - per-region top neurons (T_glue, A_glue, A_d1, A_d2, A_d3, A_code)
  - aggregate "A_all" = weighted mean over all answer regions (plain QA diff)
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemini_3_flash_preview_code_qa_02.json"
OUT_DIR = ROOT / "outputs" / "code_region_qa02"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
DEV_IDX = 0
MAX_NEW_TOKENS = 4500
ACTIVE_THRESH = 1.0
CHUNK = 1024
MIN_BLOCK = 6
MIN_COVERAGE = 0.5

NUM_REGIONS = 10
REGION_NAMES = [
    "unlabeled",
    "T_glue", "T_d1", "T_d2", "T_d3",
    "A_glue", "A_d1", "A_d2", "A_d3",
    "A_code",
]
T_BASE = 1
A_BASE = 5  # A_glue=5, A_d1=6, A_d2=7, A_d3=8, A_code=9


def _exact_spans(text, needle):
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


def find_fuzzy_spans(text, sentence):
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


def label_thought_tokens_flat(offsets):
    """cot is a single paragraph; everything -> T_glue (label=1)."""
    n = len(offsets)
    return torch.full((n,), T_BASE, dtype=torch.uint8)


def label_answer_tokens(segment_text, offsets, conclusion, glue, base):
    """Like the QA01 labeller, but unmatched tokens get base+4 (A_code)
    instead of staying at 0 (unlabeled). This way every answer token is
    accounted for, and the C++ code body shows up in A_code."""
    n = len(offsets)
    labels = torch.zeros(n, dtype=torch.uint8)
    found = {"glue": 0, "d1": 0, "d2": 0, "d3": 0}
    missed = {"glue": 0, "d1": 0, "d2": 0, "d3": 0}

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
            spans = find_fuzzy_spans(segment_text, s)
            if spans is None:
                missed[key] += 1
                continue
            for lo, hi in spans:
                mark_span(label_id, lo, hi)
            found[key] += 1

    mark(glue, base + 0, "glue")
    mark(conclusion.get("depth_1_technical_surface", []), base + 1, "d1")
    mark(conclusion.get("depth_2_structural_narrative", []), base + 2, "d2")
    mark(conclusion.get("depth_3_core_thematic", []), base + 3, "d3")

    code_label = base + 4
    for i in range(n):
        if int(labels[i]) == 0:
            labels[i] = code_label

    return labels, found, missed


def get_layers(model):
    return model.model.language_model.layers


def build_prompt_ids(tokenizer, user_input):
    msgs = [{"role": "user", "content": user_input}]
    o = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    return o["input_ids"][0]


def gen_gemma(model, tokenizer, prompt_ids):
    eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    eos_ids = [i for i in eos_ids if i is not None and i >= 0]
    print(f"  [gen] prompt_len={prompt_ids.shape[0]}")
    t0 = time.time()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.generate(
            prompt_ids.unsqueeze(0).to(DEVICE),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=1.0, top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )
    full = out[0].cpu()
    gen_ids = full[prompt_ids.shape[0]:]
    print(f"  [gen] {gen_ids.shape[0]} tokens in {time.time()-t0:.1f}s")
    return gen_ids


def build_reference_and_labels(tokenizer, qa_data):
    soc = tokenizer.convert_tokens_to_ids("<|channel>")
    eoc = tokenizer.convert_tokens_to_ids("<channel|>")
    eot = tokenizer.convert_tokens_to_ids("<turn|>")

    cot_text = qa_data["cot"]
    out_text = qa_data["output"]

    cot_enc = tokenizer(cot_text, add_special_tokens=False, return_offsets_mapping=True)
    cot_ids = cot_enc["input_ids"]
    cot_offsets = cot_enc["offset_mapping"]
    cot_labels = label_thought_tokens_flat(cot_offsets)
    print(f"  [label B-thought] cot_tokens={len(cot_ids)} all->T_glue")

    out_enc = tokenizer(out_text, add_special_tokens=False, return_offsets_mapping=True)
    out_ids = out_enc["input_ids"]
    out_offsets = out_enc["offset_mapping"]
    out_labels, out_found, out_missed = label_answer_tokens(
        out_text, out_offsets,
        qa_data.get("conclusion_analysis", {}),
        qa_data.get("glue_sentences", []),
        base=A_BASE,
    )
    print(f"  [label B-answer] found={out_found}, missed={out_missed}")

    th_intro = tokenizer.encode("thought\n", add_special_tokens=False)
    nl = tokenizer.encode("\n", add_special_tokens=False)

    full_ids = []
    full_labels = []

    full_ids.append(soc); full_labels.append(0)
    for t in th_intro:
        full_ids.append(t); full_labels.append(0)
    thought_lo = len(full_ids)
    for tid, lab in zip(cot_ids, cot_labels.tolist()):
        full_ids.append(tid); full_labels.append(lab)
    thought_hi = len(full_ids)
    for t in nl:
        full_ids.append(t); full_labels.append(0)
    full_ids.append(eoc); full_labels.append(0)
    answer_lo = len(full_ids)
    for tid, lab in zip(out_ids, out_labels.tolist()):
        full_ids.append(tid); full_labels.append(lab)
    answer_hi = len(full_ids)
    full_ids.append(eot); full_labels.append(0)

    return (
        torch.tensor(full_ids, dtype=torch.long),
        torch.tensor(full_labels, dtype=torch.uint8),
        thought_lo, thought_hi, answer_lo, answer_hi,
    )


def find_a_segments(gen_ids, soc_id, eoc_id, eot_id):
    eoc_pos = (gen_ids == eoc_id).nonzero(as_tuple=True)[0]
    if eoc_pos.numel() == 0:
        return None
    eoc_idx = int(eoc_pos[0].item())
    soc_pos = (gen_ids == soc_id).nonzero(as_tuple=True)[0]
    soc_idx = int(soc_pos[0].item()) if soc_pos.numel() > 0 else 0
    thought_lo = soc_idx + 1
    thought_hi = eoc_idx
    eot_pos = (gen_ids == eot_id).nonzero(as_tuple=True)[0]
    answer_lo = eoc_idx + 1
    answer_hi = int(eot_pos[0].item()) if eot_pos.numel() > 0 else gen_ids.shape[0]
    return thought_lo, thought_hi, answer_lo, answer_hi


def forward_capture_regions(model, layers, full_ids, span_start, region_labels):
    T = full_ids.shape[0]
    span_len = T - span_start
    assert region_labels.shape[0] == span_len

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
            seg = x[0, local_lo:, :].to(torch.float32)
            seg_abs_start = global_start + local_lo
            rel_lo = seg_abs_start - span_start
            rel_hi = rel_lo + seg.shape[0]
            seg_labels = region_labels_gpu[rel_lo:rel_hi]
            for r in range(NUM_REGIONS):
                mask = (seg_labels == r)
                cnt = int(mask.sum().item())
                if cnt == 0:
                    continue
                sub = seg[mask, :]
                sums[idx][r] += sub.sum(dim=0).cpu()
                sumsq[idx][r] += (sub * sub).sum(dim=0).cpu()
                active[idx][r] += (sub.abs() > ACTIVE_THRESH).to(torch.float32).sum(dim=0).cpu()
                counts[idx][r] += cnt
        return fn

    hooks = [layer.mlp.down_proj.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    past_kv = None
    pos = 0
    try:
        while pos < T:
            end = min(pos + CHUNK, T)
            chunk_ids = full_ids[pos:end].unsqueeze(0).to(DEVICE)
            chunk_offset["v"] = pos
            cache_pos = torch.arange(pos, end, device=DEVICE)
            with torch.no_grad():
                out = model(input_ids=chunk_ids, past_key_values=past_kv,
                            use_cache=True, cache_position=cache_pos)
            past_kv = out.past_key_values
            pos = end
            torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    means, rates = [], []
    for li in range(len(layers)):
        mr, ar = [], []
        for r in range(NUM_REGIONS):
            c = counts[li][r]
            if c == 0:
                mr.append(None); ar.append(None); continue
            m = sums[li][r] / c
            mr.append(m); ar.append(active[li][r] / c)
        means.append(mr); rates.append(ar)
    return {"mean": means, "active_rate": rates, "counts": counts}


def render_top(diff, stats_a, stats_b, n_layers, region_id, label, top=15, l_lo=15):
    print(f"\n=== Top |diff| in '{label}' (region {region_id}, L>={l_lo}, top {top}) ===")
    pairs = []
    for li in range(l_lo, n_layers):
        d = diff[li][region_id]
        if d is None:
            continue
        for n in range(d.shape[0]):
            pairs.append((abs(d[n].item()), li, n, d[n].item()))
    pairs.sort(reverse=True)
    print(f"{'rank':>4} {'layer#nrn':>11} {'diff':>7} {'rate_A':>7} {'rate_B':>7}")
    for k, (_, li, n, v) in enumerate(pairs[:top]):
        ra = stats_a["active_rate"][li][region_id]
        rb = stats_b["active_rate"][li][region_id]
        ra_v = ra[n].item() if ra is not None else float("nan")
        rb_v = rb[n].item() if rb is not None else float("nan")
        print(f"  {k+1:>2}  L{li:02d}#{n:<5d}  {v:+.3f}  {ra_v:.3f}  {rb_v:.3f}")


def render_top_aggregate(stats_a, stats_b, n_layers, region_ids, label, top=20, l_lo=15):
    """Plain QA diff: weighted mean over multiple regions."""
    print(f"\n=== Top |diff| in '{label}' (regions={region_ids}, L>={l_lo}, top {top}) ===")
    pairs = []
    for li in range(l_lo, n_layers):
        total_a = sum(stats_a["counts"][li][r] for r in region_ids)
        total_b = sum(stats_b["counts"][li][r] for r in region_ids)
        if total_a == 0 or total_b == 0:
            continue
        sum_a = None
        sum_b = None
        for r in region_ids:
            ma = stats_a["mean"][li][r]
            mb = stats_b["mean"][li][r]
            ca = stats_a["counts"][li][r]
            cb = stats_b["counts"][li][r]
            if ma is not None and ca > 0:
                sum_a = ma * ca if sum_a is None else sum_a + ma * ca
            if mb is not None and cb > 0:
                sum_b = mb * cb if sum_b is None else sum_b + mb * cb
        if sum_a is None or sum_b is None:
            continue
        mean_a = sum_a / total_a
        mean_b = sum_b / total_b
        d = mean_b - mean_a
        for n in range(d.shape[0]):
            pairs.append((abs(d[n].item()), li, n, d[n].item()))
    pairs.sort(reverse=True)
    print(f"{'rank':>4} {'layer#nrn':>11} {'diff':>7}")
    for k, (_, li, n, v) in enumerate(pairs[:top]):
        print(f"  {k+1:>2}  L{li:02d}#{n:<5d}  {v:+.3f}")


def main():
    print("[load] tokenizer + model (BF16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
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
    gc.collect(); torch.cuda.empty_cache()
    layers = get_layers(model)
    n_layers = len(layers)
    sizes = [l.mlp.down_proj.in_features for l in layers]
    print(f"[load] alloc={torch.cuda.memory_allocated(DEV_IDX)/1e9:.2f}GB, layers={n_layers}")

    qa_data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_input = qa_data["input"]

    soc_id = tokenizer.convert_tokens_to_ids("<|channel>")
    eoc_id = tokenizer.convert_tokens_to_ids("<channel|>")
    eot_id = tokenizer.convert_tokens_to_ids("<turn|>")
    print(f"[tok] soc={soc_id}, eoc={eoc_id}, eot={eot_id}")

    prompt_ids = build_prompt_ids(tokenizer, user_input)
    P = prompt_ids.shape[0]
    print(f"[prompt] tokens={P}")

    cache_path = ROOT / "outputs" / "thought_qa02" / "gemma_full.pt"
    if cache_path.exists():
        gemma_gen = torch.load(cache_path, weights_only=True)
        print(f"[reuse] {cache_path} -> {gemma_gen.shape[0]} tokens")
    else:
        print("[gen-A] generating Gemma free output (this is a one-time step)...")
        gemma_gen = gen_gemma(model, tokenizer, prompt_ids)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(gemma_gen, cache_path)

    a_seg = find_a_segments(gemma_gen, soc_id, eoc_id, eot_id)
    if a_seg is None:
        print("[err] could not find segments in Gemma generation (no <channel|>?)")
        return
    a_th_lo, a_th_hi, a_an_lo, a_an_hi = a_seg
    print(f"[seg-A] thought=[{a_th_lo},{a_th_hi}) ({a_th_hi-a_th_lo}t), answer=[{a_an_lo},{a_an_hi}) ({a_an_hi-a_an_lo}t)")

    print("[ref] building reference assistant ids + labels...")
    ref_ids, ref_labels, b_th_lo, b_th_hi, b_an_lo, b_an_hi = build_reference_and_labels(
        tokenizer, qa_data,
    )
    print(f"[seg-B] thought=[{b_th_lo},{b_th_hi}) ({b_th_hi-b_th_lo}t), answer=[{b_an_lo},{b_an_hi}) ({b_an_hi-b_an_lo}t)")

    th_min = min(a_th_hi - a_th_lo, b_th_hi - b_th_lo)
    an_min = min(a_an_hi - a_an_lo, b_an_hi - b_an_lo)
    print(f"[align] thought_min={th_min}, answer_min={an_min}")

    a_span_len = gemma_gen.shape[0]
    a_labels = torch.zeros(a_span_len, dtype=torch.uint8)
    a_labels[a_th_lo:a_th_lo + th_min] = ref_labels[b_th_lo:b_th_lo + th_min]
    a_labels[a_an_lo:a_an_lo + an_min] = ref_labels[b_an_lo:b_an_lo + an_min]

    b_span_len = ref_ids.shape[0]
    b_labels_aligned = torch.zeros(b_span_len, dtype=torch.uint8)
    b_labels_aligned[b_th_lo:b_th_lo + th_min] = ref_labels[b_th_lo:b_th_lo + th_min]
    b_labels_aligned[b_an_lo:b_an_lo + an_min] = ref_labels[b_an_lo:b_an_lo + an_min]

    counts_a = torch.bincount(a_labels.long(), minlength=NUM_REGIONS).tolist()
    counts_b = torch.bincount(b_labels_aligned.long(), minlength=NUM_REGIONS).tolist()
    print(f"[label A] " + ", ".join(f"{REGION_NAMES[i]}={counts_a[i]}" for i in range(NUM_REGIONS)))
    print(f"[label B] " + ", ".join(f"{REGION_NAMES[i]}={counts_b[i]}" for i in range(NUM_REGIONS)))

    seq_a = torch.cat([prompt_ids, gemma_gen], dim=0)
    seq_b = torch.cat([prompt_ids, ref_ids], dim=0)
    print(f"[seq] A_len={seq_a.shape[0]}, B_len={seq_b.shape[0]}")

    print("[fwd-A]...")
    stats_a = forward_capture_regions(model, layers, seq_a, span_start=P, region_labels=a_labels)
    torch.cuda.empty_cache()

    print("[fwd-B]...")
    stats_b = forward_capture_regions(model, layers, seq_b, span_start=P, region_labels=b_labels_aligned)
    torch.cuda.empty_cache()

    diff = []
    for li in range(n_layers):
        per_r = []
        for r in range(NUM_REGIONS):
            ma = stats_a["mean"][li][r]
            mb = stats_b["mean"][li][r]
            per_r.append(None if (ma is None or mb is None) else (mb - ma))
        diff.append(per_r)

    for r in range(1, NUM_REGIONS):
        if counts_a[r] >= 6 and counts_b[r] >= 6:
            render_top(diff, stats_a, stats_b, n_layers, r, REGION_NAMES[r], top=15)
        else:
            print(f"\n[skip] region {REGION_NAMES[r]}: counts A={counts_a[r]} B={counts_b[r]} too small")

    render_top_aggregate(stats_a, stats_b, n_layers,
                         region_ids=[5, 6, 7, 8, 9],
                         label="A_all (plain QA diff over entire answer)",
                         top=20)

    torch.save(
        {
            "qa_id": "code_region_qa02",
            "n_layers": n_layers,
            "sizes": sizes,
            "prompt_len": P,
            "stats_a": stats_a,
            "stats_b": stats_b,
            "diff_per_region": diff,
            "region_names": REGION_NAMES,
            "counts_a": counts_a,
            "counts_b": counts_b,
            "labels_a": a_labels,
            "labels_b": b_labels_aligned,
            "a_segments": (a_th_lo, a_th_hi, a_an_lo, a_an_hi),
            "b_segments": (b_th_lo, b_th_hi, b_an_lo, b_an_hi),
            "thought_min": th_min,
            "answer_min": an_min,
        },
        OUT_DIR / "diff_region.pt",
    )
    print(f"\n[save] {OUT_DIR / 'diff_region.pt'}")


if __name__ == "__main__":
    main()
