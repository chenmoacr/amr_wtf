"""
Block-wise per-anchor activation probe for gemma_code_GB01.

Goal: find neurons that fire strongly at B's correct-reasoning checkpoints
but NOT at A's broken-reasoning checkpoints.

Pipeline:
  1. Read anchor lists from gemma_code_GB01_de.json (Alist, Blist)
  2. Build A and B sequences with proper soc/eoc/eot wrapping
  3. For each anchor: find token window via fuzzy SequenceMatcher
     (no fail; we already verified 100% coverage)
  4. Single forward per side; per-anchor mean activation captured online
  5. Per-anchor top-20 neurons (L>=15) by |mean activation|
  6. Cross-side ranking:
        score_in_A[layer,nrn] = max_over_Alist_anchors |mean|
        score_in_B[layer,nrn] = max_over_Blist_anchors |mean|
        diff_score = score_in_B - score_in_A
     Top by |diff_score| reveals "B-uniquely-active" candidates.

Outputs:
  - stdout report
  - outputs/gemma_code_GB01/blockwise.pt with raw means
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
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
DE_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01_de.json"
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
DEV_IDX = 0
CHUNK = 1024
L_LO = 15
TOP_PER_ANCHOR = 20


# ---------- text helpers (same as probe_GB01.py) ----------
def strip_user_block(query: str) -> str:
    s = query
    if "[用户]" in s:
        s = s.split("[用户]", 1)[1]
    if "[助手]" in s:
        s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def strip_response_prefix(resp: str) -> str:
    s = resp.lstrip("\n")
    for prefix in ("<|channel|>", "<|channel>", "<channel|>", "<channel>"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.lstrip("\n")


def build_sequence(tokenizer, item, soc, eoc, eot):
    user_content = strip_user_block(item["query"])
    cot_text = item["cot"].strip("\n")
    resp_text = strip_response_prefix(item["response"])

    msgs = [{"role": "user", "content": user_content}]
    prompt_o = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    prompt_ids = prompt_o["input_ids"][0].tolist()

    th_intro = tokenizer.encode("thought\n", add_special_tokens=False)
    cot_body = tokenizer.encode(cot_text, add_special_tokens=False)
    nl = tokenizer.encode("\n", add_special_tokens=False)
    resp_ids = tokenizer.encode(resp_text, add_special_tokens=False)

    out = list(prompt_ids)
    out.append(soc)
    out.extend(th_intro)
    out.extend(cot_body)
    out.extend(nl)
    out.append(eoc)
    out.extend(resp_ids)
    out.append(eot)

    return torch.tensor(out, dtype=torch.long), len(prompt_ids)


# ---------- fuzzy anchor → token window (binary-search decode) ----------
def find_anchor_token_window(tokenizer, full_ids, span_start, anchor_text, fuzzy_min=0.5):
    span_ids = full_ids[span_start:].tolist()
    decoded = tokenizer.decode(span_ids, skip_special_tokens=False)

    char_start = decoded.find(anchor_text)
    if char_start >= 0:
        char_end = char_start + len(anchor_text)
    else:
        sm = difflib.SequenceMatcher(None, decoded, anchor_text, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size >= 10]
        if not blocks:
            return None
        matched_total = sum(b.size for b in blocks)
        if matched_total < fuzzy_min * len(anchor_text):
            return None
        char_start = min(b.a for b in blocks)
        char_end = max(b.a + b.size for b in blocks)

    # Binary search for tokens covering [char_start, char_end)
    def first_ti_ge(target_chars):
        lo, hi = 0, len(span_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            text = tokenizer.decode(span_ids[: mid + 1], skip_special_tokens=False)
            if len(text) >= target_chars:
                hi = mid
            else:
                lo = mid + 1
        return lo

    tok_lo = first_ti_ge(char_start + 1)
    tok_hi = first_ti_ge(char_end) + 1
    tok_hi = min(tok_hi, len(span_ids))
    return tok_lo, tok_hi, char_start, char_end


# ---------- forward + per-anchor capture ----------
def get_layers(model):
    return model.model.language_model.layers


def forward_capture_anchors(model, layers, full_ids, span_start, anchor_windows):
    """anchor_windows: list of (name, lo, hi) where lo/hi are span-local token indices.
    Returns: per_anchor[name][layer_idx] -> Tensor[hidden] (mean activation)."""
    T = full_ids.shape[0]
    sizes = [layer.mlp.down_proj.in_features for layer in layers]
    n_anc = len(anchor_windows)
    if n_anc == 0:
        return {}
    sums = [[torch.zeros(s, dtype=torch.float32) for _ in range(n_anc)] for s in sizes]
    counts = [0] * n_anc

    chunk_offset = {"v": 0}

    def make_hook(idx):
        def fn(module, inputs, output):
            x = inputs[0]
            chunk_len = x.shape[1]
            global_start = chunk_offset["v"]
            for ai, (name, lo, hi) in enumerate(anchor_windows):
                abs_lo = span_start + lo
                abs_hi = span_start + hi
                ov_lo = max(global_start, abs_lo)
                ov_hi = min(global_start + chunk_len, abs_hi)
                if ov_lo >= ov_hi:
                    continue
                local_lo = ov_lo - global_start
                local_hi = ov_hi - global_start
                seg = x[0, local_lo:local_hi, :].to(torch.float32)
                sums[idx][ai] += seg.sum(dim=0).cpu()
                if idx == 0:
                    counts[ai] += (local_hi - local_lo)
        return fn

    hooks = [layers[i].mlp.down_proj.register_forward_hook(make_hook(i)) for i in range(len(layers))]
    past_kv = None
    pos = 0
    try:
        while pos < T:
            end = min(pos + CHUNK, T)
            chunk_ids = full_ids[pos:end].unsqueeze(0).to(DEVICE)
            chunk_offset["v"] = pos
            cache_pos = torch.arange(pos, end, device=DEVICE)
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = model(input_ids=chunk_ids, past_key_values=past_kv,
                            use_cache=True, cache_position=cache_pos)
            past_kv = out.past_key_values
            pos = end
            torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    result = {}
    for ai, (name, _, _) in enumerate(anchor_windows):
        c = max(counts[ai], 1)
        result[name] = [sums[li][ai] / c for li in range(len(layers))]
    return result


# ---------- reporting ----------
def render_per_anchor_top(name, means_per_layer, n_layers, top=TOP_PER_ANCHOR, l_lo=L_LO):
    print(f"\n--- {name} ---")
    pairs = []
    for li in range(l_lo, n_layers):
        m = means_per_layer[li]
        for n in range(m.shape[0]):
            v = m[n].item()
            pairs.append((abs(v), li, n, v))
    pairs.sort(reverse=True)
    print(f"  {'rank':>4} {'layer#nrn':>11} {'mean':>7}")
    for k, (_, li, n, v) in enumerate(pairs[:top]):
        print(f"    {k+1:>2}  L{li:02d}#{n:<5d}  {v:+.3f}")


def cross_diff_top(a_per_anchor, b_per_anchor, n_layers, sizes, top=30, l_lo=L_LO):
    """For each (layer, neuron):
       score_A = max over Alist anchors of |mean|
       score_B = max over Blist anchors of |mean|
       diff_score = score_B - score_A
       (signed by direction of B's strongest hit)
    Returns sorted list of top |diff_score|."""
    print(f"\n=== CROSS-SIDE DIFF (B max-anchor strength minus A max-anchor strength, L>={l_lo}) ===")
    pairs = []
    for li in range(l_lo, n_layers):
        sz = sizes[li]
        max_a = torch.zeros(sz, dtype=torch.float32)
        max_b = torch.zeros(sz, dtype=torch.float32)
        sgn_b = torch.zeros(sz, dtype=torch.float32)
        for name, layer_means in a_per_anchor.items():
            mag = layer_means[li].abs()
            max_a = torch.maximum(max_a, mag)
        for name, layer_means in b_per_anchor.items():
            v = layer_means[li]
            mag = v.abs()
            update = mag > max_b
            max_b[update] = mag[update]
            sgn_b[update] = v[update].sign()
        diff_score = max_b - max_a
        signed = diff_score * sgn_b  # carry direction from B's strongest firing
        for n in range(sz):
            pairs.append((abs(diff_score[n].item()), li, n, signed[n].item(),
                          max_a[n].item(), max_b[n].item()))
    pairs.sort(reverse=True)
    print(f"  {'rank':>4} {'layer#nrn':>11} {'B_max':>7} {'A_max':>7} {'diff':>7} {'signed':>7}")
    for k, (_, li, n, sgn_v, am, bm) in enumerate(pairs[:top]):
        print(f"    {k+1:>2}  L{li:02d}#{n:<5d}  {bm:+.3f}  {am:+.3f}  {bm-am:+.3f}  {sgn_v:+.3f}")
    return pairs[:top]


def main():
    print("[load] tokenizer + model (BF16)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, low_cpu_mem_usage=True,
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
    print(f"[load] alloc={torch.cuda.memory_allocated(DEV_IDX)/1e9:.2f}GB layers={n_layers}")

    soc = tokenizer.convert_tokens_to_ids("<|channel>")
    eoc = tokenizer.convert_tokens_to_ids("<channel|>")
    eot = tokenizer.convert_tokens_to_ids("<turn|>")
    print(f"[tok] soc={soc} eoc={eoc} eot={eot}")

    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    de = json.loads(DE_PATH.read_text(encoding="utf-8"))
    item_a = data["A"][0]
    item_b = data["B"][0]
    a_anchors = de["Alist"]
    b_anchors = de["Blist"]

    # ---- build sequences ----
    full_a, P_a = build_sequence(tokenizer, item_a, soc, eoc, eot)
    full_b, P_b = build_sequence(tokenizer, item_b, soc, eoc, eot)
    print(f"[seq-A] total={full_a.shape[0]} prompt={P_a}  span={full_a.shape[0]-P_a}")
    print(f"[seq-B] total={full_b.shape[0]} prompt={P_b}  span={full_b.shape[0]-P_b}")

    # ---- locate anchor windows ----
    print("\n[anchor-A] locating...")
    a_windows = []
    for i, anc in enumerate(a_anchors):
        name = f"A[{i+1}]"
        loc = find_anchor_token_window(tokenizer, full_a, P_a, anc)
        if loc is None:
            print(f"  {name}  MISS (skipped)")
            continue
        lo, hi, cs, ce = loc
        head = anc.replace("\n", "\\n")[:60]
        print(f"  {name}  tok [{lo:>4d},{hi:>4d}) ({hi-lo}t)  char [{cs}..{ce})  {head!r}")
        a_windows.append((name, lo, hi))

    print("\n[anchor-B] locating...")
    b_windows = []
    for i, anc in enumerate(b_anchors):
        name = f"B[{i+1}]"
        loc = find_anchor_token_window(tokenizer, full_b, P_b, anc)
        if loc is None:
            print(f"  {name}  MISS (skipped)")
            continue
        lo, hi, cs, ce = loc
        head = anc.replace("\n", "\\n")[:60]
        print(f"  {name}  tok [{lo:>4d},{hi:>4d}) ({hi-lo}t)  char [{cs}..{ce})  {head!r}")
        b_windows.append((name, lo, hi))

    # ---- forward each side ----
    print("\n[fwd-A]...")
    t0 = time.time()
    a_means = forward_capture_anchors(model, layers, full_a, P_a, a_windows)
    print(f"  done in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()

    print("[fwd-B]...")
    t0 = time.time()
    b_means = forward_capture_anchors(model, layers, full_b, P_b, b_windows)
    print(f"  done in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()

    # ---- per-anchor reports ----
    print("\n" + "=" * 60)
    print("PER-ANCHOR TOP NEURONS (|mean activation|, L>=15)")
    print("=" * 60)
    print("\n[A side: broken-reasoning checkpoints]")
    for (name, lo, hi) in a_windows:
        render_per_anchor_top(f"{name}  tok[{lo},{hi})", a_means[name], n_layers, top=TOP_PER_ANCHOR)
    print("\n[B side: correct-reasoning checkpoints]")
    for (name, lo, hi) in b_windows:
        render_per_anchor_top(f"{name}  tok[{lo},{hi})", b_means[name], n_layers, top=TOP_PER_ANCHOR)

    # ---- cross-side diff ----
    print("\n" + "=" * 60)
    print("CROSS-SIDE: NEURONS UNIQUE TO B (correct) ANCHORS")
    print("=" * 60)
    top_pairs = cross_diff_top(a_means, b_means, n_layers, sizes, top=40, l_lo=L_LO)

    # ---- save ----
    torch.save({
        "qa_id": "gemma_code_GB01_blockwise",
        "n_layers": n_layers,
        "sizes": sizes,
        "a_windows": a_windows,
        "b_windows": b_windows,
        "a_means": a_means,
        "b_means": b_means,
        "top_diff": [(li, n, signed) for (_, li, n, signed, _, _) in top_pairs],
    }, OUT_DIR / "blockwise.pt")
    print(f"\n[save] {OUT_DIR / 'blockwise.pt'}")


if __name__ == "__main__":
    main()
