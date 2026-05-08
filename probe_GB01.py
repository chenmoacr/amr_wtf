"""
Paired minimal-edit differential for gemma_code_GB01.json.

Input JSON:
  - A: list of {query, cot, response}  — Gemma's wrong trajectory
  - B: list of {query, cot, response}  — Gemma's correct trajectory (after Gemini-rebuilt cot)
  - keypoints (optional): list of {name, side, anchor, window_before, window_after}

Pipeline:
  1. Strip stray <|channel|> prefixes from response
  2. Build A and B sequences with proper soc/eoc/eot wrapping + chat template prompt
  3. Use difflib.SequenceMatcher on token IDs to find matching/divergent blocks
  4. Label each token: 1=cot_common, 2=cot_divergent, 3=output_common, 4=output_divergent, 0=unlabeled
  5. Forward both via chunked KV cache; capture per-(layer, region, neuron) stats
  6. Diff B-A per region; report top neurons L>=15 per region
  7. For each keypoint anchor: capture per-layer top neurons in the token window
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
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
DEV_IDX = 0
ACTIVE_THRESH = 1.0
CHUNK = 1024
L_LO = 15

NUM_REGIONS = 5
REGION_NAMES = [
    "unlabeled",
    "cot_common", "cot_divergent",
    "output_common", "output_divergent",
]


# ---------- text helpers ----------
def strip_user_block(query: str) -> str:
    """Extract user content from a [用户]\\n... [助手]\\n style block."""
    s = query
    if "[用户]" in s:
        s = s.split("[用户]", 1)[1]
    if "[助手]" in s:
        s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def strip_response_prefix(resp: str) -> str:
    """Strip stray channel markers at the start of a response field."""
    s = resp.lstrip("\n")
    for prefix in ("<|channel|>", "<|channel>", "<channel|>", "<channel>"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.lstrip("\n")


# ---------- sequence build ----------
def build_sequence(tokenizer, item, soc, eoc, eot):
    """
    Returns:
      full_ids   : 1-D LongTensor [P + 1 + L_cot + 1 + L_resp + 1]
      P          : prompt length (tokens before assistant content)
      cot_lo, cot_hi : (in full_ids) bounds of cot inner content (between soc and eoc)
      resp_lo, resp_hi : bounds of response content (between eoc and eot)
    """
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
    cot_lo = len(out)
    out.extend(cot_body)
    cot_hi = len(out)
    out.extend(nl)
    out.append(eoc)
    resp_lo = len(out)
    out.extend(resp_ids)
    resp_hi = len(out)
    out.append(eot)

    return torch.tensor(out, dtype=torch.long), len(prompt_ids), cot_lo, cot_hi, resp_lo, resp_hi


# ---------- token-level diff -> labels ----------
def label_pair(a_ids: list, b_ids: list, common_label: int, divergent_label: int):
    """Run SequenceMatcher; return per-side label arrays. Match in (a-pos, b-pos, size)."""
    sm = difflib.SequenceMatcher(None, a_ids, b_ids, autojunk=False)
    blocks = sm.get_matching_blocks()
    a_lab = [divergent_label] * len(a_ids)
    b_lab = [divergent_label] * len(b_ids)
    for blk in blocks:
        if blk.size == 0:
            continue
        for k in range(blk.size):
            a_lab[blk.a + k] = common_label
            b_lab[blk.b + k] = common_label
    return a_lab, b_lab


def make_full_labels(seq_len, prompt_len, cot_lo, cot_hi, resp_lo, resp_hi, cot_lab, resp_lab):
    labels = [0] * seq_len  # unlabeled (prompt + special tokens)
    for i, lab in enumerate(cot_lab):
        labels[cot_lo + i] = lab
    for i, lab in enumerate(resp_lab):
        labels[resp_lo + i] = lab
    return torch.tensor(labels, dtype=torch.uint8)


# ---------- forward + region stats ----------
def get_layers(model):
    return model.model.language_model.layers


def forward_capture_regions(model, layers, full_ids, span_start, region_labels):
    T = full_ids.shape[0]
    span_len = T - span_start
    assert region_labels.shape[0] == span_len

    sizes = [layer.mlp.down_proj.in_features for layer in layers]
    sums = [[torch.zeros(s, dtype=torch.float32) for _ in range(NUM_REGIONS)] for s in sizes]
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
            mr.append(sums[li][r] / c)
            ar.append(active[li][r] / c)
        means.append(mr); rates.append(ar)
    return {"mean": means, "active_rate": rates, "counts": counts}


# ---------- per-token capture for keypoints ----------
def forward_capture_window(model, layers, full_ids, span_start, window_lo, window_hi):
    """Capture activations for every token in [window_lo, window_hi) of the assistant span.
    Returns per_layer[li] -> Tensor[window_len, hidden]."""
    T = full_ids.shape[0]
    sizes = [layer.mlp.down_proj.in_features for layer in layers]
    win_len = window_hi - window_lo
    out_buf = [torch.zeros(win_len, s, dtype=torch.float32) for s in sizes]

    chunk_offset = {"v": 0}

    def make_hook(idx):
        def fn(module, inputs, output):
            x = inputs[0]
            chunk_len = x.shape[1]
            global_start = chunk_offset["v"]
            chunk_lo = global_start
            chunk_hi = global_start + chunk_len
            abs_lo = span_start + window_lo
            abs_hi = span_start + window_hi
            ov_lo = max(chunk_lo, abs_lo)
            ov_hi = min(chunk_hi, abs_hi)
            if ov_lo >= ov_hi:
                return
            local_lo = ov_lo - chunk_lo
            local_hi = ov_hi - chunk_lo
            buf_lo = ov_lo - abs_lo
            buf_hi = buf_lo + (local_hi - local_lo)
            seg = x[0, local_lo:local_hi, :].to(torch.float32).cpu()
            out_buf[idx][buf_lo:buf_hi] = seg
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
    return out_buf


# ---------- reporting ----------
def render_top(diff, stats_a, stats_b, n_layers, region_id, label, top=15, l_lo=L_LO):
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


def find_anchor_token_window(tokenizer, full_ids, span_start, anchor_text, window_before, window_after):
    """Decode chunk-by-chunk and find anchor; return (window_lo, window_hi) in span-local indices."""
    span_ids = full_ids[span_start:].tolist()
    decoded = tokenizer.decode(span_ids, skip_special_tokens=False)
    char_pos = decoded.find(anchor_text)
    if char_pos < 0:
        return None
    cum_text = ""
    target_token = None
    for ti, t in enumerate(span_ids):
        cum_text = tokenizer.decode(span_ids[:ti + 1], skip_special_tokens=False)
        if len(cum_text) > char_pos:
            target_token = ti
            break
    if target_token is None:
        return None
    lo = max(0, target_token - window_before)
    hi = min(len(span_ids), target_token + window_after)
    return lo, hi, target_token


def render_keypoint(name, side, anchor, target_token, win_a, win_b, full_a, full_b, span_a, span_b, tokenizer, n_layers, top=10):
    print(f"\n=== KEYPOINT '{name}' (side={side}, anchor={anchor!r}) ===")
    if win_a is None and win_b is None:
        print("  [skip] anchor not found in either side")
        return
    if win_a is not None:
        lo_a, hi_a, tgt_a = win_a
        ctx_a = tokenizer.decode(full_a[span_a + lo_a: span_a + hi_a].tolist(), skip_special_tokens=False)
        print(f"  A window [{lo_a}..{hi_a}) target={tgt_a}\n    A: ...{ctx_a[:300]}...")
    if win_b is not None:
        lo_b, hi_b, tgt_b = win_b
        ctx_b = tokenizer.decode(full_b[span_b + lo_b: span_b + hi_b].tolist(), skip_special_tokens=False)
        print(f"  B window [{lo_b}..{hi_b}) target={tgt_b}\n    B: ...{ctx_b[:300]}...")


def keypoint_window_diff(act_a_per_layer, act_b_per_layer, l_lo=L_LO, top=15):
    """Mean over the window per side, then diff. act_*_per_layer[li] = [win_len, hidden]."""
    n_layers = len(act_a_per_layer)
    pairs = []
    for li in range(l_lo, n_layers):
        ma = act_a_per_layer[li].mean(dim=0)
        mb = act_b_per_layer[li].mean(dim=0)
        d = mb - ma
        for n in range(d.shape[0]):
            pairs.append((abs(d[n].item()), li, n, d[n].item()))
    pairs.sort(reverse=True)
    print(f"  {'rank':>4} {'layer#nrn':>11} {'diff':>7}  (window-mean B - A)")
    for k, (_, li, n, v) in enumerate(pairs[:top]):
        print(f"    {k+1:>2}  L{li:02d}#{n:<5d}  {v:+.3f}")


# ---------- main ----------
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
    print(f"[load] alloc={torch.cuda.memory_allocated(DEV_IDX)/1e9:.2f}GB layers={n_layers}")

    soc = tokenizer.convert_tokens_to_ids("<|channel>")
    eoc = tokenizer.convert_tokens_to_ids("<channel|>")
    eot = tokenizer.convert_tokens_to_ids("<turn|>")
    print(f"[tok] soc={soc} eoc={eoc} eot={eot}")

    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    item_a = data["A"][0]
    item_b = data["B"][0]
    keypoints = data.get("keypoints", [])

    # ---- build sequences ----
    full_a, P_a, cot_lo_a, cot_hi_a, resp_lo_a, resp_hi_a = build_sequence(tokenizer, item_a, soc, eoc, eot)
    full_b, P_b, cot_lo_b, cot_hi_b, resp_lo_b, resp_hi_b = build_sequence(tokenizer, item_b, soc, eoc, eot)
    if P_a != P_b:
        print(f"[warn] prompt lengths differ: A={P_a} B={P_b}. Using A's prompt as span_start anchor.")
    print(f"[seq-A] total={full_a.shape[0]} prompt={P_a} cot=[{cot_lo_a},{cot_hi_a}) resp=[{resp_lo_a},{resp_hi_a})")
    print(f"[seq-B] total={full_b.shape[0]} prompt={P_b} cot=[{cot_lo_b},{cot_hi_b}) resp=[{resp_lo_b},{resp_hi_b})")

    # ---- token-level diff in cot and response ----
    cot_a_ids = full_a[cot_lo_a:cot_hi_a].tolist()
    cot_b_ids = full_b[cot_lo_b:cot_hi_b].tolist()
    resp_a_ids = full_a[resp_lo_a:resp_hi_a].tolist()
    resp_b_ids = full_b[resp_lo_b:resp_hi_b].tolist()

    cot_lab_a, cot_lab_b = label_pair(cot_a_ids, cot_b_ids, common_label=1, divergent_label=2)
    resp_lab_a, resp_lab_b = label_pair(resp_a_ids, resp_b_ids, common_label=3, divergent_label=4)

    cot_match_a = sum(1 for x in cot_lab_a if x == 1)
    cot_div_a = len(cot_lab_a) - cot_match_a
    cot_match_b = sum(1 for x in cot_lab_b if x == 1)
    cot_div_b = len(cot_lab_b) - cot_match_b
    resp_match_a = sum(1 for x in resp_lab_a if x == 3)
    resp_div_a = len(resp_lab_a) - resp_match_a
    resp_match_b = sum(1 for x in resp_lab_b if x == 3)
    resp_div_b = len(resp_lab_b) - resp_match_b
    print(f"[diff] cot:    A_common={cot_match_a} A_diverg={cot_div_a} | B_common={cot_match_b} B_diverg={cot_div_b}")
    print(f"[diff] output: A_common={resp_match_a} A_diverg={resp_div_a} | B_common={resp_match_b} B_diverg={resp_div_b}")

    span_a = P_a
    span_b = P_b
    labels_a = make_full_labels(full_a.shape[0] - span_a, P_a,
                                cot_lo_a - span_a, cot_hi_a - span_a,
                                resp_lo_a - span_a, resp_hi_a - span_a,
                                cot_lab_a, resp_lab_a)
    labels_b = make_full_labels(full_b.shape[0] - span_b, P_b,
                                cot_lo_b - span_b, cot_hi_b - span_b,
                                resp_lo_b - span_b, resp_hi_b - span_b,
                                cot_lab_b, resp_lab_b)

    counts_a = torch.bincount(labels_a.long(), minlength=NUM_REGIONS).tolist()
    counts_b = torch.bincount(labels_b.long(), minlength=NUM_REGIONS).tolist()
    print(f"[label A] " + ", ".join(f"{REGION_NAMES[i]}={counts_a[i]}" for i in range(NUM_REGIONS)))
    print(f"[label B] " + ", ".join(f"{REGION_NAMES[i]}={counts_b[i]}" for i in range(NUM_REGIONS)))

    # ---- forward both ----
    print("[fwd-A]...")
    t0 = time.time()
    stats_a = forward_capture_regions(model, layers, full_a, span_start=span_a, region_labels=labels_a)
    print(f"  done in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()

    print("[fwd-B]...")
    t0 = time.time()
    stats_b = forward_capture_regions(model, layers, full_b, span_start=span_b, region_labels=labels_b)
    print(f"  done in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()

    # ---- diff per region ----
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

    # ---- keypoint analysis ----
    if keypoints:
        print("\n" + "=" * 50)
        print("KEYPOINT ANALYSIS")
        print("=" * 50)
        for kp in keypoints:
            name = kp.get("name", "?")
            anchor = kp.get("anchor", "")
            wb = int(kp.get("window_before", 5))
            wa = int(kp.get("window_after", 30))

            win_a = find_anchor_token_window(tokenizer, full_a, span_a, anchor, wb, wa)
            win_b = find_anchor_token_window(tokenizer, full_b, span_b, anchor, wb, wa)

            render_keypoint(name, "both", anchor, None, win_a, win_b, full_a, full_b, span_a, span_b,
                            tokenizer, n_layers, top=10)

            if win_a is not None and win_b is not None:
                lo_a, hi_a, _ = win_a
                lo_b, hi_b, _ = win_b
                # Match the smaller window length so we can do per-position diff if desired
                act_a = forward_capture_window(model, layers, full_a, span_a, lo_a, hi_a)
                act_b = forward_capture_window(model, layers, full_b, span_b, lo_b, hi_b)
                # Truncate to common length
                w = min(hi_a - lo_a, hi_b - lo_b)
                act_a_t = [a[:w] for a in act_a]
                act_b_t = [b[:w] for b in act_b]
                keypoint_window_diff(act_a_t, act_b_t, l_lo=L_LO, top=15)

    # ---- save ----
    torch.save({
        "qa_id": "gemma_code_GB01",
        "n_layers": n_layers,
        "span_a": span_a, "span_b": span_b,
        "stats_a": stats_a, "stats_b": stats_b,
        "diff_per_region": diff,
        "region_names": REGION_NAMES,
        "counts_a": counts_a, "counts_b": counts_b,
        "labels_a": labels_a, "labels_b": labels_b,
        "full_a": full_a, "full_b": full_b,
        "cot_a_ids": cot_a_ids, "cot_b_ids": cot_b_ids,
        "resp_a_ids": resp_a_ids, "resp_b_ids": resp_b_ids,
    }, OUT_DIR / "diff.pt")
    print(f"\n[save] {OUT_DIR / 'diff.pt'}")


if __name__ == "__main__":
    main()
