"""
深度探针：基于逻辑分岔点锚定的块级神经元差分。

与 probe_GB01_blockwise.py 的区别：
  - 使用新的锚点定义（deepseek_gemini_code_GB01_de.json）
  - 同时跑 Alist_v1 + Alist_v2 两组 A 锚点
  - 输出两套跨侧差分，取并集
  - 额外输出 A 侧 pivot 转折点前后的自差分（A_before vs A_after）

用法：
  python probe_GB01_deepseek_blockwise.py
"""
from __future__ import annotations

import difflib, gc, json, os, sys, time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

# ---------- config ----------
ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
DE_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "deepseek_gemini_code_GB01_de.json"
OUT_DIR = ROOT / "outputs" / "gemma_code_GB01_deepseek"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
DEV_IDX = 0
CHUNK = 1024
L_LO = 15
TOP_PER_ANCHOR = 20
TOP_CROSS_DIFF = 40

# ---------- text helpers ----------
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

# ---------- fuzzy anchor → token window ----------
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

def cross_diff(side_a_means, side_b_means, n_layers, sizes, label, top=TOP_CROSS_DIFF, l_lo=L_LO):
    """diff = max_abs_B - max_abs_A, signed by B's direction"""
    print(f"\n=== CROSS-SIDE DIFF: {label} (B unique, L>={l_lo}) ===")
    max_a = [torch.zeros(s) for s in sizes]
    max_b = [torch.zeros(s) for s in sizes]
    sgn_b = [torch.zeros(s) for s in sizes]
    for name, layer_means in side_a_means.items():
        for li in range(l_lo, n_layers):
            max_a[li] = torch.maximum(max_a[li], layer_means[li].abs())
    for name, layer_means in side_b_means.items():
        for li in range(l_lo, n_layers):
            v = layer_means[li]
            mag = v.abs()
            update = mag > max_b[li]
            max_b[li][update] = mag[update]
            sgn_b[li][update] = v[update].sign()
    pairs = []
    for li in range(l_lo, n_layers):
        sz = sizes[li]
        diff_score = max_b[li] - max_a[li]
        signed = diff_score * sgn_b[li]
        for n in range(sz):
            pairs.append((abs(diff_score[n].item()), li, n, signed[n].item(),
                          max_a[li][n].item(), max_b[li][n].item()))
    pairs.sort(reverse=True)
    print(f"  {'rank':>4} {'layer#nrn':>11} {'B_max':>7} {'A_max':>7} {'|diff|':>7} {'signed':>7}")
    for k, (_, li, n, sgn_v, am, bm) in enumerate(pairs[:top]):
        print(f"    {k+1:>2}  L{li:02d}#{n:<5d}  {bm:+.3f}  {am:+.3f}  {bm-am:+.3f}  {sgn_v:+.3f}")
    return pairs[:top]

def combine_neurons(pairs1, pairs2, top=TOP_CROSS_DIFF):
    """取 v1 和 v2 两组 diff 的并集（按 |diff| 重排）"""
    merged = {}
    for pairs in [pairs1, pairs2]:
        for _, li, n, sgn, am, bm in pairs:
            key = (li, n)
            if key not in merged or abs(sgn) > abs(merged[key][2]):
                merged[key] = (li, n, sgn, am, bm)
    combined = sorted(merged.values(), key=lambda x: abs(x[2]), reverse=True)
    print(f"\n=== COMBINED (v1 ∪ v2, top {top}) ===")
    print(f"  {'rank':>4} {'layer#nrn':>11} {'signed':>7} {'A_max':>7} {'B_max':>7}")
    for k, (li, n, sgn, am, bm) in enumerate(combined[:top]):
        print(f"    {k+1:>2}  L{li:02d}#{n:<5d}  {sgn:+.3f}  {am:+.3f}  {bm:+.3f}")
    return combined[:top]


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

    # ---- load data ----
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    de = json.loads(DE_PATH.read_text(encoding="utf-8"))
    item_a = data["A"][0]
    item_b = data["B"][0]
    a_anchors_v1 = de["Alist_v1"]
    a_anchors_v2 = de.get("Alist_v2", [])
    b_anchors = de["Blist"]
    pivot_info = de.get("A_pivot_window", None)

    # ---- build sequences ----
    full_a, P_a = build_sequence(tokenizer, item_a, soc, eoc, eot)
    full_b, P_b = build_sequence(tokenizer, item_b, soc, eoc, eot)
    print(f"[seq-A] total={full_a.shape[0]} prompt={P_a}")
    print(f"[seq-B] total={full_b.shape[0]} prompt={P_b}")

    # ---- locate anchor windows ----
    def locate_all(full, span, anchor_list, prefix):
        windows = []
        for i, anc in enumerate(anchor_list):
            name = f"{prefix}[{i+1}]"
            loc = find_anchor_token_window(tokenizer, full, span, anc)
            if loc is None:
                print(f"  {name}  MISS")
            else:
                lo, hi, _, _ = loc
                head = anc.replace("\n", "\\n")[:60]
                print(f"  {name}  tok[{lo:>4d},{hi:>4d}) ({hi-lo}t)  {head!r}")
                windows.append((name, lo, hi))
        return windows

    print("\n[anchor-A-v1] 混沌期句子...")
    a_wins_v1 = locate_all(full_a, P_a, a_anchors_v1, "A_v1")
    print("\n[anchor-A-v2] 字符串Digit DP打转...")
    a_wins_v2 = locate_all(full_a, P_a, a_anchors_v2, "A_v2")
    print("\n[anchor-B] 清晰数位DP分解...")
    b_wins = locate_all(full_b, P_b, b_anchors, "B")

    # ---- A 侧 pivot 转折点 (within-A) ----
    a_pivot_wins = []
    if pivot_info:
        anc = pivot_info["anchor"]
        wb = pivot_info.get("window_before", 15)
        wa = pivot_info.get("window_after", 5)
        loc = find_anchor_token_window(tokenizer, full_a, P_a, anc)
        if loc:
            lo, hi, _, _ = loc
            # before: [lo - wb, lo)
            lo_b = max(0, lo - wb)
            hi_b = lo
            # after: [hi, hi + wa)
            lo_a = hi
            hi_a = min(full_a.shape[0] - P_a, hi + wa)
            print(f"\n[anchor-A-pivot] '{anc[:60]}' -> before [{lo_b},{hi_b}) after [{lo_a},{hi_a})")
            a_pivot_wins.append(("A_pivot_before", lo_b, hi_b))
            a_pivot_wins.append(("A_pivot_after", lo_a, hi_a))
        else:
            print("\n[anchor-A-pivot] MISS")

    # ---- forward each side ----
    print("\n[fwd-A]...")
    t0 = time.time()
    a_means_all = forward_capture_anchors(model, layers, full_a, P_a, a_wins_v1 + a_wins_v2 + a_pivot_wins)
    print(f"  done in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()

    print("[fwd-B]...")
    t0 = time.time()
    b_means = forward_capture_anchors(model, layers, full_b, P_b, b_wins)
    print(f"  done in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()

    # ---- split A means back into groups ----
    nv1 = len(a_wins_v1)
    nv2 = len(a_wins_v2)
    a_means_v1 = {name: a_means_all[name] for name, _, _ in a_wins_v1}
    a_means_v2 = {name: a_means_all[name] for name, _, _ in a_wins_v2}
    a_pivot_means = {}
    for name, _, _ in a_pivot_wins:
        if name in a_means_all:
            a_pivot_means[name] = a_means_all[name]

    # ---- per-anchor reports (B only, because that's where the correct logic is) ----
    print("\n" + "=" * 60)
    print("B-SIDE PER-ANCHOR (correct reasoning checkpoints)")
    print("=" * 60)
    for (name, lo, hi) in b_wins:
        render_per_anchor_top(f"{name}  tok[{lo},{hi})", b_means[name], n_layers)

    # ---- cross-side diffs ----
    print("\n" + "=" * 60)
    print("CROSS-SIDE DIFFS")
    print("=" * 60)
    diff_v1 = cross_diff(a_means_v1, b_means, n_layers, sizes, "B vs A_v1 (混沌期)")
    diff_v2 = cross_diff(a_means_v2, b_means, n_layers, sizes, "B vs A_v2 (字符串DP)")
    combined = combine_neurons(diff_v1, diff_v2, top=TOP_CROSS_DIFF)

    # ---- A pivot self-diff (before vs after) ----
    if "A_pivot_before" in a_pivot_means and "A_pivot_after" in a_pivot_means:
        print("\n" + "=" * 60)
        print("A PIVOT SELF-DIFF: A_after - A_before (Gemma's own 'aha' moment)")
        print("=" * 60)
        sizes_a = [l.mlp.down_proj.in_features for l in layers]
        cross_diff(
            {"before": a_pivot_means["A_pivot_before"]},
            {"after": a_pivot_means["A_pivot_after"]},
            n_layers, sizes_a, "A_after vs A_before", top=TOP_CROSS_DIFF
        )

    # ---- save ----
    save_dict = {
        "qa_id": "gemma_code_GB01_deepseek_blockwise",
        "n_layers": n_layers,
        "sizes": sizes,
        "a_wins_v1": a_wins_v1, "a_wins_v2": a_wins_v2,
        "b_wins": b_wins,
        "a_pivot_wins": a_pivot_wins,
        "a_means_v1": a_means_v1, "a_means_v2": a_means_v2,
        "b_means": b_means,
        "a_pivot_means": a_pivot_means,
        "diff_v1_top": [(li, n, sgn) for (_, li, n, sgn, _, _) in diff_v1],
        "diff_v2_top": [(li, n, sgn) for (_, li, n, sgn, _, _) in diff_v2],
        "combined_top": [(li, n, sgn) for (li, n, sgn, _, _) in combined],
    }
    torch.save(save_dict, OUT_DIR / "blockwise.pt")
    print(f"\n[save] {OUT_DIR / 'blockwise.pt'}")


if __name__ == "__main__":
    main()