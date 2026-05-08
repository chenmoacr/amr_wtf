"""
D. Attribution patching on GB01.

Metric M = sum of activation[L33, 9054] over B's anchor windows.
For each target_li in 0..33, we anchor that layer's MLP intermediate
(detach + requires_grad_(True), replace input to down_proj), forward B,
backward M, read grad of the anchor.

attribution_pure[L, n] = sum_t  grad[L, t, n] * h_B[L, t, n]
attribution_diff[L, n] = sum_t  grad[L, t, n] * (h_B[L, t, n] - mean_t' h_A[L, n])

Top-K |attribution_diff| = neurons whose B-vs-A difference most causally
drives M. This is the "circuit" feeding into L33#9054 in the correct chain.
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

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
DE_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01_de.json"
NEURONS_JSON = ROOT / "chat" / "neurons.json"
OUT_DIR = ROOT / "outputs" / "circuit_attr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
TARGET_LI = 33
TARGET_NRN = 9054
TOP_K = 30


# ---------- text helpers (match probe_GB01_blockwise.py exactly) ----------
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
    return tok_lo, tok_hi


# ---------- attribution per target layer ----------
def attribute_one_layer(model, layers, target_li, B_input_ids, anchor_abs_ranges,
                        metric_li, target_nrn):
    """Forward B with one anchor at target_li; backward M (sum over anchor positions of metric);
    return (act, grad) at target_li, both shape [1, T, intermediate]."""
    target_holder = {"h": None}
    metric_holder = {"h": None}

    def target_hook(module, inputs):
        x = inputs[0]
        h_new = x.detach().clone().requires_grad_(True)
        target_holder["h"] = h_new
        return (h_new,)

    def metric_hook(module, inputs):
        metric_holder["h"] = inputs[0]

    h1 = layers[target_li].mlp.down_proj.register_forward_pre_hook(target_hook)
    h2 = None
    if metric_li != target_li:
        h2 = layers[metric_li].mlp.down_proj.register_forward_pre_hook(metric_hook)
    try:
        with torch.enable_grad(), sdpa_kernel([SDPBackend.MATH]):
            model(input_ids=B_input_ids.unsqueeze(0).to(DEVICE), use_cache=False)
        if metric_li == target_li:
            metric_t = target_holder["h"]
        else:
            metric_t = metric_holder["h"]
        slices = []
        for lo, hi in anchor_abs_ranges:
            slices.append(metric_t[0, lo:hi, target_nrn])
        M = torch.cat(slices).sum()
        M.backward()
    finally:
        h1.remove()
        if h2 is not None:
            h2.remove()
    h = target_holder["h"]
    return h.detach().cpu(), h.grad.detach().cpu()


def capture_act_only(model, layers, ids):
    """Run forward (no grad), capture mean activation per layer at down_proj input."""
    saved = {}
    def make_hook(li):
        def fn(module, inputs):
            saved[li] = inputs[0].detach().mean(dim=1).squeeze(0).cpu()  # [intermediate]
        return fn
    hooks = [layers[i].mlp.down_proj.register_forward_pre_hook(make_hook(i)) for i in range(len(layers))]
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
            model(input_ids=ids.unsqueeze(0).to(DEVICE), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return saved


def main():
    print(f"[load] {MODEL_PATH}  (BF16)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, low_cpu_mem_usage=True,
    )
    model.eval()
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.language_model.layers
    n_layers = len(layers)
    print(f"  layers={n_layers}, alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB")

    soc = tokenizer.convert_tokens_to_ids("<|channel>")
    eoc = tokenizer.convert_tokens_to_ids("<channel|>")
    eot = tokenizer.convert_tokens_to_ids("<turn|>")
    print(f"[tok] soc={soc} eoc={eoc} eot={eot}")

    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    de = json.loads(DE_PATH.read_text(encoding="utf-8"))
    item_a = data["A"][0]
    item_b = data["B"][0]
    b_anchors = de["Blist"]

    full_a, P_a = build_sequence(tokenizer, item_a, soc, eoc, eot)
    full_b, P_b = build_sequence(tokenizer, item_b, soc, eoc, eot)
    print(f"[seq-A] T={full_a.shape[0]} (prompt={P_a})")
    print(f"[seq-B] T={full_b.shape[0]} (prompt={P_b})")

    # Locate B anchor windows -> absolute token positions on full_b
    anchor_abs = []
    print(f"\n[B-anchors] locating in B sequence...")
    for i, anc in enumerate(b_anchors):
        loc = find_anchor_token_window(tokenizer, full_b, P_b, anc)
        if loc is None:
            print(f"  B[{i+1}]  MISS")
            continue
        lo, hi = loc
        abs_lo, abs_hi = P_b + lo, P_b + hi
        head = anc.replace("\n", "\\n")[:50]
        print(f"  B[{i+1}]  abs_tok[{abs_lo:>4d},{abs_hi:>4d})  {head!r}")
        anchor_abs.append((abs_lo, abs_hi))
    if not anchor_abs:
        print("[fatal] no B anchors located"); return

    # ---------- baseline pass: A's mean activation per layer ----------
    print("\n[A-baseline] capturing per-layer mean activation on A...")
    t0 = time.time()
    a_means = capture_act_only(model, layers, full_a)
    print(f"  done in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()

    # ---------- main loop: per-layer attribution on B ----------
    # Save attribution arrays per target layer
    attribution_pure = {}   # li -> [intermediate]  (sum_t grad * act)
    attribution_diff = {}   # li -> [intermediate]  (sum_t grad * (act - a_mean))

    print(f"\n[attribution] target metric: M = sum over B-anchors of act[L{TARGET_LI}, #{TARGET_NRN}]")
    print(f"  (running per-layer forward+backward, target_li=0..{TARGET_LI})")

    for tli in range(TARGET_LI + 1):
        torch.cuda.empty_cache()
        gc.collect()
        try:
            t0 = time.time()
            h_act, h_grad = attribute_one_layer(
                model, layers, tli, full_b, anchor_abs, TARGET_LI, TARGET_NRN,
            )
            # h_act, h_grad: [1, T, intermediate]
            act = h_act[0].float()    # [T, intermediate]
            grad = h_grad[0].float()
            # pure attribution: sum_t grad * act
            pure = (grad * act).sum(dim=0)    # [intermediate]
            # diff attribution: sum_t grad * (act - a_mean[tli])
            diff = (grad * (act - a_means[tli].unsqueeze(0).float())).sum(dim=0)
            attribution_pure[tli] = pure
            attribution_diff[tli] = diff
            dt = time.time() - t0
            top1_pure = pure.abs().argmax().item()
            top1_diff = diff.abs().argmax().item()
            print(f"  L{tli:02d}  ({dt:5.1f}s)  top|pure|=#{top1_pure}({pure[top1_pure]:+.3f})  "
                  f"top|diff|=#{top1_diff}({diff[top1_diff]:+.3f})")
        except torch.cuda.OutOfMemoryError as e:
            print(f"  L{tli:02d}  OOM, skipping")
            torch.cuda.empty_cache()
            continue

    # ---------- rank ----------
    def collect_top(attr_dict, top=TOP_K):
        rows = []
        for li, vec in attr_dict.items():
            for ni in range(vec.shape[0]):
                rows.append((abs(vec[ni].item()), li, ni, vec[ni].item()))
        rows.sort(reverse=True)
        return rows[:top]

    print(f"\n=== TOP {TOP_K} pure attribution (sum_t grad * act_B) ===")
    print(f"  {'rk':>3}  {'L#nrn':>10}  {'attr':>9}")
    for k, (_, li, ni, v) in enumerate(collect_top(attribution_pure)):
        print(f"   {k+1:>2}  L{li:02d}#{ni:<5d}  {v:+.4f}")

    print(f"\n=== TOP {TOP_K} diff attribution (sum_t grad * (act_B - a_mean)) ===")
    print(f"  {'rk':>3}  {'L#nrn':>10}  {'attr':>9}")
    top_diff = collect_top(attribution_diff)
    for k, (_, li, ni, v) in enumerate(top_diff):
        print(f"   {k+1:>2}  L{li:02d}#{ni:<5d}  {v:+.4f}")

    # cross-reference with NEURON_INVENTORY
    nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
    inventory = {(n["layer"], n["index"]): n for n in nj["known_neurons"]}
    print(f"\n=== TOP {TOP_K} diff attribution -- INVENTORY MATCHES ===")
    for k, (_, li, ni, v) in enumerate(top_diff):
        if (li, ni) in inventory:
            n = inventory[(li, ni)]
            print(f"   #{k+1:>2}  L{li:02d}#{ni:<5d}  {v:+.4f}  [{n['tier']:>15}, gain={n['default_gain']:+.1f}]  {n.get('label','')}")

    # save
    out = {
        "target_layer": TARGET_LI,
        "target_neuron": TARGET_NRN,
        "anchor_abs": anchor_abs,
        "attribution_pure": attribution_pure,
        "attribution_diff": attribution_diff,
    }
    torch.save(out, OUT_DIR / "circuit_attr_GB01.pt")
    print(f"\n[save] {OUT_DIR / 'circuit_attr_GB01.pt'}")


if __name__ == "__main__":
    main()
