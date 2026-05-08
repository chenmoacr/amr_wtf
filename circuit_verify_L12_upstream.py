"""
Phase 2-A verify: causal ablation of L12#4638 upstream top-5 nodes.

Same machinery as circuit_verify_topK.py, but:
  metric  M' = sum_{B anchors} act[L12, #4638]
  candidates = top-5 from circuit_attr_L12 (L09 cluster + L11 + L07)

Goal:
  - Confirm sign-consistency for each upstream node
  - Test the FFI hypothesis on L09#6039: it was driver of L33#9054 (+16),
    here appears as suppressor of L12#4638 (-2.33). If invert here
    raises L12#4638, the FFI motif is empirically real.
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
ATTR_PT = ROOT / "outputs" / "circuit_attr" / "circuit_attr_GB01.pt"
OUT_DIR = ROOT / "outputs" / "circuit_attr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
TARGET_LI = 12
TARGET_NRN = 4638

# Candidates from circuit_attr_L12 diff attribution top-10:
#   (layer, neuron, attribution_diff, label)
CANDIDATES = [
    ( 9,  357,  +4.52, "L09 strongest driver"),
    ( 9,  420,  -4.28, "L09 strongest suppressor"),
    (11, 2089,  -2.34, "L11 main amplifier"),
    ( 7, 2800,  +2.09, "earliest layer node"),
    ( 9, 6039,  -2.33, "FFI candidate (driver of L33, suppressor of L12)"),
]


# ---------- text helpers (copied from circuit_attr_GB01.py) ----------
def strip_user_block(query):
    s = query
    if "[用户]" in s:
        s = s.split("[用户]", 1)[1]
    if "[助手]" in s:
        s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def strip_response_prefix(resp):
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
    out.append(soc); out.extend(th_intro); out.extend(cot_body); out.extend(nl)
    out.append(eoc); out.extend(resp_ids); out.append(eot)
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


# ---------- intervention + measurement ----------
def make_intervene_hook(neuron_idx, mode):
    """mode in {'ablate', 'amplify', 'invert'}"""
    def pre(module, inputs):
        x = inputs[0].clone()
        if mode == "ablate":
            x[..., neuron_idx] = 0
        elif mode == "amplify":
            x[..., neuron_idx] = x[..., neuron_idx] * 2
        elif mode == "invert":
            x[..., neuron_idx] = x[..., neuron_idx] * -1
        else:
            raise ValueError(mode)
        return (x,) + inputs[1:]
    return pre


def run_measure(model, layers, ids, anchor_abs, intervene=None):
    """intervene = None | (li, nrn, mode)
    Returns (M, raw_act_at_intervene_node_or_None, intervene_target_layer_act_at_target)
    """
    metric_holder = {"h": None}
    intervene_obs = {"h": None}

    def metric_hook(module, inputs):
        metric_holder["h"] = inputs[0].detach().clone()

    hooks = [layers[TARGET_LI].mlp.down_proj.register_forward_pre_hook(metric_hook)]

    if intervene is not None:
        ili, inrn, imode = intervene
        # observation hook BEFORE the intervention hook (forward_pre runs hooks in registration order),
        # but to be safe let's just record the raw via a separate tap that runs first.
        def obs_hook(module, inputs):
            intervene_obs["h"] = inputs[0].detach()[..., inrn].clone()
        h_obs = layers[ili].mlp.down_proj.register_forward_pre_hook(obs_hook)
        h_int = layers[ili].mlp.down_proj.register_forward_pre_hook(
            make_intervene_hook(inrn, imode))
        hooks.append(h_obs); hooks.append(h_int)

    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
            model(input_ids=ids.unsqueeze(0).to(DEVICE), use_cache=False)
        h_metric = metric_holder["h"]
        slices = []
        for lo, hi in anchor_abs:
            slices.append(h_metric[0, lo:hi, TARGET_NRN].float())
        M = float(torch.cat(slices).sum().item())
        return M, intervene_obs["h"]
    finally:
        for h in hooks:
            h.remove()


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
    print(f"  layers={len(layers)}, alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB")

    soc = tokenizer.convert_tokens_to_ids("<|channel>")
    eoc = tokenizer.convert_tokens_to_ids("<channel|>")
    eot = tokenizer.convert_tokens_to_ids("<turn|>")

    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    de = json.loads(DE_PATH.read_text(encoding="utf-8"))
    item_b = data["B"][0]
    b_anchors = de["Blist"]

    full_b, P_b = build_sequence(tokenizer, item_b, soc, eoc, eot)
    print(f"[seq-B] T={full_b.shape[0]} (prompt={P_b})")

    anchor_abs = []
    for i, anc in enumerate(b_anchors):
        loc = find_anchor_token_window(tokenizer, full_b, P_b, anc)
        if loc is None:
            print(f"  B[{i+1}]  MISS"); continue
        lo, hi = loc
        anchor_abs.append((P_b + lo, P_b + hi))
        head = anc.replace("\n", "\\n")[:40]
        print(f"  B[{i+1}]  abs[{P_b+lo:>4d},{P_b+hi:>4d})  {head!r}")

    # ---- baseline ----
    print(f"\n[baseline] target M = sum_anchors act[L{TARGET_LI},#{TARGET_NRN}]")
    t0 = time.time()
    M_base, _ = run_measure(model, layers, full_b, anchor_abs, intervene=None)
    print(f"  M_baseline = {M_base:+.3f}   ({time.time()-t0:.1f}s)")

    # ---- per-candidate sweep ----
    rows = []
    for (li, nrn, attr_score, lbl) in CANDIDATES:
        print(f"\n--- candidate L{li:02d}#{nrn}  attr={attr_score:+.1f}  ({lbl}) ---")
        # Per-mode runs
        cand_results = {}
        for mode in ("ablate", "amplify", "invert"):
            t0 = time.time()
            M_int, raw = run_measure(model, layers, full_b, anchor_abs,
                                     intervene=(li, nrn, mode))
            delta = M_int - M_base
            rel = delta / abs(M_base) if abs(M_base) > 1e-6 else 0.0
            mean_raw = float(raw.float().mean().item())
            std_raw = float(raw.float().std().item())
            print(f"  [{mode:>7}]  M = {M_int:+.3f}   delta = {delta:+.3f}  "
                  f"({rel*100:+5.1f}%)   raw mean/std = {mean_raw:+.3f}/{std_raw:.3f}  "
                  f"({time.time()-t0:.1f}s)")
            cand_results[mode] = {"M": M_int, "delta": delta, "rel": rel,
                                   "raw_mean": mean_raw, "raw_std": std_raw}
        rows.append({
            "layer": li, "neuron": nrn, "attr_diff": attr_score, "label": lbl,
            **{f"M_{m}": cand_results[m]["M"] for m in cand_results},
            **{f"rel_{m}": cand_results[m]["rel"] for m in cand_results},
            "raw_mean": cand_results["ablate"]["raw_mean"],
            "raw_std":  cand_results["ablate"]["raw_std"],
        })

    # ---- summary table ----
    print("\n" + "=" * 78)
    print(f"  SUMMARY  (M_baseline = {M_base:+.3f})")
    print("=" * 78)
    print(f"  {'L#nrn':>10}  {'attr':>7}    {'rel_ablate':>10}  {'rel_amplify':>11}  {'rel_invert':>10}  {'raw_mean':>9}")
    for r in rows:
        print(f"  L{r['layer']:02d}#{r['neuron']:<5d}  {r['attr_diff']:+7.1f}    "
              f"{r['rel_ablate']*100:+9.1f}%  {r['rel_amplify']*100:+10.1f}%  "
              f"{r['rel_invert']*100:+9.1f}%  {r['raw_mean']:+9.3f}")
    print("=" * 78)
    print("Interpretation:")
    print("  rel_ablate >  +5%  → silencing this neuron INCREASES M  (suppressor)")
    print("  rel_ablate <  -5%  → silencing this neuron DECREASES M  (driver)")
    print("  rel_invert large   → sign of the neuron matters causally")
    print("  rel_amplify in line with sign(attr)  → attribution prediction holds")

    out = {"M_baseline": M_base, "anchors": anchor_abs, "candidates": rows,
           "target_layer": TARGET_LI, "target_neuron": TARGET_NRN}
    torch.save(out, OUT_DIR / "circuit_verify_L12_upstream.pt")
    print(f"\n[save] {OUT_DIR / 'circuit_verify_L12_upstream.pt'}")


if __name__ == "__main__":
    main()
