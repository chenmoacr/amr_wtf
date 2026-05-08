"""
Phase 2-A. Attribution patching upstream of L12#4638.

Same machinery as circuit_attr_GB01.py, but:
  metric M' = sum over B-anchors of act[L12, #4638]
  target_li scan range = [0, 12]   (only layers that can causally feed L12)

Goal: find the early-layer (L00..L11) drivers of L12#4638, the strongest
mid-early node confirmed in circuit_verify_topK on 2026-05-01.
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

# Reuse all the helpers
from circuit_attr_GB01 import (
    build_sequence,
    find_anchor_token_window,
    attribute_one_layer,
    capture_act_only,
)

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
DE_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01_de.json"
NEURONS_JSON = ROOT / "chat" / "neurons.json"
OUT_DIR = ROOT / "outputs" / "circuit_attr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
TARGET_LI = 12      # metric layer
TARGET_NRN = 4638   # metric neuron
SCAN_TO = 12        # attribute target_li in [0, SCAN_TO]
TOP_K = 30


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

    print("\n[A-baseline] capturing per-layer mean activation on A...")
    t0 = time.time()
    a_means = capture_act_only(model, layers, full_a)
    print(f"  done in {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()

    attribution_pure = {}
    attribution_diff = {}

    print(f"\n[attribution] target metric: M' = sum over B-anchors of act[L{TARGET_LI}, #{TARGET_NRN}]")
    print(f"  scanning target_li=0..{SCAN_TO} (upstream of L{TARGET_LI})")

    for tli in range(SCAN_TO + 1):
        torch.cuda.empty_cache()
        gc.collect()
        try:
            t0 = time.time()
            h_act, h_grad = attribute_one_layer(
                model, layers, tli, full_b, anchor_abs, TARGET_LI, TARGET_NRN,
            )
            act = h_act[0].float()
            grad = h_grad[0].float()
            pure = (grad * act).sum(dim=0)
            diff = (grad * (act - a_means[tli].unsqueeze(0).float())).sum(dim=0)
            attribution_pure[tli] = pure
            attribution_diff[tli] = diff
            dt = time.time() - t0
            top1_pure = pure.abs().argmax().item()
            top1_diff = diff.abs().argmax().item()
            print(f"  L{tli:02d}  ({dt:5.1f}s)  top|pure|=#{top1_pure}({pure[top1_pure]:+.3f})  "
                  f"top|diff|=#{top1_diff}({diff[top1_diff]:+.3f})")
        except torch.cuda.OutOfMemoryError:
            print(f"  L{tli:02d}  OOM, skipping")
            torch.cuda.empty_cache()
            continue

    def collect_top(attr_dict, top=TOP_K, exclude_metric=True):
        rows = []
        for li, vec in attr_dict.items():
            for ni in range(vec.shape[0]):
                if exclude_metric and li == TARGET_LI and ni == TARGET_NRN:
                    continue
                rows.append((abs(vec[ni].item()), li, ni, vec[ni].item()))
        rows.sort(reverse=True)
        return rows[:top]

    print(f"\n=== TOP {TOP_K} pure attribution (sum_t grad * act_B), excluding metric self ===")
    print(f"  {'rk':>3}  {'L#nrn':>10}  {'attr':>9}")
    for k, (_, li, ni, v) in enumerate(collect_top(attribution_pure)):
        print(f"   {k+1:>2}  L{li:02d}#{ni:<5d}  {v:+.4f}")

    print(f"\n=== TOP {TOP_K} diff attribution (sum_t grad * (act_B - a_mean)), excluding metric self ===")
    print(f"  {'rk':>3}  {'L#nrn':>10}  {'attr':>9}")
    top_diff = collect_top(attribution_diff)
    for k, (_, li, ni, v) in enumerate(top_diff):
        print(f"   {k+1:>2}  L{li:02d}#{ni:<5d}  {v:+.4f}")

    # cross-reference with NEURON_INVENTORY
    if NEURONS_JSON.exists():
        nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
        inventory = {(n["layer"], n["index"]): n for n in nj["known_neurons"]}
        matches = [(k, li, ni, v) for k, (_, li, ni, v) in enumerate(top_diff) if (li, ni) in inventory]
        print(f"\n=== TOP {TOP_K} diff attribution -- INVENTORY MATCHES ({len(matches)}) ===")
        for k, li, ni, v in matches:
            n = inventory[(li, ni)]
            print(f"   #{k+1:>2}  L{li:02d}#{ni:<5d}  {v:+.4f}  [{n['tier']:>15}, gain={n['default_gain']:+.1f}]  {n.get('label','')}")
    else:
        print(f"[warn] {NEURONS_JSON} missing, skipping inventory cross-ref")

    # ---- per-layer top-3 (for chain visualization) ----
    print(f"\n=== Per-layer top-3 |diff| (excluding metric self) ===")
    print(f"  {'L':>3}    rank1 (n, attr)         rank2                    rank3")
    for li in sorted(attribution_diff.keys()):
        vec = attribution_diff[li].clone()
        if li == TARGET_LI:
            vec[TARGET_NRN] = 0.0
        absv = vec.abs()
        topi = absv.topk(3).indices.tolist()
        cells = []
        for ni in topi:
            cells.append(f"#{ni:<5d} {vec[ni].item():+7.3f}")
        print(f"  L{li:02d}    " + "    ".join(cells))

    out = {
        "target_layer": TARGET_LI,
        "target_neuron": TARGET_NRN,
        "scan_to": SCAN_TO,
        "anchor_abs": anchor_abs,
        "attribution_pure": attribution_pure,
        "attribution_diff": attribution_diff,
    }
    torch.save(out, OUT_DIR / "circuit_attr_L12.pt")
    print(f"\n[save] {OUT_DIR / 'circuit_attr_L12.pt'}")


if __name__ == "__main__":
    main()
