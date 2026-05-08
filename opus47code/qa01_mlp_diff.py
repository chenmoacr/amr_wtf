"""
QA01 MLP differential — testing the "L24 is the working layer" hypothesis.

Hypothesis (user):
  L24 is Gemma 4 E2B's core "GPT working layer" — the place where actual
  output decisions form. The surrounding layers (L0-L23 and L25-L34) are
  wrappers turning that core into a chat-style assistant.

Test:
  Run two teacher-forced forward passes with the SAME user prompt:
    Seq A: assistant = Gemma's own baseline answer (extracted from earlier run)
    Seq B: assistant = Opus's reference answer
  Capture mlp.down_proj input at multiple candidate layers (20..32).

  For each layer L, compute:
    A_mean[n] = mean over assistant tokens of intermediate[L, t, n]  (Seq A)
    B_mean[n] = mean over assistant tokens of intermediate[L, t, n]  (Seq B)
    diff[n]   = B_mean[n] - A_mean[n]

  If L24 is the working layer, max|diff| at L24 should be markedly larger
  than at L22, L26, etc. — i.e. L24 differentiates Gemma-style vs
  Opus-style output more sharply than its neighbors.

  Top-K neurons by |diff| at L24 = candidate "Opus-quality carrier"
  neurons that previous NEURON_INVENTORY may have missed.

Output:
  outputs/opus47_mlp_diff/
    top_neurons.txt      ranked top-K per layer + cross-layer separation
    raw.pt               full means + diff vectors
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
BASELINE_TXT = ROOT / "outputs" / "qa01_compare" / "baseline.txt"
NEURONS_JSON = ROOT / "chat" / "neurons.json"
OUT_DIR = ROOT / "outputs" / "opus47_mlp_diff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
LAYERS_TO_TEST = [20, 22, 24, 26, 28, 30, 32]   # focus on L24, sweep neighbors
TOP_K = 50


def extract_answer_from_baseline_txt(path: Path) -> str:
    txt = path.read_text(encoding="utf-8")
    s = txt.find("ANSWER (response)")
    if s < 0:
        raise RuntimeError(f"no ANSWER section in {path}")
    s = txt.find("\n", s) + 1
    e = txt.find("================ META")
    ans = txt[s:e].strip()
    # strip leading separator equals signs
    while ans.startswith("="):
        ans = ans.lstrip("=").lstrip("\n").strip()
    return ans


def main():
    print(f"[load] {MODEL_PATH}", flush=True)
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
    layers = model.model.language_model.layers
    print(f"  layers={len(layers)}  alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)

    # ---- load data ----
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = qa["input"]
    opus_answer = qa["output"]
    baseline_answer = extract_answer_from_baseline_txt(BASELINE_TXT)
    print(f"[data] user={len(user_text)}c  opus_answer={len(opus_answer)}c  "
          f"baseline_answer={len(baseline_answer)}c", flush=True)
    print(f"[data] baseline preview: {baseline_answer[:120]!r}", flush=True)

    # ---- determine assistant_start ----
    msgs_user_only = [{"role": "user", "content": user_text}]
    user_enc = tokenizer.apply_chat_template(
        msgs_user_only, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    assistant_start = user_enc["input_ids"].shape[1]
    print(f"[align] assistant_start = {assistant_start}", flush=True)

    # ---- run two forwards, capture mlp.down_proj input at chosen layers ----
    layer_means = {L: {} for L in LAYERS_TO_TEST}

    for label, ans in [("opus", opus_answer), ("baseline", baseline_answer)]:
        msgs = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": ans},
        ]
        enc = tokenizer.apply_chat_template(
            msgs, tokenize=True, return_tensors="pt", return_dict=True,
            enable_thinking=False,
        )
        ids = enc["input_ids"]
        T = ids.shape[1]
        n_assistant = T - assistant_start
        print(f"\n[{label}] seq T={T}  assistant_tokens={n_assistant}", flush=True)

        # Hooks: capture inputs[0] of each chosen down_proj
        captured = {}
        def make_pre(name):
            def pre(module, inputs):
                captured[name] = inputs[0].detach()  # [1, T, intermediate]
            return pre
        handles = []
        for L in LAYERS_TO_TEST:
            h = layers[L].mlp.down_proj.register_forward_pre_hook(make_pre(f"L{L}"))
            handles.append(h)

        try:
            t0 = time.time()
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                model(input_ids=ids.to(DEVICE), use_cache=False)
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated(0) / 1e9
            print(f"  forward done in {dt:.1f}s  peak={peak:.2f}GB", flush=True)
            torch.cuda.reset_peak_memory_stats(0)
        finally:
            for h in handles:
                h.remove()

        # mean over assistant tokens (skip user region)
        for L in LAYERS_TO_TEST:
            cap = captured[f"L{L}"]
            mean = cap[0, assistant_start:].float().mean(dim=0).cpu()
            layer_means[L][label] = mean
        del captured
        torch.cuda.empty_cache()

    # ---- compute diff and rank ----
    print("\n========================================================================")
    print(f"  Layer separation strength (max|diff| / mean|diff| / count(|diff|>thr))")
    print("========================================================================")
    sep_table = []
    for L in LAYERS_TO_TEST:
        diff = layer_means[L]["opus"] - layer_means[L]["baseline"]
        max_abs = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()
        std_diff = diff.std().item()
        n_05 = (diff.abs() > 0.5).sum().item()
        n_10 = (diff.abs() > 1.0).sum().item()
        n_20 = (diff.abs() > 2.0).sum().item()
        sep_table.append((L, max_abs, mean_abs, std_diff, n_05, n_10, n_20))
        print(f"  L{L:>2}: max|diff|={max_abs:>6.3f}  mean|diff|={mean_abs:.4f}  "
              f"std={std_diff:.3f}  n>0.5={n_05:>3d}  n>1.0={n_10:>3d}  n>2.0={n_20:>3d}",
              flush=True)

    # which layer has the strongest separation?
    best_L = max(sep_table, key=lambda r: r[1])[0]
    print(f"\n  → strongest max|diff| at L{best_L}", flush=True)

    # top-K per layer
    all_results = {}
    print("\n========================================================================")
    print(f"  Per-layer top-15 neurons by |diff|")
    print("========================================================================")
    for L in LAYERS_TO_TEST:
        diff = layer_means[L]["opus"] - layer_means[L]["baseline"]
        topk = diff.abs().topk(TOP_K)
        results = []
        for idx, val_abs in zip(topk.indices.tolist(), topk.values.tolist()):
            results.append({
                "neuron": idx,
                "diff": diff[idx].item(),
                "opus_mean": layer_means[L]["opus"][idx].item(),
                "base_mean": layer_means[L]["baseline"][idx].item(),
            })
        all_results[L] = results
        print(f"\n  --- L{L} top-15 ---", flush=True)
        for i, r in enumerate(results[:15]):
            arrow = "Opus↑" if r["diff"] > 0 else "Base↑"
            print(f"    {i+1:>2}. L{L}#{r['neuron']:<5d}  "
                  f"diff={r['diff']:+7.3f}  ({arrow})  "
                  f"opus={r['opus_mean']:+6.3f}  base={r['base_mean']:+6.3f}",
                  flush=True)

    # ---- cross-reference NEURON_INVENTORY ----
    inventory = {}
    if NEURONS_JSON.exists():
        nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
        inventory = {(n["layer"], n["index"]): n for n in nj["known_neurons"]}
    print("\n========================================================================")
    print(f"  Inventory matches in per-layer top-{TOP_K}")
    print("========================================================================")
    for L in LAYERS_TO_TEST:
        matches = []
        for r in all_results[L]:
            key = (L, r["neuron"])
            if key in inventory:
                matches.append((r, inventory[key]))
        print(f"\n  L{L:>2}: {len(matches)} matches", flush=True)
        for r, inv in matches:
            print(f"    L{L}#{r['neuron']:<5d}  diff={r['diff']:+7.3f}  "
                  f"[{inv.get('tier','?'):>15}, gain={inv.get('default_gain',0):+.1f}]  "
                  f"{inv.get('label','')}", flush=True)

    # ---- save ----
    torch.save({
        "layer_means": layer_means,
        "all_results": all_results,
        "sep_table": sep_table,
        "best_L": best_L,
    }, OUT_DIR / "raw.pt")

    with open(OUT_DIR / "top_neurons.txt", "w", encoding="utf-8") as f:
        f.write("================ QA01 MLP Differential ================\n")
        f.write(f"Hypothesis: L24 is Gemma 4 E2B's core 'working layer'.\n")
        f.write(f"Method:\n")
        f.write(f"  Two teacher-forced sequences with same user prompt:\n")
        f.write(f"    Seq A = Gemma baseline answer ({len(baseline_answer)} chars)\n")
        f.write(f"    Seq B = Opus reference answer ({len(opus_answer)} chars)\n")
        f.write(f"  Capture mlp.down_proj input at L = {LAYERS_TO_TEST}.\n")
        f.write(f"  Mean over assistant region tokens, then diff = B - A.\n\n")
        f.write(f"================ Layer separation ================\n")
        f.write(f"  L  | max|diff|  mean|diff|  std  >0.5  >1.0  >2.0\n")
        for L, max_abs, mean_abs, std_d, n_05, n_10, n_20 in sep_table:
            f.write(f"  L{L:>2}  | {max_abs:>9.3f}  {mean_abs:>9.4f}  "
                    f"{std_d:.3f}  {n_05:>4d}  {n_10:>4d}  {n_20:>4d}\n")
        f.write(f"\n  → strongest max|diff| at L{best_L}\n")

        for L in LAYERS_TO_TEST:
            f.write(f"\n\n================ L{L} top-{TOP_K} neurons ================\n")
            f.write(f"  rank  L#nrn          diff       opus_mean   base_mean   sign\n")
            for i, r in enumerate(all_results[L]):
                sign = "Opus↑" if r["diff"] > 0 else "Base↑"
                f.write(f"  {i+1:>4}  L{L}#{r['neuron']:<5d}    "
                        f"{r['diff']:+8.3f}    "
                        f"{r['opus_mean']:+8.3f}    "
                        f"{r['base_mean']:+8.3f}   {sign}\n")

        f.write(f"\n\n================ Inventory matches (per layer in top-{TOP_K}) ================\n")
        if not inventory:
            f.write("  (no neurons.json found)\n")
        else:
            for L in LAYERS_TO_TEST:
                matches = [(r, inventory[(L, r["neuron"])]) for r in all_results[L]
                           if (L, r["neuron"]) in inventory]
                f.write(f"\n  L{L:>2}: {len(matches)} of top-{TOP_K} match inventory\n")
                for r, inv in matches:
                    f.write(f"    L{L}#{r['neuron']:<5d}  diff={r['diff']:+7.3f}  "
                            f"[{inv.get('tier','?')}, gain={inv.get('default_gain',0):+.1f}]  "
                            f"{inv.get('label','')}\n")

    print(f"\n[save] {OUT_DIR / 'raw.pt'}", flush=True)
    print(f"[save] {OUT_DIR / 'top_neurons.txt'}", flush=True)


if __name__ == "__main__":
    main()
