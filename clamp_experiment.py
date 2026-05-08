"""
Clamp / steering experiment: shift candidate L15+ persistent neurons toward
Opus's mean activations during Gemma's own generation, and observe the output
change.

Each candidate neuron n in layer li is shifted by:
    x[:, :, n] += alpha * diff[li][n]   (where diff = mean_opus - mean_gemma)

We run several configurations:
    - baseline already exists: outputs/qa01/gemma_answer.txt
    - top-5 L15+ persistent  @ alpha=1.0
    - top-25 L15+ persistent @ alpha=1.0
    - top-25 L15+ persistent @ alpha=2.0
    - L27#10024 alone        @ alpha=2.0   (the strongest single candidate)
"""
from __future__ import annotations

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
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "claudeopusQA01.json"
DIFF_PATH = ROOT / "outputs" / "qa01" / "diff.pt"
OUT_DIR = ROOT / "outputs" / "qa01" / "clamp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
MAX_NEW_TOKENS = 600


def pick_L15_persistent(diff, stats_a, stats_b, n_layers, k,
                        diff_thresh=0.2, rate_thresh=0.3):
    cands = []  # (priority, li, n, diff_val)
    for li in range(15, n_layers):
        d = diff[li]
        ra = stats_a["active_rate"][li]
        rb = stats_b["active_rate"][li]
        rmax = torch.maximum(ra, rb)
        for n in range(d.shape[0]):
            dv = d[n].item()
            r = rmax[n].item()
            if abs(dv) >= diff_thresh and r >= rate_thresh:
                cands.append((abs(dv) * r, li, n, dv))
    cands.sort(reverse=True)
    return cands[:k]


def install_clamp_hooks(layers, candidates, alpha):
    """candidates: list of (priority, li, n, diff_val).
    Returns list of hook handles. shift = alpha * diff_val."""
    by_layer: dict[int, list[tuple[int, float]]] = {}
    for _, li, n, dv in candidates:
        by_layer.setdefault(li, []).append((n, alpha * dv))
    hooks = []
    for li, items in by_layer.items():
        idx = torch.tensor([n for n, _ in items], dtype=torch.long)
        shift = torch.tensor([s for _, s in items], dtype=torch.float32)
        def make_fn(idx, shift):
            def pre(module, inputs):
                x = inputs[0].clone()
                i = idx.to(x.device)
                s = shift.to(x.dtype).to(x.device)
                x[:, :, i] = x[:, :, i] + s
                return (x,) + inputs[1:]
            return pre
        h = layers[li].mlp.down_proj.register_forward_pre_hook(make_fn(idx, shift))
        hooks.append(h)
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def generate(model, tokenizer, prompt_ids, label):
    print(f"[gen-{label}] starting (prompt_len={prompt_ids.shape[1]})...")
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
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    print(f"[gen-{label}] {gen_ids.shape[0]} tokens in {time.time()-t0:.1f}s")
    return text


def main():
    # --- load data + model ---
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
    n_layers = len(layers)
    print(f"[load] alloc={torch.cuda.memory_allocated()/1e9:.2f}GB, layers={n_layers}")

    # --- load diff ---
    data = torch.load(DIFF_PATH, weights_only=False)
    diff = data["diff"]
    stats_a = data["stats_a"]
    stats_b = data["stats_b"]
    print(f"[diff] loaded; n_layers={data['n_layers']}")

    # --- candidate selection ---
    top25 = pick_L15_persistent(diff, stats_a, stats_b, n_layers, k=25)
    top5 = top25[:5]
    only_27_10024 = [c for c in top25 if c[1] == 27 and c[2] == 10024]
    if not only_27_10024:
        only_27_10024 = [(0, 27, 10024, diff[27][10024].item())]

    print("[cands] top-5:")
    for _, li, n, dv in top5:
        print(f"  L{li:02d}#{n:<5d}  diff={dv:+.3f}")

    # --- prepare prompt ---
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    msgs = [{"role": "user", "content": qa["input"]}]
    out = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True)
    prompt_ids = out["input_ids"]
    print(f"[prompt] tokens={prompt_ids.shape[1]}")

    # --- run configs ---
    configs = [
        ("baseline_alpha0",        [],                    0.0),
        ("top5_a1.0",              top5,                  1.0),
        ("top25_a1.0",             top25,                 1.0),
        ("top25_a2.0",             top25,                 2.0),
        ("L27_10024_only_a2.0",    only_27_10024,         2.0),
        ("L27_10024_only_a4.0",    only_27_10024,         4.0),
    ]

    results = {}
    for label, cands, alpha in configs:
        hooks = install_clamp_hooks(layers, cands, alpha) if cands and alpha != 0 else []
        info = f"{label}: clamping {len(cands)} neurons at alpha={alpha}"
        print(f"\n=== {info} ===")
        text = generate(model, tokenizer, prompt_ids, label)
        if hooks:
            remove_hooks(hooks)
        out_path = OUT_DIR / f"{label}.txt"
        out_path.write_text(text, encoding="utf-8")
        # save a brief summary line
        results[label] = {
            "alpha": alpha,
            "n_clamped": len(cands),
            "neurons": [(li, n, dv) for _, li, n, dv in cands[:10]],
            "preview": text[:300],
        }
        print(f"  preview: {text[:200]!r}")

    # save summary
    (OUT_DIR / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[save] summary -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
