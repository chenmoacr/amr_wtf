"""
Build the OpusSAE Gemma model: real bias-merge (not just .pt offsets).

Math:
    Clamp: pre-down_proj activation x  ->  x with x[..., n] += off_n
    Effect on layer output:  W @ (x + delta) = W @ x + sum_n off_n * W[:, n]
    So merging is equivalent to adding bias_vec = sum_n off_n * W[:, n]
    to down_proj output. We materialize this by replacing down_proj
    with a bias=True nn.Linear and saving model.

Sign convention:  off = global_alpha * gain * tier_scale
    gain sign preserved; tier_scale always positive; alpha=1.

Outputs to gemma7_E2b_OpusSAE/:
    - safetensors (full merged model)
    - down_proj_biases.pt (per-layer bias vectors, fallback)
    - neuron_offsets.pt (raw offsets, fallback for hook-based usage)
    - merge_metadata.json (provenance)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
ORIGINAL_PATH = Path("J:/amr/models/gemma-4-E2B-it")
NEW_PATH = ROOT / "gemma7_E2b_OpusSAE"
NEURONS_JSON = ROOT / "chat" / "neurons.json"
DTYPE = torch.bfloat16

# Tier merge scales
TIER_SCALE = {
    "verified": 1.0,
    "thought_quality": 1.0,
    "general": 1.0,
    "lit": 0.5,
    "math": 0.5,
    "candidate": 0.0,  # SKIP (n=1 unverified)
}
GLOBAL_ALPHA = 1.0


def load_neurons():
    data = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
    return data["known_neurons"]


def compute_offsets(neurons):
    """Layer -> {neuron_idx: total_offset}; preserves negative gain signs."""
    offsets = {}
    by_tier_count = {}
    skipped = 0
    for n in neurons:
        tier = n["tier"]
        scale = TIER_SCALE.get(tier, 0.0)
        if scale == 0.0:
            skipped += 1
            continue
        gain = float(n["default_gain"])
        off = GLOBAL_ALPHA * gain * scale  # sign preserved (scale > 0)
        li, ni = int(n["layer"]), int(n["index"])
        offsets.setdefault(li, {})
        offsets[li][ni] = offsets[li].get(ni, 0.0) + off
        by_tier_count[tier] = by_tier_count.get(tier, 0) + 1
    return offsets, by_tier_count, skipped


def main():
    print(f"[init] source={ORIGINAL_PATH}  dest={NEW_PATH}")
    if NEW_PATH.exists():
        print(f"[init] removing existing {NEW_PATH}")
        shutil.rmtree(NEW_PATH)
    NEW_PATH.mkdir(parents=True)

    print("[1/5] copying tokenizer / config / etc...")
    for f in ORIGINAL_PATH.iterdir():
        if f.is_file() and not f.name.endswith((".safetensors", ".bin")):
            shutil.copy2(f, NEW_PATH / f.name)
    print(f"  copied {sum(1 for _ in NEW_PATH.iterdir())} small files")

    print("[2/5] loading original model on CPU (BF16)...")
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        ORIGINAL_PATH, torch_dtype=DTYPE, device_map="cpu", low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    layers = model.model.language_model.layers
    print(f"  {len(layers)} transformer layers")

    print("[3/5] computing per-layer down_proj bias vectors...")
    neurons = load_neurons()
    offsets, by_tier_count, skipped = compute_offsets(neurons)
    print(f"  merged tiers: {by_tier_count}  | skipped: {skipped}")
    print(f"  affected layers: {len(offsets)}")

    biases_for_save = {}
    for li in sorted(offsets.keys()):
        neuron_offsets = offsets[li]
        W = layers[li].mlp.down_proj.weight  # [hidden, intermediate]
        out_dim = W.shape[0]
        bias = torch.zeros(out_dim, dtype=torch.float32)
        for ni, off in neuron_offsets.items():
            bias += off * W[:, ni].to(torch.float32)
        biases_for_save[li] = bias.to(DTYPE)
        print(f"  L{li:02d}: {len(neuron_offsets):>3d} neurons | |bias|={bias.norm().item():.3f}")

    print("[4/5] replacing down_proj with bias=True versions...")
    for li, bias_vec in biases_for_save.items():
        old = layers[li].mlp.down_proj
        new = nn.Linear(old.in_features, old.out_features, bias=True,
                        dtype=old.weight.dtype, device=old.weight.device)
        new.weight.data.copy_(old.weight.data)
        new.bias.data.copy_(bias_vec.to(old.weight.dtype))
        layers[li].mlp.down_proj = new
    print(f"  replaced {len(biases_for_save)} down_proj layers")

    print("[5/5] saving merged model...")
    # config tweak so future loads know about bias on down_proj
    if hasattr(model.config, "text_config"):
        model.config.text_config.mlp_bias = True
    model.config.mlp_bias = True
    model.save_pretrained(NEW_PATH, safe_serialization=True)
    tokenizer.save_pretrained(NEW_PATH)

    # Always also save offsets / biases for fallback hook-based usage
    torch.save(biases_for_save, NEW_PATH / "down_proj_biases.pt")
    torch.save(offsets, NEW_PATH / "neuron_offsets.pt")

    metadata = {
        "source_model": str(ORIGINAL_PATH),
        "tier_scales": TIER_SCALE,
        "global_alpha": GLOBAL_ALPHA,
        "merged_neurons_by_tier": by_tier_count,
        "total_merged_neurons": sum(by_tier_count.values()),
        "skipped_neurons": skipped,
        "affected_layers": sorted(offsets.keys()),
        "merge_kind": "down_proj_bias_addition",
        "math_equivalence": "bias_vec[i] = sum_n offset_n * down_proj.weight[i, n]",
    }
    (NEW_PATH / "merge_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[done] OpusSAE model saved to {NEW_PATH}")
    print(f"  - safetensors  (full merged model with bias=True down_proj)")
    print(f"  - down_proj_biases.pt  (fallback)")
    print(f"  - neuron_offsets.pt    (raw offsets, fallback for hooks)")
    print(f"  - merge_metadata.json  (provenance)")


if __name__ == "__main__":
    main()
