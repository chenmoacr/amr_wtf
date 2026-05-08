"""
QA01 — proper region-aware hook injection (no bias-merge, no chain).

Uses the existing chat/runtime.py implementation (ClampChatRuntime) which
correctly gates clamp hooks by token region:
  - thought-region neurons: active before <channel|>, off after
  - answer-region neurons: off before <channel|>, active after
  - always-region neurons: active everywhere

This is the only path that respects the region: field in neurons.json.
The bias-merge OpusSAE model loses this distinction (always-on for all 67
neurons), which we now suspect is why the allin run looked like vanilla
Gemma boilerplate.

Output: outputs/qa01_compare/hook_proper.txt
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import from chat/ (where ClampChatRuntime lives)
sys.path.insert(0, str(Path("J:/amr/amr_wtf/chat")))

import torch
from runtime import ClampChatRuntime  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
NEURONS_JSON = ROOT / "chat" / "neurons.json"
QA01_PATH = ROOT / "data" / "claudeopusQA01.json"
OUT_DIR = ROOT / "outputs" / "qa01_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "hook_proper.txt"

# Same tier scaling as build_opus_sae_model.py — keeps gain conventions
TIER_SCALE = {
    "verified": 1.0,
    "thought_quality": 1.0,
    "general": 1.0,
    "lit": 0.5,
    "math": 0.5,
    "candidate": 0.0,  # SKIP unverified
}
GLOBAL_ALPHA = 1.0
MAX_NEW_TOKENS = 1536


def split_thought_response(text: str):
    eoc = "<channel|>"
    if eoc in text:
        thought, response = text.split(eoc, 1)
        for prefix in ("<|channel>thought\n", "<|channel>thought"):
            if thought.startswith(prefix):
                thought = thought[len(prefix):]
                break
        return thought.rstrip("\n"), response.rstrip("<turn|>").rstrip()
    return text, ""


def main():
    print("[load] ClampChatRuntime + neurons.json", flush=True)
    runtime = ClampChatRuntime(str(NEURONS_JSON))
    runtime.load()
    print(f"  alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)

    # Build steering selection: 57 neurons (skip candidate tier)
    selection = []
    by_tier = {}
    by_region = {}
    for n in runtime.known_neurons:
        scale = TIER_SCALE.get(n["tier"], 0.0)
        if scale == 0.0:
            continue
        eff_gain = float(n["default_gain"]) * scale
        selection.append({
            "id": n["id"],
            "layer": n["layer"],
            "index": n["index"],
            "gain": eff_gain,
            "enabled": True,
        })
        by_tier[n["tier"]] = by_tier.get(n["tier"], 0) + 1
        by_region[n.get("region", "answer")] = by_region.get(n.get("region", "answer"), 0) + 1
    print(f"[selection] {len(selection)} neurons selected")
    print(f"  by tier: {by_tier}")
    print(f"  by region: {by_region}")

    snapshot = {
        "enabled": True,
        "think_mode": True,
        "global_alpha": GLOBAL_ALPHA,
        "temperature": 0.0,    # greedy
        "max_new_tokens": MAX_NEW_TOKENS,
        "neurons": selection,
    }

    # Load QA01 prompt
    qa01 = json.loads(QA01_PATH.read_text(encoding="utf-8"))
    prompt_text = qa01["input"]
    print(f"[input] QA01 = {len(prompt_text)} chars", flush=True)

    print("\n[generate] running region-aware hook injection...", flush=True)
    t0 = time.time()
    raw = runtime.generate_reply(
        history=[],
        user_text=prompt_text,
        steering_snapshot=snapshot,
    )
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"[done] {dt:.1f}s  peak={peak:.2f}GB  raw_chars={len(raw)}", flush=True)

    thought, response = split_thought_response(raw)
    OUT_FILE.write_text(
        "================ INPUT ================\n"
        + prompt_text
        + "\n\n================ COT (thought) ================\n"
        + thought
        + "\n\n================ ANSWER (response) ================\n"
        + response
        + "\n\n================ META ================\n"
        + "label         = hook_proper (region-aware, no chain)\n"
        + f"n_selected    = {len(selection)}\n"
        + f"by_tier       = {by_tier}\n"
        + f"by_region     = {by_region}\n"
        + f"elapsed_s     = {dt:.2f}\n"
        + f"peak_gpu_gb   = {peak:.2f}\n"
        + f"thought_chars = {len(thought)}\n"
        + f"answer_chars  = {len(response)}\n",
        encoding="utf-8",
    )
    print(f"[save] {OUT_FILE}", flush=True)
    print(f"[save] thought={len(thought)}c  answer={len(response)}c", flush=True)
    print(f"[save] answer head: {response[:200].replace(chr(10), ' ')!r}", flush=True)


if __name__ == "__main__":
    main()
