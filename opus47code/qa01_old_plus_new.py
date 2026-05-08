"""
QA01 generate with combined OLD + NEW neuron set.

Step 1 (record):
  Build NEURON_INVENTORY_v2:
    - Old: all 57 from chat/neurons.json (skip candidate tier with scale=0)
    - New: per-layer top-K of MLP QA-differential (|diff| > threshold,
           and not already in old inventory)
  Write outputs/qa01_compare/NEURON_INVENTORY_v2.txt for record-keeping.

Step 2 (generate):
  Use chat/runtime.py-style region-aware clamp hooks (the implementation
  that is known to actually fire, unlike the broken bias-merge model).
  Old neurons keep their region tag; new neurons default to region="answer".
  Gain rule for new ones:
    sign(diff) * 1.0  — push toward Opus side, suppress Gemma side
  All run under tier_scale=1.0 (we want to see full effect).

  max_new_tokens = 65535
  Stop conditions:
    - EOS (turn end)
    - LoopStopper: a 30-char chunk repeats >= 4 times in last 600 chars
                   (catches repetition collapse without false-positive
                   on bullet lists)

Output:
  outputs/qa01_compare/NEURON_INVENTORY_v2.txt
  outputs/qa01_compare/old_plus_new.txt   (input + thought + answer + meta)
  outputs/qa01_compare/old_plus_new.log
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, str(Path("J:/amr/amr_wtf/chat")))   # for steering.py

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    AutoTokenizer,
    Gemma4ForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)
from steering import (   # noqa: E402
    ClampGate,
    install_clamp_hooks,
    install_segment_gate,
    remove_hooks,
)

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "claudeopusQA01.json"
NEURONS_JSON = ROOT / "chat" / "neurons.json"
MLP_DIFF_PT = ROOT / "outputs" / "opus47_mlp_diff" / "raw.pt"
OUT_DIR = ROOT / "outputs" / "qa01_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INVENTORY_V2_TXT = OUT_DIR / "NEURON_INVENTORY_v2.txt"
OUT_TXT = OUT_DIR / "old_plus_new.txt"

DEVICE = "cuda:0"
MAX_NEW_TOKENS = 65535

# Tier scaling — keep same conventions as build_opus_sae_model.py
TIER_SCALE = {
    "verified": 1.0,
    "thought_quality": 1.0,
    "general": 1.0,
    "lit": 0.5,
    "math": 0.5,
    "candidate": 0.0,    # SKIP
    "mlp_diff_v2": 1.0,  # NEW tier for this experiment
}
GLOBAL_ALPHA = 1.0

# New-neuron selection thresholds
TOPK_PER_LAYER = 10           # take per-layer top-N from mlp_diff
MIN_ABS_DIFF = 0.15           # |diff| threshold


# -------- loop stopper --------
class LoopStopper(StoppingCriteria):
    """Stop if recent generated text shows obvious repetition collapse."""
    def __init__(self, tokenizer, prompt_len, check_every=120,
                 window_chars=600, chunk_size=30, repeat_count=4):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.check_every = check_every
        self.window_chars = window_chars
        self.chunk_size = chunk_size
        self.repeat_count = repeat_count
        self.last_check_n = 0
        self.tripped = False
        self.reason = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.tripped:
            return True
        n_total = input_ids.shape[1]
        n_gen = n_total - self.prompt_len
        if n_gen - self.last_check_n < self.check_every:
            return False
        self.last_check_n = n_gen
        # decode last ~window worth of tokens (over-decode to be safe)
        tail_ids = input_ids[0, max(self.prompt_len, n_total - 800):].tolist()
        text = self.tokenizer.decode(tail_ids, skip_special_tokens=False)
        if len(text) > self.window_chars:
            text = text[-self.window_chars:]
        # repetition check: any chunk of length chunk_size that recurs
        # repeat_count times => collapse
        seen = {}
        n_text = len(text)
        if n_text < self.chunk_size * 2:
            return False
        # use stride 1 but skip ahead once a repetition is found
        for i in range(0, n_text - self.chunk_size, 1):
            chunk = text[i:i + self.chunk_size]
            seen[chunk] = seen.get(chunk, 0) + 1
            if seen[chunk] >= self.repeat_count:
                self.tripped = True
                self.reason = (f"chunk repeated {seen[chunk]}× in last "
                               f"{n_text} chars: {chunk!r}")
                print(f"  [LoopStopper] TRIPPED: {self.reason}", flush=True)
                return True
        return False


# -------- inventory builder --------
def build_combined_inventory():
    """Return (combined_list, old_count, new_count, by_layer_summary)."""
    # Old neurons
    nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
    old_known = nj["known_neurons"]
    old_keys = set()
    combined = []
    skipped_candidate = 0
    for n in old_known:
        scale = TIER_SCALE.get(n["tier"], 0.0)
        if scale == 0.0:
            skipped_candidate += 1
            continue
        eff = GLOBAL_ALPHA * float(n["default_gain"]) * scale
        combined.append({
            "id": n["id"],
            "layer": int(n["layer"]),
            "index": int(n["index"]),
            "gain": float(n["default_gain"]),
            "effective_offset": eff,
            "tier": n["tier"],
            "region": n.get("region", "answer"),
            "label": n.get("label", ""),
            "source": "old",
        })
        old_keys.add((int(n["layer"]), int(n["index"])))
    old_count = len(combined)

    # New neurons from mlp_diff
    mlp = torch.load(MLP_DIFF_PT, weights_only=False)
    all_results = mlp["all_results"]
    new_added = []
    for L, results in sorted(all_results.items()):
        # take top-N entries with |diff| > threshold and not already known
        added_for_L = 0
        for r in results:
            if added_for_L >= TOPK_PER_LAYER:
                break
            if abs(r["diff"]) < MIN_ABS_DIFF:
                continue
            key = (L, r["neuron"])
            if key in old_keys:
                continue
            sign = 1.0 if r["diff"] > 0 else -1.0
            gain = sign * 1.0
            scale = TIER_SCALE["mlp_diff_v2"]
            eff = GLOBAL_ALPHA * gain * scale
            entry = {
                "id": f"L{L}#{r['neuron']}",
                "layer": L,
                "index": r["neuron"],
                "gain": gain,
                "effective_offset": eff,
                "tier": "mlp_diff_v2",
                "region": "answer",   # default; could refine later
                "label": (f"[mlp_diff diff={r['diff']:+.3f} "
                          f"opus={r['opus_mean']:+.3f} base={r['base_mean']:+.3f}]"),
                "source": "new",
                "diff": r["diff"],
            }
            combined.append(entry)
            new_added.append(entry)
            old_keys.add(key)
            added_for_L += 1
    new_count = len(new_added)

    # Per-layer summary
    by_layer = {}
    for x in combined:
        by_layer.setdefault(x["layer"], {"old": 0, "new": 0})
        by_layer[x["layer"]][x["source"]] += 1

    return combined, old_count, new_count, by_layer, skipped_candidate


def write_inventory_v2(combined, old_count, new_count, by_layer, skipped):
    with open(INVENTORY_V2_TXT, "w", encoding="utf-8") as f:
        f.write(f"================ NEURON_INVENTORY_v2 ================\n")
        f.write(f"  Generated by qa01_old_plus_new.py\n")
        f.write(f"  Sources:\n")
        f.write(f"    OLD: chat/neurons.json (handpicked, region/blockwise probes)\n")
        f.write(f"    NEW: outputs/opus47_mlp_diff/raw.pt (Opus vs Gemma mean diff at L20-L32)\n\n")
        f.write(f"  Counts:\n")
        f.write(f"    OLD (kept):     {old_count}\n")
        f.write(f"    OLD (skipped candidate): {skipped}\n")
        f.write(f"    NEW (mlp_diff): {new_count}\n")
        f.write(f"    TOTAL:          {old_count + new_count}\n\n")
        f.write(f"  Per-layer breakdown:\n")
        f.write(f"    Layer | old | new | total\n")
        for L in sorted(by_layer.keys()):
            o = by_layer[L]["old"]
            n = by_layer[L]["new"]
            f.write(f"    L{L:>3}   | {o:>3} | {n:>3} | {o+n:>5}\n")
        f.write("\n")
        f.write(f"================ Detail listing ================\n")
        f.write(f"  ID            tier            region   gain   eff    label\n")
        f.write(f"  " + "-" * 90 + "\n")
        for x in sorted(combined, key=lambda e: (e["layer"], e["index"])):
            f.write(f"  {x['id']:<14}{x['tier']:<16}{x['region']:<8} "
                    f"{x['gain']:+5.2f}  {x['effective_offset']:+5.2f}  "
                    f"{x['label'][:40]}\n")
        # Highlight new ones at end with diff value
        f.write(f"\n================ New neurons (with diff) ================\n")
        f.write(f"  ID           layer  diff       sign\n")
        for x in combined:
            if x["source"] != "new":
                continue
            diff = x.get("diff", 0)
            f.write(f"  {x['id']:<13}  L{x['layer']:>2}    "
                    f"{diff:+7.3f}    {'+' if diff > 0 else '-'}\n")


# -------- main --------
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


def get_eos_ids(tokenizer):
    eos = []
    for s in ("<turn|>", "<end_of_turn>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
                eos.append(tid)
        except Exception:
            pass
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return list({i for i in eos if i is not None})


def main():
    print("=" * 70, flush=True)
    print("  Step 1: build NEURON_INVENTORY_v2 (old + new)", flush=True)
    print("=" * 70, flush=True)
    combined, old_count, new_count, by_layer, skipped = build_combined_inventory()
    write_inventory_v2(combined, old_count, new_count, by_layer, skipped)
    print(f"  OLD: {old_count}  NEW: {new_count}  total: {len(combined)}  "
          f"(skipped {skipped} candidate-tier)", flush=True)
    print(f"  per-layer:", flush=True)
    for L in sorted(by_layer.keys()):
        print(f"    L{L:>2}: old={by_layer[L]['old']:>2} new={by_layer[L]['new']:>2}",
              flush=True)
    print(f"  saved → {INVENTORY_V2_TXT}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  Step 2: load model + install hooks + generate (loop detection on)", flush=True)
    print("=" * 70, flush=True)

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
    print(f"  alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)

    eoc_id = tokenizer.convert_tokens_to_ids("<channel|>")

    # ---- prepare prompt ----
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = qa["input"]
    msgs = [{"role": "user", "content": user_text}]
    enc = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    ids = enc["input_ids"].to(DEVICE)
    am = enc["attention_mask"].to(DEVICE)
    prompt_len = ids.shape[1]
    print(f"  prompt_tokens = {prompt_len}", flush=True)

    # ---- region splits ----
    def to_sel(x):
        return {"id": x["id"], "layer": x["layer"], "index": x["index"],
                "gain": x["effective_offset"]}   # gain here already includes tier_scale
    thought_sel = [to_sel(x) for x in combined if x["region"] == "thought"]
    answer_sel  = [to_sel(x) for x in combined if x["region"] == "answer"]
    always_sel  = [to_sel(x) for x in combined if x["region"] == "always"]
    print(f"  region split: thought={len(thought_sel)}  answer={len(answer_sel)}  always={len(always_sel)}",
          flush=True)

    # ---- install hooks (mirror chat/runtime.py logic for think_mode=True) ----
    hooks = []
    if thought_sel:
        tg = ClampGate(allow=True)
        hooks += install_clamp_hooks(layers, thought_sel, 1.0, gate=tg)
        hooks.append(install_segment_gate(model, tg, eoc_id, on_seen_set_to=False))
    if answer_sel:
        ag = ClampGate(allow=False)
        hooks += install_clamp_hooks(layers, answer_sel, 1.0, gate=ag)
        hooks.append(install_segment_gate(model, ag, eoc_id, on_seen_set_to=True))
    if always_sel:
        hooks += install_clamp_hooks(layers, always_sel, 1.0, gate=None)
    print(f"  installed {len(hooks)} hooks", flush=True)

    # ---- generate with loop stopper ----
    stopper = LoopStopper(tokenizer, prompt_len=prompt_len)
    eos_ids = get_eos_ids(tokenizer)

    print(f"\n[generate] max_new_tokens={MAX_NEW_TOKENS}, eos_ids={eos_ids}", flush=True)
    t0 = time.time()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model.generate(
                input_ids=ids,
                attention_mask=am,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=eos_ids if eos_ids else None,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([stopper]),
            )
    finally:
        remove_hooks(hooks)

    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    full = out[0].detach().cpu()
    gen = full[prompt_len:]
    text = tokenizer.decode(gen, skip_special_tokens=False)
    print(f"[done] {gen.shape[0]} tokens in {dt:.1f}s  peak={peak:.2f}GB  "
          f"loop_tripped={stopper.tripped}", flush=True)

    # ---- save ----
    thought, response = split_thought_response(text)
    OUT_TXT.write_text(
        "================ INPUT ================\n"
        + user_text
        + "\n\n================ COT (thought) ================\n"
        + thought
        + "\n\n================ ANSWER (response) ================\n"
        + response
        + "\n\n================ META ================\n"
        + "label             = old_plus_new (region-aware hooks, no chain)\n"
        + f"old_neuron_count  = {old_count}\n"
        + f"new_neuron_count  = {new_count}\n"
        + f"total_neurons     = {len(combined)}\n"
        + f"prompt_tokens     = {prompt_len}\n"
        + f"gen_tokens        = {gen.shape[0]}\n"
        + f"elapsed_s         = {dt:.2f}\n"
        + f"peak_gpu_gb       = {peak:.2f}\n"
        + f"thought_chars     = {len(thought)}\n"
        + f"answer_chars      = {len(response)}\n"
        + f"loop_stopped      = {stopper.tripped}\n"
        + f"loop_reason       = {stopper.reason}\n",
        encoding="utf-8",
    )
    print(f"\n[save] {OUT_TXT}", flush=True)
    print(f"[save] thought={len(thought)}c  answer={len(response)}c", flush=True)
    print(f"[save] answer head: {response[:200].replace(chr(10), ' ')!r}", flush=True)


if __name__ == "__main__":
    main()
