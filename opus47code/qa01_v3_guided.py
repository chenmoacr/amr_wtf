"""
QA01 v3 — focused structural neurons run, with new guided_diff candidates.

Compared to qa01_old_plus_new.py (which dumped everything):
  * Drops math-tier neurons (different domain — would interfere with literary critique)
  * Drops thought-only neurons (Gemma 4 CoT and output use mostly disjoint
    neuron sets — thought neurons add noise without affecting answer)
  * Keeps verified / general / lit / guided_diff in {answer, always} regions
  * NON-think mode (think mode suppresses output neurons)

Selected set (~22 neurons):
  - L15#1511 / L26#449 / L16#1298  (verified scaffolds)
  - L26#5430 / L27#2890            (general tone / structure)
  - All lit/answer entries (L15#8146, L23#287, L27#10024, L27#6644, L27#8968,
    L27#5590, L28#7686, L31#604, L32#8474, L32#9383, L34#8522)
  - 7 NEW guided_diff candidates (L27#10897/8248/6013/7036/10300/5512/10719)

All run with default_gain × tier_scale=1.0. No global alpha boost.
Think mode = False — entire generation IS the answer; per region_legend
"answer: 非 think 模式下视作全程", so answer neurons install as always-on.

Output:
  outputs/qa01_compare/v3_guided.txt  (input + answer + meta)
  outputs/qa01_compare/v3_guided.log
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, str(Path("J:/amr/amr_wtf/chat")))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    AutoTokenizer,
    Gemma4ForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)
from steering import (   # noqa: E402
    install_clamp_hooks,
    remove_hooks,
)

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "claudeopusQA01.json"
NEURONS_JSON = ROOT / "chat" / "neurons.json"
OUT_DIR = ROOT / "outputs" / "qa01_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT = OUT_DIR / "v3_guided.txt"

DEVICE = "cuda:0"
MAX_NEW_TOKENS = 4096

KEEP_TIERS = {"verified", "general", "lit", "guided_diff"}
KEEP_REGIONS = {"answer", "always"}


class LoopStopper(StoppingCriteria):
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
        tail_ids = input_ids[0, max(self.prompt_len, n_total - 800):].tolist()
        text = self.tokenizer.decode(tail_ids, skip_special_tokens=False)
        if len(text) > self.window_chars:
            text = text[-self.window_chars:]
        if len(text) < self.chunk_size * 2:
            return False
        seen = {}
        for i in range(0, len(text) - self.chunk_size, 1):
            chunk = text[i:i + self.chunk_size]
            seen[chunk] = seen.get(chunk, 0) + 1
            if seen[chunk] >= self.repeat_count:
                self.tripped = True
                self.reason = (f"chunk repeated {seen[chunk]}× : {chunk!r}")
                print(f"  [LoopStopper] TRIPPED: {self.reason}", flush=True)
                return True
        return False


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
    print("  QA01 v3 — focused structural neurons + 7 guided_diff candidates", flush=True)
    print("=" * 70, flush=True)

    nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
    selected = []
    for n in nj["known_neurons"]:
        tier = n["tier"]
        region = n.get("region", "answer")
        if tier not in KEEP_TIERS:
            continue
        if region not in KEEP_REGIONS:
            continue
        selected.append(n)

    print(f"\n  Selected {len(selected)} neurons (tier ∈ {KEEP_TIERS}, "
          f"region ∈ {KEEP_REGIONS}):", flush=True)
    for n in selected:
        print(f"    {n['id']:<12}  L{n['layer']:>2}#{n['index']:<5}  "
              f"gain={n['default_gain']:+5.1f}  tier={n['tier']:<12}  "
              f"region={n['region']:<7}  {n.get('label','')[:50]}",
              flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  Loading model ...", flush=True)
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

    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = qa["input"]
    msgs = [{"role": "user", "content": user_text}]
    enc = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    ids = enc["input_ids"].to(DEVICE)
    am = enc["attention_mask"].to(DEVICE)
    prompt_len = ids.shape[1]
    print(f"\n  prompt_tokens = {prompt_len}  (think_mode=False)", flush=True)

    # Non-think mode: entire generation = answer; per region_legend,
    # answer-region neurons install as always-on (no segment gate needed).
    def to_sel(x):
        return {"id": x["id"], "layer": x["layer"], "index": x["index"],
                "gain": float(x["default_gain"])}
    all_sel = [to_sel(x) for x in selected]
    print(f"  installing {len(all_sel)} neurons as always-on "
          f"(answer + always merged, non-think mode)", flush=True)

    hooks = install_clamp_hooks(layers, all_sel, 1.0, gate=None)
    print(f"  installed {len(hooks)} hooks\n", flush=True)

    stopper = LoopStopper(tokenizer, prompt_len=prompt_len)
    eos_ids = get_eos_ids(tokenizer)
    print(f"[generate] max_new_tokens={MAX_NEW_TOKENS}, eos_ids={eos_ids}",
          flush=True)
    t0 = time.time()
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = model.generate(
                input_ids=ids, attention_mask=am,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
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
    print(f"\n[done] {gen.shape[0]} tokens in {dt:.1f}s  peak={peak:.2f}GB  "
          f"loop_tripped={stopper.tripped}", flush=True)

    response = text.rstrip("<turn|>").rstrip()
    OUT_TXT.write_text(
        "================ INPUT ================\n"
        + user_text
        + "\n\n================ ANSWER (response, non-think mode) ================\n"
        + response
        + "\n\n================ META ================\n"
        + "label             = v3_guided (focused structural + 7 new guided_diff, NO COT)\n"
        + f"selected_neurons  = {len(selected)}\n"
        + f"prompt_tokens     = {prompt_len}\n"
        + f"gen_tokens        = {gen.shape[0]}\n"
        + f"elapsed_s         = {dt:.2f}\n"
        + f"peak_gpu_gb       = {peak:.2f}\n"
        + f"answer_chars      = {len(response)}\n"
        + f"loop_stopped      = {stopper.tripped}\n"
        + f"loop_reason       = {stopper.reason}\n",
        encoding="utf-8",
    )
    print(f"\n[save] {OUT_TXT}", flush=True)
    print(f"[save] answer={len(response)}c", flush=True)
    print(f"[save] answer head: {response[:300].replace(chr(10), ' ')!r}",
          flush=True)


if __name__ == "__main__":
    main()
