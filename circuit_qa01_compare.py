"""
QA01 (丽萨小姐) compare: baseline vs allin

- baseline: vanilla Gemma 4 E2B-it, no intervention
- allin:    OpusSAE merged model (57 SAE bias already in weights)
            + 7-node chain amplify hook (×2 multiplier)

Both runs are greedy (T=0), max_new_tokens 4096 (well above QA01 typical
output), with enable_thinking=true so we see thought + response.

Two models are loaded sequentially (not in parallel) so the 12 GB GPU
fits one at a time. Each load is ~9 GB, generate peaks ~11 GB.

Output:
  outputs/qa01_compare/baseline.txt
  outputs/qa01_compare/allin.txt
  outputs/qa01_compare/diff.txt    (char-level + section-level)
  outputs/qa01_compare/run.log
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
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
BASELINE_MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
ALLIN_MODEL_PATH = ROOT / "gemma4_E2b_OpusSAE"
QA01_PATH = ROOT / "data" / "claudeopusQA01.json"
OUT_DIR = ROOT / "outputs" / "qa01_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
# QA01 baseline output (Opus reference) is ~1082 chars ≈ 500-700 tok.
# Cap at 1536 — enough headroom while keeping KV growth small to avoid
# OOM on 12GB GPU given QA01's ~3000 token prefill.
MAX_NEW_TOKENS = 1536

# Same 7-node chain as circuit_chain_qa.py
CHAIN = [
    ( 9,  357),
    ( 9,  420),
    ( 9, 6039),
    (11, 2089),
    (12, 4638),
    (24, 5330),
    (33, 9054),
]


def make_amplify_hook(neuron_idx, factor=2.0):
    def pre(module, inputs):
        x = inputs[0].clone()
        x[..., neuron_idx] = x[..., neuron_idx] * factor
        return (x,) + inputs[1:]
    return pre


def install_chain_amplify(layers):
    handles = []
    for li, ni in CHAIN:
        h = layers[li].mlp.down_proj.register_forward_pre_hook(
            make_amplify_hook(ni)
        )
        handles.append(h)
    return handles


def remove_handles(handles):
    for h in handles:
        h.remove()


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
    eos_ids = []
    for tok_str in ("<turn|>", "<end_of_turn>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
                eos_ids.append(tid)
        except Exception:
            pass
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    return list({i for i in eos_ids if i is not None})


def generate_one(model, tokenizer, prompt_text, label):
    msgs = [{"role": "user", "content": prompt_text}]
    enc = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    ids = enc["input_ids"].to(DEVICE)
    am = enc["attention_mask"].to(DEVICE)
    eos_ids = get_eos_ids(tokenizer)
    t0 = time.time()
    # Force EFFICIENT (memory-efficient) attention: fused Q@K@V without
    # materializing the [B, H, T, T] attention map. Crucial on 12GB with
    # 3000+ token prefill — MATH backend would OOM.
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.generate(
            input_ids=ids,
            attention_mask=am,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids if eos_ids else None,
            use_cache=True,
        )
    full = out[0].detach().cpu()
    gen = full[ids.shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=False)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    prompt_tok = ids.shape[1]
    print(f"    [{label}] {gen.shape[0]} tokens in {dt:.1f}s  (peak={peak:.2f}GB)", flush=True)
    del out, ids, am
    torch.cuda.reset_peak_memory_stats(0)
    return text, gen.shape[0], dt, prompt_tok


def load_and_disable_multimodal(path):
    print(f"[load] {path}", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    model.eval()
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    print(f"  alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)
    return model


def write_run(out_path: Path, label: str, prompt: str, text: str,
              n_tok: int, dt: float, prompt_tok: int):
    thought, response = split_thought_response(text)
    out_path.write_text(
        "================ INPUT ================\n"
        + prompt
        + "\n\n================ COT (thought) ================\n"
        + thought
        + "\n\n================ ANSWER (response) ================\n"
        + response
        + "\n\n================ META ================\n"
        + f"label         = {label}\n"
        + f"prompt_tokens = {prompt_tok}\n"
        + f"gen_tokens    = {n_tok}\n"
        + f"elapsed_s     = {dt:.2f}\n"
        + f"thought_chars = {len(thought)}\n"
        + f"answer_chars  = {len(response)}\n",
        encoding="utf-8",
    )
    return thought, response


def main():
    print("=" * 70)
    print("  QA01 (丽萨小姐) — baseline vs allin (OpusSAE + 7-chain amplify)")
    print("=" * 70)
    print("[warn] If vLLM is still serving on this GPU, this script will OOM.", flush=True)
    print("[warn] Stop vLLM (ctrl-C in WSL) before running this.", flush=True)

    qa01 = json.loads(QA01_PATH.read_text(encoding="utf-8"))
    prompt = qa01["input"]
    print(f"[input] QA01 prompt = {len(prompt)} chars", flush=True)

    # tokenizer is the same for both runs
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL_PATH)

    # ---------- run baseline ----------
    print("\n--- run 1/2: baseline (vanilla Gemma 4 E2B) ---", flush=True)
    model = load_and_disable_multimodal(BASELINE_MODEL_PATH)
    text_b, n_b, dt_b, p_b = generate_one(model, tokenizer, prompt, "baseline")
    th_b, ans_b = write_run(
        OUT_DIR / "baseline.txt", "baseline", prompt, text_b, n_b, dt_b, p_b,
    )
    print(f"      thought={len(th_b)}c  answer={len(ans_b)}c", flush=True)
    print(f"      answer head: {ans_b[:200].replace(chr(10), ' ')!r}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU after unload: {torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)

    # ---------- run allin ----------
    print("\n--- run 2/2: allin (OpusSAE merged + 7-chain amplify) ---", flush=True)
    print(f"  loading OpusSAE merged model from {ALLIN_MODEL_PATH}", flush=True)
    model = load_and_disable_multimodal(str(ALLIN_MODEL_PATH))
    layers = model.model.language_model.layers
    print(f"  installing 7-node chain amplify hooks ({len(CHAIN)} nodes)", flush=True)
    handles = install_chain_amplify(layers)
    try:
        text_a, n_a, dt_a, p_a = generate_one(model, tokenizer, prompt, "allin")
    finally:
        remove_handles(handles)
    th_a, ans_a = write_run(
        OUT_DIR / "allin.txt", "allin (SAE bias + 7-chain amplify)",
        prompt, text_a, n_a, dt_a, p_a,
    )
    print(f"      thought={len(th_a)}c  answer={len(ans_a)}c", flush=True)
    print(f"      answer head: {ans_a[:200].replace(chr(10), ' ')!r}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------- diff summary ----------
    print("\n--- diff summary ---", flush=True)
    import difflib
    sm_full = difflib.SequenceMatcher(None, text_b, text_a, autojunk=False)
    sm_th = difflib.SequenceMatcher(None, th_b, th_a, autojunk=False)
    sm_an = difflib.SequenceMatcher(None, ans_b, ans_a, autojunk=False)
    diff_path = OUT_DIR / "diff.txt"
    diff_path.write_text(
        "================ QA01 baseline vs allin diff summary ================\n\n"
        + f"Generation tokens:\n"
        + f"  baseline = {n_b}\n"
        + f"  allin    = {n_a}\n"
        + f"  delta    = {n_a - n_b:+d}  ({(n_a-n_b)/max(n_b,1)*100:+.1f}%)\n\n"
        + f"Thought (CoT) chars:\n"
        + f"  baseline = {len(th_b)}\n"
        + f"  allin    = {len(th_a)}\n"
        + f"  similarity (SequenceMatcher ratio) = {sm_th.ratio():.3f}\n\n"
        + f"Answer (response) chars:\n"
        + f"  baseline = {len(ans_b)}\n"
        + f"  allin    = {len(ans_a)}\n"
        + f"  similarity (SequenceMatcher ratio) = {sm_an.ratio():.3f}\n\n"
        + f"Full text similarity = {sm_full.ratio():.3f}\n"
        + f"  (1.0 = identical, 0.0 = no overlap; expect 0.3-0.7 for meaningful intervention)\n\n"
        + "================ baseline answer (full) ================\n"
        + ans_b
        + "\n\n================ allin answer (full) ================\n"
        + ans_a
        + "\n",
        encoding="utf-8",
    )
    print(f"  full ratio={sm_full.ratio():.3f}  thought ratio={sm_th.ratio():.3f}  "
          f"answer ratio={sm_an.ratio():.3f}", flush=True)
    print(f"\n[save] {OUT_DIR}/baseline.txt", flush=True)
    print(f"[save] {OUT_DIR}/allin.txt", flush=True)
    print(f"[save] {OUT_DIR}/diff.txt   (full answer side-by-side)", flush=True)


if __name__ == "__main__":
    main()
