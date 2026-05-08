"""
Multi-QA neuron-diff pipeline.

For each QA file in data/ (except the template), generate Gemma's own answer,
forward both [prompt + Gemma_ans] and [prompt + Opus_ans] (chunked, KV-cached),
capture per-neuron mean/std/active_rate over the answer span, and save the diff.

Each QA's results go to outputs/qa<NN>/diff.pt . Skips QAs that already have
a saved diff (so this script can resume).
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
DATA_DIR = ROOT / "data"
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DEVICE = "cuda:0"
MAX_NEW_TOKENS = 900
ACTIVE_THRESH = 1.0
CHUNK = 1024


def list_qa_files():
    files = sorted(DATA_DIR.glob("claudeopusQA*.json"))
    return files


def build_prompt_ids(tokenizer, user_input):
    msgs = [{"role": "user", "content": user_input}]
    o = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
    ids = o["input_ids"]
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids)
    return ids


def gen_answer(model, tokenizer, prompt_ids, label):
    print(f"  [gen-{label}] prompt_len={prompt_ids.shape[1]}")
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
    print(f"  [gen-{label}] {gen_ids.shape[0]} tokens in {time.time()-t0:.1f}s")
    return gen_ids


def forward_capture_stats(model, layers, full_ids, span_start):
    T = full_ids.shape[0]
    span_len = T - span_start
    sizes = [layer.mlp.down_proj.in_features for layer in layers]
    sums = [torch.zeros(s, dtype=torch.float32) for s in sizes]
    sumsq = [torch.zeros(s, dtype=torch.float32) for s in sizes]
    active = [torch.zeros(s, dtype=torch.float32) for s in sizes]
    counts = [0] * len(layers)
    chunk_offset = {"v": 0}

    def make_hook(idx):
        def fn(module, inputs, output):
            x = inputs[0]
            chunk_len = x.shape[1]
            global_start = chunk_offset["v"]
            local_lo = max(0, span_start - global_start)
            if local_lo >= chunk_len:
                return
            seg = x[0, local_lo:, :].to(torch.float32)
            sums[idx] += seg.sum(dim=0).cpu()
            sumsq[idx] += (seg * seg).sum(dim=0).cpu()
            active[idx] += (seg.abs() > ACTIVE_THRESH).to(torch.float32).sum(dim=0).cpu()
            counts[idx] += seg.shape[0]
        return fn

    hooks = [layer.mlp.down_proj.register_forward_hook(make_hook(i))
             for i, layer in enumerate(layers)]
    past_kv = None
    pos = 0
    try:
        while pos < T:
            end = min(pos + CHUNK, T)
            chunk_ids = full_ids[pos:end].unsqueeze(0).to(DEVICE)
            chunk_offset["v"] = pos
            cache_pos = torch.arange(pos, end, device=DEVICE)
            with torch.no_grad():
                out = model(
                    input_ids=chunk_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    cache_position=cache_pos,
                )
            past_kv = out.past_key_values
            pos = end
            torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    means, stds, rates = [], [], []
    for s, sq, ac, c in zip(sums, sumsq, active, counts):
        assert c == span_len, f"count mismatch {c} vs {span_len}"
        m = s / c
        var = (sq / c) - m * m
        var.clamp_(min=0.0)
        means.append(m)
        stds.append(var.sqrt())
        rates.append(ac / c)
    return {"mean": means, "std": stds, "active_rate": rates}


def process_one_qa(model, tokenizer, layers, qa_path):
    qa_id = qa_path.stem.replace("claudeopusQA", "qa")  # qa01, qa02, ...
    out_dir = ROOT / "outputs" / qa_id
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_path = out_dir / "diff.pt"
    if diff_path.exists():
        print(f"[{qa_id}] skip (diff.pt exists)")
        return diff_path

    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    print(f"[{qa_id}] model={qa.get('model')}, input={len(qa['input'])} chars, "
          f"output={len(qa['output'])} chars")

    prompt_ids = build_prompt_ids(tokenizer, qa["input"])
    P = prompt_ids.shape[1]

    gemma_ans_ids = gen_answer(model, tokenizer, prompt_ids, qa_id)
    (out_dir / "gemma_answer.txt").write_text(
        tokenizer.decode(gemma_ans_ids, skip_special_tokens=False),
        encoding="utf-8",
    )

    opus_ans_ids = tokenizer.encode(qa["output"], add_special_tokens=False, return_tensors="pt")[0]

    aligned_len = min(int(gemma_ans_ids.shape[0]), int(opus_ans_ids.shape[0]))
    gemma_ans_ids = gemma_ans_ids[:aligned_len]
    opus_ans_ids = opus_ans_ids[:aligned_len]
    print(f"  [align] span_len={aligned_len}")

    seq_a = torch.cat([prompt_ids[0], gemma_ans_ids], dim=0)
    seq_b = torch.cat([prompt_ids[0], opus_ans_ids], dim=0)

    print(f"  [fwd-A]")
    stats_a = forward_capture_stats(model, layers, seq_a, span_start=P)
    torch.cuda.empty_cache()
    print(f"  [fwd-B]")
    stats_b = forward_capture_stats(model, layers, seq_b, span_start=P)
    torch.cuda.empty_cache()

    diff = [b - a for a, b in zip(stats_a["mean"], stats_b["mean"])]

    n_layers = len(layers)
    sizes = [l.mlp.down_proj.in_features for l in layers]
    torch.save(
        {
            "qa_id": qa_id,
            "model": qa.get("model"),
            "stats_a": stats_a,
            "stats_b": stats_b,
            "diff": diff,
            "n_layers": n_layers,
            "sizes": sizes,
            "prompt_len": P,
            "answer_span_len": aligned_len,
        },
        diff_path,
    )
    print(f"  [save] {diff_path}")
    return diff_path


def main():
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
    print(f"[load] alloc={torch.cuda.memory_allocated()/1e9:.2f}GB, layers={len(layers)}")

    qa_files = list_qa_files()
    print(f"[plan] processing {len(qa_files)} QAs:")
    for f in qa_files:
        print(f"  - {f.name}")

    for qa_path in qa_files:
        try:
            process_one_qa(model, tokenizer, layers, qa_path)
        except Exception as e:
            print(f"[err] {qa_path.name}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    main()
