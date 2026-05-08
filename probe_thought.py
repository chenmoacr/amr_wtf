"""
Thought-segment differential probe.

Goal: find functional neurons that drive "good thinking" by diffing Gemma's
own thought-mode generation against a synthetic ideal CoT (anchored with
specific knowledge for chat.openai.com GPT-3.5 era).

Regions (per token):
  0 = unlabeled (prompt + control tokens)
  1 = thought   (between <|channel>thought\\n and <channel|>)
  2 = answer    (after <channel|> until <turn|>)

Outputs:
  outputs/thought_qa01/diff_thought.pt   (per-region per-neuron stats)
  outputs/thought_qa01/gemma_thought.txt (Gemma's own thought, for inspection)
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
SYN_PATH = ROOT / "data" / "synthetic_thought_qa01.json"
REF_ANS_PATH = ROOT / "data" / "gemini_3_flash_preview_code_qa_01.json"
OUT_DIR = ROOT / "outputs" / "thought_qa01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
MAX_NEW_TOKENS = 4500  # thought + answer
ACTIVE_THRESH = 1.0
CHUNK = 1024
NUM_REGIONS = 3
REGION_NAMES = ["unlabeled", "thought", "answer"]


def get_layers(model):
    return model.model.language_model.layers


def build_prompt_ids(tokenizer, user_input: str):
    """Prompt with enable_thinking=True, ending at <|turn>model\\n."""
    msgs = [{"role": "user", "content": user_input}]
    o = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=True,
    )
    return o["input_ids"][0]  # [P]


def gen_gemma_thought(model, tokenizer, prompt_ids):
    """Generate Gemma's full thought + answer in think mode."""
    print(f"  [gen] prompt_len={prompt_ids.shape[0]}")
    eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    eos_ids = [i for i in eos_ids if i is not None and i >= 0]
    t0 = time.time()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.generate(
            prompt_ids.unsqueeze(0).to(DEVICE),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )
    full = out[0].cpu()
    gen_ids = full[prompt_ids.shape[0]:]
    print(f"  [gen] {gen_ids.shape[0]} tokens in {time.time()-t0:.1f}s")
    return gen_ids


def build_reference_assistant_ids(tokenizer, synthetic_thought: str, answer: str):
    """Build [<|channel>, 'thought\\n', synth, '\\n', <channel|>, answer, <turn|>]."""
    soc = tokenizer.convert_tokens_to_ids("<|channel>")
    eoc = tokenizer.convert_tokens_to_ids("<channel|>")
    eot = tokenizer.convert_tokens_to_ids("<turn|>")
    parts = []
    parts.append(torch.tensor([soc], dtype=torch.long))
    parts.append(torch.tensor(tokenizer.encode("thought\n", add_special_tokens=False), dtype=torch.long))
    parts.append(torch.tensor(tokenizer.encode(synthetic_thought, add_special_tokens=False), dtype=torch.long))
    parts.append(torch.tensor(tokenizer.encode("\n", add_special_tokens=False), dtype=torch.long))
    parts.append(torch.tensor([eoc], dtype=torch.long))
    parts.append(torch.tensor(tokenizer.encode(answer, add_special_tokens=False), dtype=torch.long))
    parts.append(torch.tensor([eot], dtype=torch.long))
    return torch.cat(parts, dim=0)


def label_regions(gen_ids: torch.Tensor, eoc_id: int, eot_id: int) -> torch.Tensor:
    """Label every token in gen_ids: 1=thought, 2=answer, 0=unlabeled (after <turn|>)."""
    n = gen_ids.shape[0]
    labels = torch.zeros(n, dtype=torch.uint8)

    eoc_pos = (gen_ids == eoc_id).nonzero(as_tuple=True)[0]
    if eoc_pos.numel() == 0:
        # no <channel|>, treat all as unlabeled
        print("  [warn] <channel|> not found in sequence; all tokens marked unlabeled")
        return labels

    eoc_idx = int(eoc_pos[0].item())
    # thought region: [0, eoc_idx)
    labels[:eoc_idx] = 1

    # answer region: from eoc_idx+1 until <turn|> or EOS, exclusive
    ans_lo = eoc_idx + 1
    eot_pos = (gen_ids[ans_lo:] == eot_id).nonzero(as_tuple=True)[0]
    if eot_pos.numel() > 0:
        ans_hi = ans_lo + int(eot_pos[0].item())
    else:
        ans_hi = n
    labels[ans_lo:ans_hi] = 2
    return labels


def forward_capture_regions(model, layers, full_ids, span_start, region_labels):
    """Chunked prefill capturing per-(layer,region,neuron) sum/sumsq/active/count."""
    T = full_ids.shape[0]
    span_len = T - span_start
    assert region_labels.shape[0] == span_len

    sizes = [layer.mlp.down_proj.in_features for layer in layers]
    sums = [[torch.zeros(s, dtype=torch.float32) for _ in range(NUM_REGIONS)] for s in sizes]
    sumsq = [[torch.zeros(s, dtype=torch.float32) for _ in range(NUM_REGIONS)] for s in sizes]
    active = [[torch.zeros(s, dtype=torch.float32) for _ in range(NUM_REGIONS)] for s in sizes]
    counts = [[0] * NUM_REGIONS for _ in sizes]
    chunk_offset = {"v": 0}
    region_labels_gpu = region_labels.to(DEVICE)

    def make_hook(idx):
        def fn(module, inputs, output):
            x = inputs[0]
            chunk_len = x.shape[1]
            global_start = chunk_offset["v"]
            local_lo = max(0, span_start - global_start)
            if local_lo >= chunk_len:
                return
            seg = x[0, local_lo:, :].to(torch.float32)
            seg_abs_start = global_start + local_lo
            rel_lo = seg_abs_start - span_start
            rel_hi = rel_lo + seg.shape[0]
            seg_labels = region_labels_gpu[rel_lo:rel_hi]
            for r in range(NUM_REGIONS):
                mask = (seg_labels == r)
                cnt = int(mask.sum().item())
                if cnt == 0:
                    continue
                sub = seg[mask, :]
                sums[idx][r] += sub.sum(dim=0).cpu()
                sumsq[idx][r] += (sub * sub).sum(dim=0).cpu()
                active[idx][r] += (sub.abs() > ACTIVE_THRESH).to(torch.float32).sum(dim=0).cpu()
                counts[idx][r] += cnt
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

    means, rates = [], []
    for li in range(len(layers)):
        mr, ar = [], []
        for r in range(NUM_REGIONS):
            c = counts[li][r]
            if c == 0:
                mr.append(None); ar.append(None); continue
            m = sums[li][r] / c
            mr.append(m); ar.append(active[li][r] / c)
        means.append(mr); rates.append(ar)
    return {"mean": means, "active_rate": rates, "counts": counts}


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
    layers = get_layers(model)
    n_layers = len(layers)
    sizes = [l.mlp.down_proj.in_features for l in layers]
    print(f"[load] alloc={torch.cuda.memory_allocated()/1e9:.2f}GB, layers={n_layers}")

    # ---- read inputs ----
    syn = json.loads(SYN_PATH.read_text(encoding="utf-8"))
    user_input = syn["input"]
    synthetic_thought = syn["synthetic_thought"]
    ref = json.loads(REF_ANS_PATH.read_text(encoding="utf-8"))
    ref_answer = ref["output"]
    print(f"[data] user_input={len(user_input)} chars, synth_thought={len(synthetic_thought)} chars, ref_answer={len(ref_answer)} chars")

    # ---- tokens ----
    soc_id = tokenizer.convert_tokens_to_ids("<|channel>")
    eoc_id = tokenizer.convert_tokens_to_ids("<channel|>")
    eot_id = tokenizer.convert_tokens_to_ids("<turn|>")
    print(f"[tok] soc={soc_id}, eoc={eoc_id}, eot={eot_id}")

    prompt_ids = build_prompt_ids(tokenizer, user_input)
    P = prompt_ids.shape[0]
    print(f"[prompt] tokens={P} (think mode ON)")

    # ---- A: Gemma's own thought + answer ----
    gemma_path = OUT_DIR / "gemma_full.pt"
    if gemma_path.exists():
        gemma_gen = torch.load(gemma_path, weights_only=True)
        print(f"[reuse] gemma_full.pt -> {gemma_gen.shape[0]} tokens")
    else:
        print("[gen-A] generating Gemma own thought+answer...")
        gemma_gen = gen_gemma_thought(model, tokenizer, prompt_ids)
        torch.save(gemma_gen, gemma_path)
        full_text = tokenizer.decode(gemma_gen, skip_special_tokens=False)
        (OUT_DIR / "gemma_full.txt").write_text(full_text, encoding="utf-8")

    labels_a = label_regions(gemma_gen, eoc_id, eot_id)
    counts_a = torch.bincount(labels_a.long(), minlength=NUM_REGIONS).tolist()
    print(f"[label A] {dict(zip(REGION_NAMES, counts_a))}")
    if counts_a[1] == 0:
        print("[err] Gemma generated no thought segment; aborting")
        return

    # ---- B: synthetic thought + Gemini answer ----
    ref_assistant_ids = build_reference_assistant_ids(tokenizer, synthetic_thought, ref_answer)
    labels_b = label_regions(ref_assistant_ids, eoc_id, eot_id)
    counts_b = torch.bincount(labels_b.long(), minlength=NUM_REGIONS).tolist()
    print(f"[label B] {dict(zip(REGION_NAMES, counts_b))}")

    # ---- forward both ----
    seq_a = torch.cat([prompt_ids, gemma_gen], dim=0)
    seq_b = torch.cat([prompt_ids, ref_assistant_ids], dim=0)
    print(f"[seq] A_len={seq_a.shape[0]}, B_len={seq_b.shape[0]}")

    print("[fwd-A]...")
    stats_a = forward_capture_regions(model, layers, seq_a, span_start=P, region_labels=labels_a)
    torch.cuda.empty_cache()

    print("[fwd-B]...")
    stats_b = forward_capture_regions(model, layers, seq_b, span_start=P, region_labels=labels_b)
    torch.cuda.empty_cache()

    # ---- diff per region ----
    diff = []
    for li in range(n_layers):
        per_r = []
        for r in range(NUM_REGIONS):
            ma = stats_a["mean"][li][r]
            mb = stats_b["mean"][li][r]
            per_r.append(None if (ma is None or mb is None) else (mb - ma))
        diff.append(per_r)

    # ---- print top neurons in thought region ----
    def render_top(region_id: int, label: str, top: int = 25, l_lo: int = 15):
        print(f"\n=== Top |diff| in region '{label}' (L>={l_lo}, top {top}) ===")
        pairs = []
        for li in range(l_lo, n_layers):
            d = diff[li][region_id]
            if d is None:
                continue
            for n in range(d.shape[0]):
                pairs.append((abs(d[n].item()), li, n, d[n].item()))
        pairs.sort(reverse=True)
        ra_list = stats_a["active_rate"]
        rb_list = stats_b["active_rate"]
        print(f"{'rank':>4} {'layer#nrn':>11} {'diff':>7} {'rate_A':>7} {'rate_B':>7}")
        for k, (_, li, n, v) in enumerate(pairs[:top]):
            ra = ra_list[li][region_id]
            rb = rb_list[li][region_id]
            ra_v = ra[n].item() if ra is not None else float("nan")
            rb_v = rb[n].item() if rb is not None else float("nan")
            print(f"  {k+1:>2}  L{li:02d}#{n:<5d}  {v:+.3f}  {ra_v:.3f}  {rb_v:.3f}")

    render_top(1, "thought", top=25)
    render_top(2, "answer", top=15)

    # ---- save ----
    torch.save(
        {
            "qa_id": "thought_qa01",
            "ref_model": syn["model"],
            "n_layers": n_layers,
            "sizes": sizes,
            "prompt_len": P,
            "stats_a": stats_a,
            "stats_b": stats_b,
            "diff_per_region": diff,
            "region_token_counts_a": counts_a,
            "region_token_counts_b": counts_b,
            "region_names": REGION_NAMES,
            "labels_a": labels_a,
            "labels_b": labels_b,
        },
        OUT_DIR / "diff_thought.pt",
    )
    print(f"\n[save] {OUT_DIR / 'diff_thought.pt'}")


if __name__ == "__main__":
    main()
