"""
Single-QA neuron differential pipeline (code domain).

Compares Gemma's own answer to a reference answer (here: gemini-3-flash-preview)
on a single front-end / code prompt. Saves diff.pt + console summary.

Adapted from probe_single.py — same MLP down_proj hook strategy, same chunked
prefill, same |diff| reporting, just pointed at a different QA file and output
dir, and labeled "gemini" instead of "opus".
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
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "gemini_3_flash_preview_code_qa_01.json"
OUT_DIR = ROOT / "outputs" / "code_qa01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 2200  # code answers tend to be longer than literary critique


def load_qa() -> tuple[str, str, str]:
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    return qa["input"], qa["output"], qa.get("model", "?")


def build_prompt_ids(tokenizer, user_input: str) -> torch.Tensor:
    msgs = [{"role": "user", "content": user_input}]
    out = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    ids = out["input_ids"] if "input_ids" in out else out
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids


def gen_gemma_answer(model, tokenizer, prompt_ids: torch.Tensor) -> torch.Tensor:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    print(f"[gen] prompt_len={prompt_ids.shape[1]}")
    t0 = time.time()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
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
    print(f"[gen] generated {gen_ids.shape[0]} tokens in {time.time()-t0:.1f}s")
    return gen_ids


def encode_ref_answer(tokenizer, ref_text: str) -> torch.Tensor:
    ids = tokenizer.encode(ref_text, add_special_tokens=False, return_tensors="pt")[0]
    return ids


def get_decoder_layers(model):
    return model.model.language_model.layers


def forward_capture_stats(model, layers, full_ids: torch.Tensor, span_start: int,
                          active_thresh: float = 1.0,
                          chunk: int = 1024) -> dict:
    T = full_ids.shape[0]
    span_len = T - span_start
    assert span_len > 0, f"span_len={span_len}"

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
            active[idx] += (seg.abs() > active_thresh).to(torch.float32).sum(dim=0).cpu()
            counts[idx] += seg.shape[0]
        return fn

    hooks = [layer.mlp.down_proj.register_forward_hook(make_hook(i))
             for i, layer in enumerate(layers)]

    past_kv = None
    pos = 0
    try:
        while pos < T:
            end = min(pos + chunk, T)
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


def main():
    print("[load] tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("[load] model (INT8)...")
    t0 = time.time()
    bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)
    gc.collect(); torch.cuda.empty_cache()
    print(f"[load] done in {time.time()-t0:.1f}s, alloc={torch.cuda.memory_allocated()/1e9:.2f}GB")

    layers = get_decoder_layers(model)
    n_layers = len(layers)
    sizes = [l.mlp.down_proj.in_features for l in layers]
    total_neurons = sum(sizes)
    print(f"[arch] layers={n_layers}, total MLP neurons={total_neurons}")

    user_input, ref_text, ref_model = load_qa()
    print(f"[qa] ref_model={ref_model}, input_chars={len(user_input)}, output_chars={len(ref_text)}")

    prompt_ids = build_prompt_ids(tokenizer, user_input)
    P = prompt_ids.shape[1]
    print(f"[prompt] tokens={P}")

    gemma_path = OUT_DIR / "gemma_answer.pt"
    if gemma_path.exists():
        gemma_ans_ids = torch.load(gemma_path, weights_only=True)
        print(f"[reuse] gemma_answer.pt -> {gemma_ans_ids.shape[0]} tokens")
    else:
        gemma_ans_ids = gen_gemma_answer(model, tokenizer, prompt_ids)
        torch.save(gemma_ans_ids, gemma_path)
        (OUT_DIR / "gemma_answer.txt").write_text(
            tokenizer.decode(gemma_ans_ids, skip_special_tokens=False),
            encoding="utf-8",
        )

    ref_ans_ids = encode_ref_answer(tokenizer, ref_text)
    print(f"[ref] tokens={ref_ans_ids.shape[0]}")

    aligned_len = min(int(gemma_ans_ids.shape[0]), int(ref_ans_ids.shape[0]))
    gemma_ans_ids = gemma_ans_ids[:aligned_len]
    ref_ans_ids = ref_ans_ids[:aligned_len]
    print(f"[align] both spans truncated to {aligned_len} tokens")

    seq_a = torch.cat([prompt_ids[0], gemma_ans_ids], dim=0)
    seq_b = torch.cat([prompt_ids[0], ref_ans_ids], dim=0)
    print(f"[seq] A_len={seq_a.shape[0]}, B_len={seq_b.shape[0]}")

    print("[fwd-A gemma]...")
    stats_a = forward_capture_stats(model, layers, seq_a, span_start=P)
    torch.cuda.empty_cache()

    print("[fwd-B ref/gemini]...")
    stats_b = forward_capture_stats(model, layers, seq_b, span_start=P)
    torch.cuda.empty_cache()

    diff = [b - a for a, b in zip(stats_a["mean"], stats_b["mean"])]
    flat_diff = torch.cat(diff)
    abs_diff = flat_diff.abs()

    print("\n==========  CODE QA NEURON DIFF SUMMARY  ==========")
    print(f"|diff| stats: mean={abs_diff.mean():.4f} std={abs_diff.std():.4f} max={abs_diff.max():.4f}")
    thresholds = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    print(f"\n{'threshold':>10} | {'count':>8} | {'pct':>7}")
    print("-" * 32)
    for t in thresholds:
        c = int((abs_diff > t).sum())
        print(f"{t:>10.2f} | {c:>8} | {100*c/total_neurons:>6.2f}%")

    print(f"\nglobal top-30 |diff| (layer, neuron, diff):")
    pairs = []
    for li, row in enumerate(diff):
        for n in range(row.shape[0]):
            pairs.append((abs(row[n].item()), li, n, row[n].item()))
    pairs.sort(reverse=True)
    for k in range(30):
        _, li, n, v = pairs[k]
        print(f"  #{k+1:>2}  L{li:02d}#{n:<5d}  diff={v:+.4f}")

    print(f"\nL15+ persistent neurons (|diff|>=0.2 AND max(rate_A, rate_B)>=0.3):")
    L15_pairs = []
    for li in range(15, n_layers):
        d = diff[li]
        ra = stats_a["active_rate"][li]
        rb = stats_b["active_rate"][li]
        rmax = torch.maximum(ra, rb)
        for n in range(d.shape[0]):
            if abs(d[n].item()) >= 0.2 and rmax[n].item() >= 0.3:
                L15_pairs.append((abs(d[n].item()), li, n, d[n].item(),
                                  ra[n].item(), rb[n].item(),
                                  stats_a["mean"][li][n].item(),
                                  stats_b["mean"][li][n].item()))
    L15_pairs.sort(reverse=True)
    print(f"  found {len(L15_pairs)} candidates (showing top 25)")
    print(f"  {'rank':>4} {'layer#nrn':>10} {'diff':>7} {'rate_A':>7} {'rate_B':>7} {'mean_A':>8} {'mean_B':>8}")
    for k, (_, li, n, v, ra, rb, ma, mb) in enumerate(L15_pairs[:25]):
        print(f"  {k+1:>4} L{li:02d}#{n:<5d}  {v:+.3f}  {ra:.3f}  {rb:.3f}  {ma:+.3f}  {mb:+.3f}")

    torch.save(
        {
            "qa_id": "code_qa01",
            "ref_model": ref_model,
            "stats_a": stats_a,
            "stats_b": stats_b,
            "diff": diff,
            "n_layers": n_layers,
            "sizes": sizes,
            "prompt_len": P,
            "answer_span_len": aligned_len,
        },
        OUT_DIR / "diff.pt",
    )
    print(f"\n[save] diff.pt -> {OUT_DIR / 'diff.pt'}")


if __name__ == "__main__":
    main()
