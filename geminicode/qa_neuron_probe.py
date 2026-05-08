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

# Path Configuration
ROOT = Path("J:/amr/amr_wtf")
DATA_PATH = ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json"
GEMINI_DIR = ROOT / "geminicode"
OUT_DIR = GEMINI_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

def load_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['A'][0], data['B'][0] # Take first pair for now

def build_sequences(tokenizer, item):
    query = item['query']
    cot = item['cot']
    resp = item['response']

    # Simple concat for analysis.
    # Note: Gemma 4 typically uses <thinking> tags or similar in some modes,
    # but here we use the raw text from JSON.

    prompt_ids = tokenizer.encode(query, add_special_tokens=True, return_tensors="pt")[0]
    cot_ids = tokenizer.encode(cot, add_special_tokens=False, return_tensors="pt")[0]
    resp_ids = tokenizer.encode(resp, add_special_tokens=False, return_tensors="pt")[0]

    full_ids = torch.cat([prompt_ids, cot_ids, resp_ids])

    # Regions
    p_len = len(prompt_ids)
    c_len = len(cot_ids)
    r_len = len(resp_ids)

    regions = {
        "prompt": (0, p_len),
        "think": (p_len, p_len + c_len),
        "output": (p_len + c_len, p_len + c_len + r_len)
    }
    return full_ids, regions

def capture_stats(model, layers, full_ids, regions, active_thresh=1.0):
    T = full_ids.shape[0]
    sizes = [layer.mlp.down_proj.in_features for layer in layers]

    # Store stats per region
    # stats[region][layer_idx] = sum_tensor
    stats = {reg: {
        "sum": [torch.zeros(s, dtype=torch.float32) for s in sizes],
        "active": [torch.zeros(s, dtype=torch.float32) for s in sizes],
        "count": 0
    } for reg in regions}

    chunk_offset = {"v": 0}

    def make_hook(layer_idx):
        def fn(module, inputs, output):
            x = inputs[0] # [batch, seq, hidden]
            curr_chunk_start = chunk_offset["v"]
            curr_chunk_len = x.shape[1]

            for reg_name, (reg_start, reg_end) in regions.items():
                # Find overlap between current chunk and region
                overlap_start = max(curr_chunk_start, reg_start)
                overlap_end = min(curr_chunk_start + curr_chunk_len, reg_end)

                if overlap_start < overlap_end:
                    # Tokens in this chunk that belong to this region
                    local_start = overlap_start - curr_chunk_start
                    local_end = overlap_end - curr_chunk_start

                    seg = x[0, local_start:local_end, :].to(torch.float32)
                    stats[reg_name]["sum"][layer_idx] += seg.sum(dim=0).cpu()
                    stats[reg_name]["active"][layer_idx] += (seg.abs() > active_thresh).sum(dim=0).cpu()
                    if layer_idx == 0: # Only count once per layer
                         stats[reg_name]["count"] += (local_end - local_start)
        return fn

    hooks = [layer.mlp.down_proj.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

    try:
        past_kv = None
        pos = 0
        chunk_size = 512
        while pos < T:
            end = min(pos + chunk_size, T)
            chunk_ids = full_ids[pos:end].unsqueeze(0).to(DEVICE)
            chunk_offset["v"] = pos
            with torch.no_grad():
                out = model(input_ids=chunk_ids, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            pos = end
    finally:
        for h in hooks: h.remove()

    # Finalize means and rates
    results = {}
    for reg_name, data in stats.items():
        count = data["count"]
        if count == 0: continue
        results[reg_name] = {
            "mean": [s / count for s in data["sum"]],
            "rate": [a / count for a in data["active"]]
        }
    return results

def main():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # Use BF16 instead of INT8 for Windows compatibility if bnb fails
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE, low_cpu_mem_usage=True
    )
    layers = model.model.language_model.layers

    item_a, item_b = load_data()
    print("Processing Case A (Baseline/Fail)...")
    ids_a, regions_a = build_sequences(tokenizer, item_a)
    stats_a = capture_stats(model, layers, ids_a, regions_a)

    print("Processing Case B (Teacher/Pass)...")
    ids_b, regions_b = build_sequences(tokenizer, item_b)
    stats_b = capture_stats(model, layers, ids_b, regions_b)

    # Analysis & Comparison
    report = []
    report.append("QA Neuron Difference Analysis Report")
    report.append(f"Data: {DATA_PATH.name}")
    report.append("-" * 40)

    comparison = {}
    for stage in ["think", "output"]:
        if stage not in stats_a or stage not in stats_b: continue

        diff_means = [mb - ma for ma, mb in zip(stats_a[stage]["mean"], stats_b[stage]["mean"])]

        # Find top neurons
        all_diffs = []
        for li, layer_diff in enumerate(diff_means):
            for ni in range(layer_diff.shape[0]):
                val = layer_diff[ni].item()
                if abs(val) > 0.2: # Significant diff
                    all_diffs.append((abs(val), val, li, ni, stage))

        all_diffs.sort(key=lambda x: x[0], reverse=True)
        comparison[stage] = all_diffs

        report.append(f"\nTop 20 {stage.upper()} stage neurons by |diff|:")
        report.append(f"{'Rank':>4} {'Layer#Nr':>12} {'Diff':>10} {'Rate_A':>8} {'Rate_B':>8}")
        for i, (av, v, li, ni, _) in enumerate(all_diffs[:20]):
            ra = stats_a[stage]["rate"][li][ni].item()
            rb = stats_b[stage]["rate"][li][ni].item()
            report.append(f"{i+1:>4} L{li:02d}#{ni:<6d} {v:>+10.4f} {ra:>8.3f} {rb:>8.3f}")

    # Save Results
    output_text = "\n".join(report)
    print(output_text)
    (OUT_DIR / "analysis_report.txt").write_text(output_text, encoding="utf-8")

    torch.save({
        "stats_a": stats_a,
        "stats_b": stats_b,
        "comparison": comparison
    }, OUT_DIR / "diff_stats.pt")

    print(f"\nResults saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
