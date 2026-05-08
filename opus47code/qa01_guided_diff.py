"""
QA01 guided-vs-native QA differential.

Going back to the original multi_qa_probe.py design, but with two changes:
  - B branch's user prompt has FORMAT_GUIDE appended (8-point structural req)
  - B branch's assistant content = Opus answer
  - A branch = pure Gemma baseline: no guide, Gemma's own free generation

So both content AND prompt shift in the same "structured-Opus-style"
direction simultaneously. The previous (qa01_format_diff.py) experiment
showed that teacher-forcing Opus on top of an unguided prompt drowns out
the format-guide signal — this version removes that confound by letting A
be Gemma's native (unguided + native answer).

Sequences:
  Seq A = user                + gemma_native_answer    (teacher-forced)
  Seq B = user + FORMAT_GUIDE + opus_answer            (teacher-forced)

Capture: mlp.down_proj input (6144-d SwiGLU activation) at every layer.
Mean over each sequence's own assistant region (lengths differ — that's OK,
we're comparing distributions, not per-position).

Output: outputs/opus47_guided_diff/
"""
from __future__ import annotations

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
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "claudeopusQA01.json"
BASELINE_TXT = ROOT / "outputs" / "qa01_compare" / "baseline.txt"
NEURONS_JSON = ROOT / "chat" / "neurons.json"
OUT_DIR = ROOT / "outputs" / "opus47_guided_diff"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
TOP_K = 50
MAX_NEW_TOKENS = 900   # only used if we have to generate Gemma native fresh

FORMAT_GUIDE = (
    "\n\n回答的格式需求:\n"
    "1.需要先以你的感受出发,并且不能说你是一个ai的被动回应.\n"
    "2.需要先讲 \"故事做的好的地方\" 并引用原文 和 核心隐喻\n"
    "3.直接引用原文中比较好的句子,并说为什么好,不需要用一个子标题来写\"原文内容\","
    "可以直接引用原句 然后 说解析.\n"
    "4.发掘内容中的文学分镜,解释用文字做的镜头语言\n"
    "5.需要再讲第二部分 需要改进的地方\n"
    "6.需要指出叙事节奏问题\n"
    "7.需要指出可能的质感脱节\n"
    "8.给出整体评价,并具备深度"
)


def extract_answer_from_baseline_txt(path: Path) -> str | None:
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8")
    s = txt.find("ANSWER (response)")
    if s < 0:
        return None
    s = txt.find("\n", s) + 1
    e = txt.find("================ META")
    if e < 0:
        e = len(txt)
    ans = txt[s:e].strip()
    while ans.startswith("="):
        ans = ans.lstrip("=").lstrip("\n").strip()
    return ans


def gen_native_answer(model, tokenizer, user_text):
    msgs = [{"role": "user", "content": user_text}]
    enc = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    ids = enc["input_ids"].to(DEVICE)
    print(f"  [gen] prompt={ids.shape[1]}t  generating up to {MAX_NEW_TOKENS}t ...",
          flush=True)
    t0 = time.time()
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.generate(
            ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id, use_cache=True,
        )
    gen_ids = out[0, ids.shape[1]:].cpu().tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"  [gen] {len(gen_ids)} tokens in {time.time()-t0:.1f}s, "
          f"{len(text)}c", flush=True)
    return text


def run_one(model, layers, tokenizer, user_text, assistant_text, label):
    n_layers = len(layers)

    msgs_user = [{"role": "user", "content": user_text}]
    user_enc = tokenizer.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    assistant_start = user_enc["input_ids"].shape[1]

    msgs_full = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    enc = tokenizer.apply_chat_template(
        msgs_full, tokenize=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    )
    ids = enc["input_ids"]
    T = ids.shape[1]
    n_assistant = T - assistant_start
    print(f"\n[{label}] user={len(user_text)}c  ans={len(assistant_text)}c  "
          f"T={T}  assistant_start={assistant_start}  assistant_tokens={n_assistant}",
          flush=True)

    layer_means = [None] * n_layers

    def make_pre(idx):
        def pre(module, inputs):
            x = inputs[0]
            seg = x[0, assistant_start:].float()
            layer_means[idx] = seg.mean(dim=0).cpu()
        return pre

    handles = [layers[L].mlp.down_proj.register_forward_pre_hook(make_pre(L))
               for L in range(n_layers)]
    try:
        t0 = time.time()
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            model(input_ids=ids.to(DEVICE), use_cache=False)
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"  forward done in {dt:.1f}s  peak={peak:.2f}GB", flush=True)
        torch.cuda.reset_peak_memory_stats(0)
    finally:
        for h in handles:
            h.remove()

    torch.cuda.empty_cache()
    return {
        "T": T, "assistant_start": assistant_start,
        "n_assistant": n_assistant, "layer_means": layer_means,
    }


def main():
    print(f"[load] {MODEL_PATH}  (BF16)", flush=True)
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
    n_layers = len(layers)
    print(f"  layers={n_layers}  alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB",
          flush=True)

    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = qa["input"]
    opus_answer = qa["output"]

    # ---- A side: Gemma native answer (no guide, free gen) ----
    gemma_native = extract_answer_from_baseline_txt(BASELINE_TXT)
    if gemma_native:
        print(f"[A.ans] reuse from {BASELINE_TXT.name} ({len(gemma_native)}c)",
              flush=True)
        print(f"        preview: {gemma_native[:120]!r}", flush=True)
    else:
        print(f"[A.ans] no baseline.txt found, generating fresh ...", flush=True)
        gemma_native = gen_native_answer(model, tokenizer, user_text)
        (OUT_DIR / "gemma_native.txt").write_text(gemma_native, encoding="utf-8")

    print(f"[B.ans] use Opus answer ({len(opus_answer)}c)", flush=True)
    print(f"[data] FORMAT_GUIDE={len(FORMAT_GUIDE)}c", flush=True)

    # ---- A: user (no guide) + gemma_native ----
    res_A = run_one(model, layers, tokenizer,
                    user_text, gemma_native, "A:native-no-guide")
    # ---- B: user + FORMAT_GUIDE + opus_answer ----
    res_B = run_one(model, layers, tokenizer,
                    user_text + FORMAT_GUIDE, opus_answer, "B:guided-opus")

    print(f"\n[align] A.assistant_tokens={res_A['n_assistant']}  "
          f"B.assistant_tokens={res_B['n_assistant']}  "
          f"(lengths differ — comparing distribution means)", flush=True)

    # ---- per-layer separation ----
    print("\n" + "=" * 78)
    print(f"  Per-layer guided-vs-native separation (Δ = mean_B - mean_A)")
    print("=" * 78)
    sep_table = []
    diffs = []
    for L in range(n_layers):
        d = res_B["layer_means"][L] - res_A["layer_means"][L]
        diffs.append(d)
        max_abs = d.abs().max().item()
        mean_abs = d.abs().mean().item()
        std_d = d.std().item()
        n_05 = (d.abs() > 0.5).sum().item()
        n_10 = (d.abs() > 1.0).sum().item()
        n_20 = (d.abs() > 2.0).sum().item()
        sep_table.append((L, max_abs, mean_abs, std_d, n_05, n_10, n_20))
        print(f"  L{L:>2}: max|Δ|={max_abs:>6.3f}  mean|Δ|={mean_abs:.4f}  "
              f"std={std_d:.3f}  n>0.5={n_05:>4d}  n>1.0={n_10:>4d}  n>2.0={n_20:>3d}",
              flush=True)

    best_L = max(sep_table, key=lambda r: r[1])[0]
    print(f"\n  → strongest max|Δ| at L{best_L}", flush=True)

    # ---- top-K per layer ----
    all_results = {}
    print("\n" + "=" * 78)
    print(f"  Per-layer top-15 by |Δ| (B-A; +=guided-Opus, -=native-Gemma)")
    print("=" * 78)
    for L in range(n_layers):
        d = diffs[L]
        topk = d.abs().topk(TOP_K)
        results = []
        for idx, val_abs in zip(topk.indices.tolist(), topk.values.tolist()):
            results.append({
                "neuron": idx,
                "diff": d[idx].item(),
                "B_mean": res_B["layer_means"][L][idx].item(),
                "A_mean": res_A["layer_means"][L][idx].item(),
            })
        all_results[L] = results
        if sep_table[L][1] >= 0.5:
            print(f"\n  --- L{L} top-15  (max|Δ|={sep_table[L][1]:.2f}) ---",
                  flush=True)
            for i, r in enumerate(results[:15]):
                arrow = "Opus↑" if r["diff"] > 0 else "Native↑"
                print(f"    {i+1:>2}. L{L}#{r['neuron']:<5d}  "
                      f"Δ={r['diff']:+7.3f}  ({arrow})  "
                      f"B={r['B_mean']:+6.3f}  A={r['A_mean']:+6.3f}",
                      flush=True)

    # ---- inventory cross-ref ----
    inventory = {}
    if NEURONS_JSON.exists():
        nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
        inventory = {(n["layer"], n["index"]): n for n in nj["known_neurons"]}
    print("\n" + "=" * 78)
    print(f"  Inventory matches in per-layer top-{TOP_K}")
    print("=" * 78)
    total_matches = 0
    for L in range(n_layers):
        matches = []
        for r in all_results[L]:
            key = (L, r["neuron"])
            if key in inventory:
                matches.append((r, inventory[key]))
        if matches:
            print(f"\n  L{L:>2}: {len(matches)} matches", flush=True)
            for r, inv in matches:
                arrow = "Opus↑" if r["diff"] > 0 else "Native↑"
                print(f"    L{L}#{r['neuron']:<5d}  Δ={r['diff']:+7.3f} ({arrow})  "
                      f"[{inv.get('tier','?')}, gain={inv.get('default_gain',0):+.1f}, "
                      f"region={inv.get('region','?')}]  {inv.get('label','')}",
                      flush=True)
            total_matches += len(matches)
    print(f"\n  total inventory matches in top-{TOP_K}: {total_matches}", flush=True)

    # ---- save ----
    torch.save({
        "config": {"TOP_K": TOP_K, "FORMAT_GUIDE": FORMAT_GUIDE},
        "res_A": res_A, "res_B": res_B,
        "diffs": diffs, "all_results": all_results,
        "sep_table": sep_table, "best_L": best_L,
        "gemma_native_preview": gemma_native[:500],
    }, OUT_DIR / "raw.pt")

    with open(OUT_DIR / "top_neurons.txt", "w", encoding="utf-8") as f:
        f.write("================ QA01 Guided-vs-Native QA Differential ================\n")
        f.write("Design (back to multi_qa_probe style):\n")
        f.write(f"  A: user                + gemma_native_answer  (teacher-forced)\n")
        f.write(f"  B: user + FORMAT_GUIDE + opus_answer          (teacher-forced)\n")
        f.write(f"  Δ = mean_B(activation) - mean_A(activation), per neuron, "
                f"averaged over each respective assistant region.\n\n")
        f.write(f"FORMAT_GUIDE ({len(FORMAT_GUIDE)} chars):\n{FORMAT_GUIDE}\n\n")
        f.write(f"Sequence shapes:\n")
        f.write(f"  A: T={res_A['T']}  assistant_start={res_A['assistant_start']}  "
                f"n_assistant={res_A['n_assistant']}  ans_chars={len(gemma_native)}\n")
        f.write(f"  B: T={res_B['T']}  assistant_start={res_B['assistant_start']}  "
                f"n_assistant={res_B['n_assistant']}  ans_chars={len(opus_answer)}\n\n")

        f.write("================ Per-layer separation ================\n")
        f.write("  L  | max|Δ|   mean|Δ|   std    >0.5    >1.0   >2.0\n")
        for L, max_abs, mean_abs, std_d, n_05, n_10, n_20 in sep_table:
            f.write(f"  L{L:>2}  | {max_abs:>6.3f}   {mean_abs:>7.4f}  "
                    f"{std_d:>5.3f}   {n_05:>4d}   {n_10:>4d}   {n_20:>3d}\n")
        f.write(f"\n  → strongest max|Δ| at L{best_L}\n")

        for L in range(n_layers):
            f.write(f"\n\n================ L{L} top-{TOP_K} ================\n")
            f.write("  rank  L#nrn          Δ          B_mean      A_mean      sign\n")
            for i, r in enumerate(all_results[L]):
                sign = "Opus↑" if r["diff"] > 0 else "Native↑"
                f.write(f"  {i+1:>4}  L{L}#{r['neuron']:<5d}    "
                        f"{r['diff']:+8.3f}    "
                        f"{r['B_mean']:+8.3f}    "
                        f"{r['A_mean']:+8.3f}   {sign}\n")

        f.write(f"\n\n================ Inventory matches per layer (top-{TOP_K}) ================\n")
        if not inventory:
            f.write("  (no neurons.json found)\n")
        else:
            for L in range(n_layers):
                matches = [(r, inventory[(L, r["neuron"])]) for r in all_results[L]
                           if (L, r["neuron"]) in inventory]
                if matches:
                    f.write(f"\n  L{L:>2}: {len(matches)} match\n")
                    for r, inv in matches:
                        sign = "Opus↑" if r["diff"] > 0 else "Native↑"
                        f.write(f"    L{L}#{r['neuron']:<5d}  Δ={r['diff']:+7.3f} ({sign})  "
                                f"[{inv.get('tier','?')}, "
                                f"gain={inv.get('default_gain',0):+.1f}, "
                                f"region={inv.get('region','?')}]  "
                                f"{inv.get('label','')}\n")

    print(f"\n[save] {OUT_DIR / 'raw.pt'}", flush=True)
    print(f"[save] {OUT_DIR / 'top_neurons.txt'}", flush=True)


if __name__ == "__main__":
    main()
