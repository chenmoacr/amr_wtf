"""
Attention anchor probe — verify the cite→depth→fade hypothesis.

Hypothesis (user, 2026-05-03):
  dl_pushD_v1 didn't learn "deep critique capability". It learned a
  citation strategy — when generating, it cites original text more
  often, which anchors attention to the input prompt for ~5-15 tokens
  ("buff window"), during which it produces specific judgment, and after
  which it fades back to vapid generic critic.

Test:
  Replay dl_pushD_v1's QA01 generation as a teacher-forced forward.
  For each generated position t, compute:
    input_attention_ratio[L][t] = sum(attn_weights[L, head, t, :prompt_len])
                                / sum(attn_weights[L, head, t, :])
  averaged across heads.

  Annotate generated tokens with:
    - is_cite: token+context is a substring of the input prompt
    - is_buzz: token is a known generic-buzzword
    - is_youli: contains "有力" (the homo-meme repetition tell)

Then look at:
  - Does ratio spike at citations and decay after?
  - Does buzzword position correspond to LOW ratio?
  - Layer differences: which layers show this anchor pattern strongest?

Memory: hook on each self_attn captures attn weights, computes our scalar
per query position, immediately replaces attn output with None to free GPU.

Output: outputs/opus47_attn_probe/
  per_token_ratios.csv     row per generated position, cols per layer
  annotated_tokens.txt     human-readable per-position breakdown
  summary.txt              statistics + correlation
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from peft import PeftModel

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "claudeopusQA01.json"
DL_PUSH_TXT = ROOT / "outputs" / "opus47_crutch_depth_sweep" / "infer_dl_pushD_v1.txt"
OUT_DIR = ROOT / "outputs" / "opus47_attn_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
MAX_PROMPT_TOKENS = 1500

ADAPTERS = {
    "A_off": ROOT / "outputs/opus47_crutch_off/lora_adapter",
    "C":     ROOT / "outputs/opus47_crutch_C/lora_adapter",
    "D":     ROOT / "outputs/opus47_crutch_D/lora_adapter",
}
DL_PUSH_WEIGHTS = (0.15, 0.30, 0.55)   # dl_pushD_v1

BUZZWORDS = ["宏大", "深刻", "晦涩", "存在主义", "象征意义", "哲学思辨",
             "意境", "氛围", "层次", "复杂", "微妙", "深远", "细腻",
             "厚重", "饱满", "克制"]


def truncate_user_text(user_text, tokenizer, max_tokens):
    msgs = [{"role": "user", "content": user_text}]
    full_len = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )["input_ids"].shape[1]
    if full_len <= max_tokens:
        return user_text
    lo, hi = 0, len(user_text)
    keep = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = user_text[-mid:]
        msgs = [{"role": "user", "content": candidate}]
        tlen = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )["input_ids"].shape[1]
        if tlen <= max_tokens:
            lo = mid; keep = mid
        else:
            hi = mid - 1
    return user_text[-keep:]


def extract_dl_push_answer(path):
    txt = path.read_text(encoding="utf-8")
    s = txt.find("================ ANSWER")
    s = txt.find("\n", s) + 1
    e = txt.find("================ META")
    return txt[s:e].strip()


def main():
    print(f"[load] {MODEL_PATH} (eager attention)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
        attn_implementation="eager",   # eager so attn weights accessible
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(base.model, attr):
            setattr(base.model, attr, None)

    # ---- load adapters and weight-merge (dl_pushD_v1) ----
    first = list(ADAPTERS.keys())[0]
    model = PeftModel.from_pretrained(base, str(ADAPTERS[first]),
                                       adapter_name=first)
    for name, path in ADAPTERS.items():
        if name == first:
            continue
        model.load_adapter(str(path), adapter_name=name)
    wA, wC, wD = DL_PUSH_WEIGHTS
    model.add_weighted_adapter(
        adapters=["A_off", "C", "D"], weights=[wA, wC, wD],
        adapter_name="dl_push", combination_type="linear",
    )
    model.set_adapter("dl_push")
    model.eval()
    print(f"[adapter] dl_pushD_v1 loaded (A={wA}/C={wC}/D={wD})", flush=True)

    inner = model.base_model.model.model.language_model
    layers = inner.layers
    n_layers = len(layers)

    # ---- prepare full sequence: prompt + dl_pushD_v1 generated answer ----
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    user_text = truncate_user_text(qa["input"], tokenizer, MAX_PROMPT_TOKENS)
    answer_text = extract_dl_push_answer(DL_PUSH_TXT)
    print(f"[input] user={len(user_text)}c  answer={len(answer_text)}c", flush=True)

    msgs_user = [{"role": "user", "content": user_text}]
    user_enc = tokenizer.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    prompt_len = user_enc["input_ids"].shape[1]

    msgs_full = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer_text},
    ]
    enc = tokenizer.apply_chat_template(
        msgs_full, tokenize=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    T = input_ids.shape[1]
    n_gen = T - prompt_len
    print(f"[seq] T={T}  prompt_len={prompt_len}  n_gen={n_gen}", flush=True)

    # decode each token for annotation
    target_ids = input_ids[0, prompt_len:].cpu().tolist()
    target_strs = [tokenizer.decode([t], skip_special_tokens=False) for t in target_ids]

    # ---- install hooks: capture attn ratio + nullify weights to free GPU ----
    # ratios_per_layer[L] = numpy array [T] of input_attention_ratio for each query position
    ratios_per_layer = {}
    prompt_len_ref = [prompt_len]   # mutable for closure

    def make_hook(L):
        def hook(module, inputs, outputs):
            if not (isinstance(outputs, tuple) and len(outputs) >= 2):
                return outputs
            aw = outputs[1]
            if not isinstance(aw, torch.Tensor):
                return outputs
            # aw: [B=1, H, T_q, T_k]
            with torch.no_grad():
                a_mean = aw.float().mean(dim=1)        # [1, T, T]
                pl = prompt_len_ref[0]
                to_input = a_mean[..., :pl].sum(dim=-1)   # [1, T]
                total = a_mean.sum(dim=-1)                 # [1, T]
                ratio = (to_input / total.clamp(min=1e-9)).detach().cpu().numpy()
                ratios_per_layer[L] = ratio[0]
            # replace attn weights with None to free GPU memory
            return (outputs[0], None) + outputs[2:]
        return hook

    handles = []
    for L in range(n_layers):
        h = layers[L].self_attn.register_forward_hook(make_hook(L))
        handles.append(h)

    # ---- forward (output_attentions=True so eager computes weights) ----
    print(f"\n[forward] running with output_attentions=True ...", flush=True)
    try:
        import time
        t0 = time.time()
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"  done in {time.time()-t0:.1f}s  peak={peak:.2f}GB", flush=True)
    finally:
        for h in handles:
            h.remove()

    if not ratios_per_layer:
        print("[fatal] no ratios captured", flush=True)
        return
    print(f"[captured] {len(ratios_per_layer)} layers", flush=True)

    # ---- detect citation tokens (char-level common-substring mask) ----
    # Strategy: for every starting index i in gen_text, find the longest L
    # such that gen_text[i:i+L] appears in user_text. Mark all those chars
    # as "quoted". Then any token whose chars include any quoted char is
    # is_cite=True.
    full_gen = "".join(target_strs)
    char_pos = []
    cum = 0
    for s in target_strs:
        char_pos.append(cum)
        cum += len(s)
    n_chars = len(full_gen)

    MIN_CITE_LEN = 4
    is_quoted_char = [False] * n_chars
    i = 0
    while i < n_chars - MIN_CITE_LEN + 1:
        if full_gen[i:i + MIN_CITE_LEN] not in user_text:
            i += 1
            continue
        L = MIN_CITE_LEN
        while i + L <= n_chars and full_gen[i:i + L] in user_text:
            L += 1
        L -= 1
        for j in range(i, i + L):
            is_quoted_char[j] = True
        i += L  # skip ahead past this match (don't double-count; keep simple)

    is_cite = []
    for ti in range(len(target_strs)):
        c = char_pos[ti]
        L = len(target_strs[ti])
        is_cite.append(any(is_quoted_char[c + k] for k in range(L)) if L > 0
                       else False)

    # ---- detect buzzword tokens (char-level mask) ----
    is_buzz_char = [False] * n_chars
    for bw in BUZZWORDS:
        start = 0
        while True:
            idx = full_gen.find(bw, start)
            if idx < 0:
                break
            for j in range(idx, idx + len(bw)):
                is_buzz_char[j] = True
            start = idx + 1
    is_buzz = []
    for ti in range(len(target_strs)):
        c = char_pos[ti]
        L = len(target_strs[ti])
        is_buzz.append(any(is_buzz_char[c + k] for k in range(L)) if L > 0
                       else False)

    is_youli = ["有力" in s for s in target_strs]

    # report
    print(f"[detect] cite_tokens={sum(is_cite)}  buzz_tokens={sum(is_buzz)}  "
          f"youli={sum(is_youli)}", flush=True)

    # ---- compute aggregates ----
    import numpy as np
    R = np.stack([ratios_per_layer[L] for L in range(n_layers)
                  if L in ratios_per_layer], axis=0)   # [n_layers, T]
    R_gen = R[:, prompt_len:]                            # [n_layers, n_gen]

    mean_ratio = R_gen.mean(axis=0)   # [n_gen]
    # specific layer cuts
    def get_layer(L):
        if L in ratios_per_layer:
            return ratios_per_layer[L][prompt_len:]
        return np.zeros(n_gen)

    L27_r = get_layer(27)
    L34_r = get_layer(34)
    L13_r = get_layer(13)
    L9_r  = get_layer(9)

    # ---- save CSV ----
    csv_path = OUT_DIR / "per_token_ratios.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("pos,token,mean_ratio,L9,L13,L27,L34,is_cite,is_buzz,is_youli\n")
        for i in range(n_gen):
            tok = target_strs[i].replace("\n", "\\n").replace(",", "，")
            f.write(f"{i},{tok!r},{mean_ratio[i]:.4f},"
                    f"{L9_r[i]:.4f},{L13_r[i]:.4f},"
                    f"{L27_r[i]:.4f},{L34_r[i]:.4f},"
                    f"{int(is_cite[i])},{int(is_buzz[i])},"
                    f"{int(is_youli[i])}\n")
    print(f"[save] {csv_path}", flush=True)

    # also save raw tensors for downstream analysis without re-forward
    torch.save({
        "T": T, "prompt_len": prompt_len, "n_gen": n_gen,
        "target_strs": target_strs, "target_ids": target_ids,
        "user_text": user_text, "answer_text": answer_text,
        "ratios_per_layer": {L: ratios_per_layer[L] for L in ratios_per_layer},
        "is_cite": is_cite, "is_buzz": is_buzz, "is_youli": is_youli,
    }, OUT_DIR / "raw.pt")
    print(f"[save] {OUT_DIR / 'raw.pt'}", flush=True)

    # ---- annotated text ----
    ann_path = OUT_DIR / "annotated_tokens.txt"
    with open(ann_path, "w", encoding="utf-8") as f:
        f.write(f"# Per-token attention to input prompt — dl_pushD_v1 generation on QA01\n")
        f.write(f"# columns: pos | mean_input_attn_ratio | L27 | L34 | flags | token\n")
        f.write(f"# flags: ★cite (chunk found in input)  ✗buzz (vapid buzzword)  "
                f"※youli (有力 repetition)\n\n")
        for i in range(n_gen):
            flags = []
            if is_cite[i]: flags.append("★cite")
            if is_buzz[i]: flags.append("✗buzz")
            if is_youli[i]: flags.append("※youli")
            flag_str = " ".join(flags) if flags else "    "
            tok = target_strs[i].replace("\n", "\\n")
            f.write(f"  {i:>4}  m={mean_ratio[i]:.3f}  L27={L27_r[i]:.3f}  "
                    f"L34={L34_r[i]:.3f}  {flag_str:<25}  '{tok}'\n")
    print(f"[save] {ann_path}", flush=True)

    # ---- statistical summary ----
    summary_path = OUT_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("================ Attention Anchor Probe — dl_pushD_v1 on QA01 ================\n\n")
        f.write(f"Sequence: T={T}  prompt_len={prompt_len}  n_generated={n_gen}\n")
        f.write(f"Layers captured: {len(ratios_per_layer)}/{n_layers}\n\n")

        n_cite = sum(is_cite)
        n_buzz = sum(is_buzz)
        n_youli = sum(is_youli)
        f.write(f"Token classification:\n")
        f.write(f"  is_cite (chunk found in input): {n_cite}/{n_gen}  "
                f"({n_cite/n_gen*100:.1f}%)\n")
        f.write(f"  is_buzz (buzzword):             {n_buzz}/{n_gen}  "
                f"({n_buzz/n_gen*100:.1f}%)\n")
        f.write(f"  is_youli ('有力'):              {n_youli}/{n_gen}\n\n")

        # mean ratio for each token class
        cite_idx = np.array([i for i in range(n_gen) if is_cite[i]])
        buzz_idx = np.array([i for i in range(n_gen) if is_buzz[i]])
        other_idx = np.array([i for i in range(n_gen)
                              if not is_cite[i] and not is_buzz[i]])

        def class_stats(idx, label):
            if len(idx) == 0:
                f.write(f"  {label:<20}  N=0\n")
                return None
            mean_m = mean_ratio[idx].mean()
            mean_27 = L27_r[idx].mean() if 27 in ratios_per_layer else float("nan")
            mean_34 = L34_r[idx].mean() if 34 in ratios_per_layer else float("nan")
            f.write(f"  {label:<20}  N={len(idx):>4}  "
                    f"mean_ratio={mean_m:.3f}  L27={mean_27:.3f}  "
                    f"L34={mean_34:.3f}\n")
            return mean_m

        f.write(f"================ Mean input-attention by token class ================\n")
        m_cite = class_stats(cite_idx, "★cite tokens")
        m_buzz = class_stats(buzz_idx, "✗buzz tokens")
        m_other = class_stats(other_idx, "  other tokens")

        # critical comparison
        if m_cite is not None and m_buzz is not None:
            f.write(f"\n  Δ(cite − buzz) = {m_cite - m_buzz:+.3f}\n")
            if m_cite > m_buzz:
                f.write("  → cite tokens DO have higher input attention. "
                        "Anchor hypothesis: SUPPORTED on average.\n")
            else:
                f.write("  → cite tokens do NOT have higher input attention. "
                        "Anchor hypothesis: not supported.\n")

        # ---- decay test: does ratio decay AFTER cite tokens? ----
        # for each cite position, look at next 1-15 positions' ratio
        f.write(f"\n================ Post-cite decay analysis ================\n")
        WIN = 15
        if len(cite_idx) > 0:
            decay_curve = np.zeros(WIN + 1)   # offset 0..WIN
            counts = np.zeros(WIN + 1)
            for c in cite_idx:
                for k in range(WIN + 1):
                    if c + k < n_gen:
                        decay_curve[k] += mean_ratio[c + k]
                        counts[k] += 1
            decay_curve = decay_curve / np.maximum(counts, 1)
            f.write(f"  offset_from_cite | mean_ratio | n_samples\n")
            for k in range(WIN + 1):
                f.write(f"    +{k:>2}            | {decay_curve[k]:.4f}   "
                        f"| {int(counts[k])}\n")
            # is there a monotonic decay?
            decay_amount = decay_curve[0] - decay_curve[-1]
            f.write(f"\n  decay_amount(0→{WIN}) = {decay_amount:+.4f}\n")
            if decay_amount > 0.01:
                f.write("  → Ratio decays after cites. Buff-window hypothesis: "
                        "SUPPORTED.\n")
            else:
                f.write("  → No clear decay. Buff-window hypothesis: not supported.\n")

        # ---- per-layer summary: which layer shows the strongest cite-vs-buzz gap? ----
        f.write(f"\n================ Per-layer cite-vs-buzz attention gap ================\n")
        f.write("  layer | cite.mean | buzz.mean | gap (cite-buzz)\n")
        gaps = []
        for L in range(n_layers):
            if L not in ratios_per_layer:
                continue
            r = ratios_per_layer[L][prompt_len:]
            c_m = r[cite_idx].mean() if len(cite_idx) > 0 else 0
            b_m = r[buzz_idx].mean() if len(buzz_idx) > 0 else 0
            gap = c_m - b_m
            gaps.append((L, c_m, b_m, gap))
        gaps.sort(key=lambda x: x[3], reverse=True)
        f.write("  Top-15 layers by gap (most anchor-driven):\n")
        for L, c_m, b_m, gap in gaps[:15]:
            f.write(f"  L{L:>2}  | {c_m:>9.3f} | {b_m:>9.3f} | {gap:+.3f}\n")
        f.write("  Bottom-5 layers (least anchor-driven):\n")
        for L, c_m, b_m, gap in gaps[-5:]:
            f.write(f"  L{L:>2}  | {c_m:>9.3f} | {b_m:>9.3f} | {gap:+.3f}\n")

    print(f"[save] {summary_path}", flush=True)
    print(f"\n[done] outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
