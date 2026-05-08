"""
QA01 1-shot SFT — INTENT-ONLY via gradient-based token weighting.

Builds on qa01_grad_probe.py output (outputs/opus47_grad_probe/raw.pt).

Token classification:
  For each Opus output position t, score(t) = init_loss(t) - final_loss(t)
  from the previous probe run.
    score > INTENT_THRESH    →  weight = 1.0   (intent / structural)
    MIXED_THRESH <= score    →  weight = MIXED_WEIGHT
    score < MIXED_THRESH     →  weight = 0.0   (pure knowledge — masked)

  Plus: positions whose token belongs to a glue_sentences string get
  weight × GLUE_BOOST (the section markers like '**故事做得好的地方**' are
  the cleanest intent signal; we want them strongly).

Result: gradient flows only through tokens the model can actually learn
in 1-shot (= intent), and skips the story-specific knowledge tokens that
1-shot SFT cannot teach.

Output: outputs/opus47_intent_sft/{raw.pt, summary.txt, run.log}
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
QA_PATH = ROOT / "data" / "claudeopusQA01.json"
NEURONS_JSON = ROOT / "chat" / "neurons.json"
PROBE_PT = ROOT / "outputs" / "opus47_grad_probe" / "raw.pt"
OUT_DIR = ROOT / "outputs" / "opus47_intent_sft"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
N_STEPS = 20
LR = 1e-4
LORA_R = 8
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
TOP_K_NEURONS = 50
MAX_PROMPT_TOKENS = 1800

# Token weighting from probe Δloss
INTENT_THRESH = 5.0     # score > this: pure intent  → weight 1.0
MIXED_THRESH  = 2.0     # score in [MIXED, INTENT]:  → weight MIXED_WEIGHT
MIXED_WEIGHT  = 0.4
GLUE_BOOST    = 2.0     # multiply weight on glue_sentences token positions


def parse_lora_param_name(name: str):
    parts = name.split(".")
    layer_idx = None
    if "layers" in parts:
        i = parts.index("layers")
        try:
            layer_idx = int(parts[i + 1])
        except (ValueError, IndexError):
            pass
    module = None
    for m in TARGET_MODULES:
        if m in parts:
            module = m
            break
    ab = "A" if "lora_A" in name else ("B" if "lora_B" in name else None)
    return layer_idx, module, ab


def find_glue_token_positions(target_strs, glue_substrings):
    """Map glue substrings to a set of token-position indices in target_strs."""
    full = "".join(target_strs)
    # build cumulative-char-end array
    ends = []
    cum = 0
    for s in target_strs:
        cum += len(s)
        ends.append(cum)

    glue_positions = set()
    for sub in glue_substrings:
        sub = sub.strip()
        if not sub:
            continue
        start = 0
        while True:
            idx = full.find(sub, start)
            if idx < 0:
                break
            char_end = idx + len(sub)
            # find tokens whose [tok_start, tok_end) overlaps [idx, char_end)
            for i, end in enumerate(ends):
                tok_start = end - len(target_strs[i])
                if tok_start < char_end and end > idx:
                    glue_positions.add(i)
            start = idx + 1
    return glue_positions


def main():
    print(f"[load probe] {PROBE_PT}", flush=True)
    if not PROBE_PT.exists():
        print(f"[fatal] probe file not found. Run qa01_grad_probe.py first.",
              flush=True)
        return
    probe = torch.load(PROBE_PT, weights_only=False)

    init_loss = probe["history"][0]["per_tok_loss"]
    final_loss = probe["history"][-1]["per_tok_loss"]
    target_strs_probe = probe["target_strs"]
    n_probe = len(init_loss)
    print(f"[probe] loaded: n_target_tokens={n_probe}  "
          f"steps={len(probe['history'])}", flush=True)

    # compute per-token score = how much it learned
    scores = [init_loss[i] - final_loss[i] for i in range(n_probe)]

    # ---- bucket assignment ----
    intent_idx, mixed_idx, knowledge_idx = [], [], []
    base_weights = []
    for i, s in enumerate(scores):
        if s > INTENT_THRESH:
            intent_idx.append(i)
            base_weights.append(1.0)
        elif s >= MIXED_THRESH:
            mixed_idx.append(i)
            base_weights.append(MIXED_WEIGHT)
        else:
            knowledge_idx.append(i)
            base_weights.append(0.0)

    print(f"[bucket] intent={len(intent_idx)} (w=1.0)  "
          f"mixed={len(mixed_idx)} (w={MIXED_WEIGHT})  "
          f"knowledge={len(knowledge_idx)} (w=0.0, masked)", flush=True)

    # ---- glue identification ----
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    glue_substrings = qa.get("glue_sentences", [])
    print(f"[glue] {len(glue_substrings)} glue substrings", flush=True)

    glue_positions = find_glue_token_positions(target_strs_probe, glue_substrings)
    print(f"[glue] mapped to {len(glue_positions)} token positions", flush=True)

    # apply glue boost — but never resurrect knowledge tokens that were masked
    final_weights = list(base_weights)
    glue_boosted = 0
    for i in glue_positions:
        if final_weights[i] > 0:
            final_weights[i] *= GLUE_BOOST
            glue_boosted += 1
    print(f"[glue] boosted {glue_boosted} token weights by ×{GLUE_BOOST}",
          flush=True)

    n_active = sum(1 for w in final_weights if w > 0)
    total_weight = sum(final_weights)
    print(f"[weights] active={n_active}/{n_probe}  "
          f"sum={total_weight:.2f}  mean(over active)="
          f"{total_weight/max(1,n_active):.3f}", flush=True)

    # ---- load model + LoRA + grad checkpointing (same as probe) ----
    print(f"\n[load] {MODEL_PATH}  (BF16)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, low_cpu_mem_usage=True,
    )
    for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
                 "embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            setattr(model.model, attr, None)

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] trainable={n_trainable/1e6:.2f}M", flush=True)

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    inner_text_model = model.base_model.model.model.language_model
    inner_text_model.config.use_cache = False
    if hasattr(inner_text_model, "gradient_checkpointing_enable"):
        inner_text_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for layer in inner_text_model.layers:
        if hasattr(layer, "gradient_checkpointing"):
            layer.gradient_checkpointing = True
    model.train()
    print(f"[gc] flag={getattr(model, 'is_gradient_checkpointing', None)}  "
          f"training={model.training}", flush=True)

    n_layers = len(inner_text_model.layers)

    # ---- prepare data (identical truncation to probe) ----
    user_text_raw = qa["input"]
    opus_answer = qa["output"]

    msgs_user_full = [{"role": "user", "content": user_text_raw}]
    enc_full = tokenizer.apply_chat_template(
        msgs_user_full, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    full_prompt_len = enc_full["input_ids"].shape[1]
    if full_prompt_len > MAX_PROMPT_TOKENS:
        instr_marker = "阅读和评价虚构故事"
        if instr_marker in user_text_raw:
            instr_idx = user_text_raw.rindex(instr_marker)
            story_part = user_text_raw[:instr_idx]
            tail_part = user_text_raw[instr_idx:]
        else:
            story_part = user_text_raw
            tail_part = ""
        lo, hi = 0, len(story_part)
        keep_chars = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = story_part[-mid:] + tail_part
            test_msgs = [{"role": "user", "content": candidate}]
            tlen = tokenizer.apply_chat_template(
                test_msgs, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True, enable_thinking=False,
            )["input_ids"].shape[1]
            if tlen <= MAX_PROMPT_TOKENS:
                lo = mid
                keep_chars = mid
            else:
                hi = mid - 1
        user_text = story_part[-keep_chars:] + tail_part
        print(f"[truncate] story → last {keep_chars} chars", flush=True)
    else:
        user_text = user_text_raw

    msgs_user = [{"role": "user", "content": user_text}]
    user_enc = tokenizer.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    assistant_start = user_enc["input_ids"].shape[1]

    msgs_full = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": opus_answer},
    ]
    enc = tokenizer.apply_chat_template(
        msgs_full, tokenize=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    T = input_ids.shape[1]
    n_target = T - assistant_start
    print(f"[data] T={T}  assistant_start={assistant_start}  "
          f"n_target={n_target}  (probe had n_target={n_probe})", flush=True)

    if n_target != n_probe:
        print(f"[fatal] target len mismatch — probe and intent run must be "
              f"identical. probe={n_probe}, this run={n_target}", flush=True)
        return

    target_ids = input_ids[0, assistant_start:].cpu().tolist()
    target_strs = [tokenizer.decode([t], skip_special_tokens=False)
                   for t in target_ids]

    K_LOGITS = n_target + 1

    # weight tensor on device
    weight_tensor_full = torch.tensor(final_weights, dtype=torch.float32,
                                       device=DEVICE)
    weight_sum = weight_tensor_full.sum().item()
    print(f"[weights] tensor on device, sum={weight_sum:.3f}", flush=True)

    # ---- optimizer ----
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=N_STEPS, eta_min=1e-6,
    )

    # ---- probe forward kw ----
    logits_kw = None
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                _p = model(input_ids=input_ids, use_cache=False,
                           **{kw: K_LOGITS})
            if _p.logits.shape[1] == K_LOGITS:
                logits_kw = kw
                del _p
                torch.cuda.empty_cache()
                break
            del _p
        except (TypeError, ValueError):
            continue
    print(f"[probe-fwd] using {logits_kw!r}", flush=True)

    # ---- training loop ----
    history = []
    print(f"\n[train] {N_STEPS} steps  LR={LR}  LoRA r={LORA_R}  "
          f"intent-weighted CE", flush=True)

    for step in range(N_STEPS):
        t0 = time.time()
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            forward_kwargs = {"input_ids": input_ids, "use_cache": False}
            if logits_kw is not None:
                forward_kwargs[logits_kw] = K_LOGITS
            out = model(**forward_kwargs)

        if logits_kw is not None:
            shift_logits = out.logits[:, :-1, :]
        else:
            shift_logits = out.logits[:, assistant_start - 1:-1, :]
        shift_labels = input_ids[:, assistant_start:]

        ce_per_tok = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        )

        # WEIGHTED loss
        weighted = ce_per_tok * weight_tensor_full
        loss = weighted.sum() / weight_tensor_full.sum().clamp(min=1.0)

        # also compute unweighted-mean for monitoring
        plain_loss = ce_per_tok.detach().mean().item()
        per_tok_loss = ce_per_tok.detach().cpu().tolist()

        del out, shift_logits
        torch.cuda.empty_cache()

        loss.backward()

        # collect grads
        grad_norms = defaultdict(float)
        per_neuron_down = {}
        for n, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            li, mod, ab = parse_lora_param_name(n)
            if li is None or mod is None or ab is None:
                continue
            gn = p.grad.float().norm().item()
            grad_norms[(li, mod)] += gn
            if mod == "down_proj" and ab == "A":
                per_neuron_down[li] = p.grad.float().norm(dim=0).cpu()

        history.append({
            "step": step,
            "weighted_loss": loss.item(),
            "plain_loss": plain_loss,
            "per_tok_loss": per_tok_loss,
            "grad_norms": dict(grad_norms),
            "per_neuron_down": per_neuron_down,
        })

        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)

        peak = torch.cuda.max_memory_allocated(0) / 1e9
        torch.cuda.reset_peak_memory_stats(0)
        print(f"  step {step+1:>2}/{N_STEPS}  "
              f"weighted_loss={loss.item():.4f}  plain={plain_loss:.4f}  "
              f"{time.time()-t0:.1f}s  peak={peak:.2f}GB", flush=True)
        torch.cuda.empty_cache()

    # ---- aggregate ----
    print("\n[aggregate]", flush=True)
    cum_grad = defaultdict(float)
    for h in history:
        for k, v in h["grad_norms"].items():
            cum_grad[k] += v
    cum_neuron_down = {}
    for h in history:
        for L, t in h["per_neuron_down"].items():
            if L not in cum_neuron_down:
                cum_neuron_down[L] = torch.zeros_like(t)
            cum_neuron_down[L] += t

    # ---- save raw ----
    torch.save({
        "config": {
            "INTENT_THRESH": INTENT_THRESH, "MIXED_THRESH": MIXED_THRESH,
            "MIXED_WEIGHT": MIXED_WEIGHT, "GLUE_BOOST": GLUE_BOOST,
            "N_STEPS": N_STEPS, "LR": LR, "LORA_R": LORA_R,
        },
        "T": T, "assistant_start": assistant_start, "n_target": n_target,
        "target_ids": target_ids, "target_strs": target_strs,
        "scores": scores,
        "base_weights": base_weights,
        "final_weights": final_weights,
        "intent_idx": intent_idx, "mixed_idx": mixed_idx,
        "knowledge_idx": knowledge_idx,
        "glue_positions": sorted(glue_positions),
        "history": history,
        "cum_grad": dict(cum_grad),
        "cum_neuron_down": cum_neuron_down,
    }, OUT_DIR / "raw.pt")
    print(f"  saved raw.pt", flush=True)

    # ---- load inventory for cross-ref ----
    inventory = {}
    if NEURONS_JSON.exists():
        nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
        inventory = {(n["layer"], n["index"]): n for n in nj["known_neurons"]}

    # ---- write summary ----
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("================ QA01 Intent-Only SFT (1-shot, weighted CE) ================\n")
        f.write(f"Setup:\n")
        f.write(f"  thresholds: INTENT>{INTENT_THRESH}  MIXED>={MIXED_THRESH}  "
                f"(mixed_weight={MIXED_WEIGHT})\n")
        f.write(f"  glue_boost: ×{GLUE_BOOST}\n")
        f.write(f"  steps={N_STEPS}  LR={LR}  LoRA r={LORA_R}\n\n")

        f.write(f"================ Token bucket sizes ================\n")
        f.write(f"  intent (w=1.0):    {len(intent_idx):>4} / {n_target}\n")
        f.write(f"  mixed  (w={MIXED_WEIGHT}):    {len(mixed_idx):>4} / {n_target}\n")
        f.write(f"  knowledge (w=0):   {len(knowledge_idx):>4} / {n_target}  ← masked\n")
        f.write(f"  glue boosted:      {sum(1 for i in glue_positions if final_weights[i] > 0):>4}\n\n")

        # ---- show sample tokens in each bucket ----
        f.write("================ Sample intent tokens (top 30 by score) ================\n")
        intent_sorted = sorted(intent_idx, key=lambda i: scores[i], reverse=True)[:30]
        for i in intent_sorted:
            ctx = "".join(target_strs[max(0, i-3):i+1])
            in_glue = "★glue" if i in glue_positions else ""
            f.write(f"  pos={i:>4}  score={scores[i]:.2f}  w={final_weights[i]:.2f}  "
                    f"tok={target_strs[i]!r:>10}  ctx={ctx[-15:]!r}  {in_glue}\n")

        f.write("\n================ Sample knowledge tokens (lowest 30 scores) ================\n")
        knowledge_sorted = sorted(knowledge_idx, key=lambda i: scores[i])[:30]
        for i in knowledge_sorted:
            ctx = "".join(target_strs[max(0, i-3):i+1])
            f.write(f"  pos={i:>4}  score={scores[i]:.2f}  final_loss={final_loss[i]:.2f}  "
                    f"tok={target_strs[i]!r:>10}  ctx={ctx[-15:]!r}\n")

        # ---- training curve ----
        f.write("\n================ Training curve (weighted vs plain) ================\n")
        f.write(f"  step   weighted_loss   plain_mean_loss\n")
        for h in history:
            f.write(f"  {h['step']+1:>3}    {h['weighted_loss']:.4f}        "
                    f"{h['plain_loss']:.4f}\n")
        f.write("\n")

        # contrast vs probe (last step plain mean)
        probe_final_mean = sum(final_loss) / len(final_loss)
        this_final_plain = history[-1]["plain_loss"]
        intent_only_init = sum(history[0]["per_tok_loss"][i] for i in intent_idx) / max(1, len(intent_idx))
        intent_only_final = sum(history[-1]["per_tok_loss"][i] for i in intent_idx) / max(1, len(intent_idx))
        f.write(f"  probe  final plain_mean       = {probe_final_mean:.4f}\n")
        f.write(f"  intent run final plain_mean   = {this_final_plain:.4f}\n")
        f.write(f"  intent run intent-bucket only = init {intent_only_init:.4f}  "
                f"→ final {intent_only_final:.4f}\n\n")

        # ---- cum grad heatmap ----
        f.write("================ Cumulative LoRA grad norm by (layer, module) ================\n")
        f.write(f"  L  |  {'q_proj':>7} {'k_proj':>7} {'v_proj':>7} "
                f"{'o_proj':>7} {'gate':>7} {'up':>7} {'down':>7}    total\n")
        for L in range(n_layers):
            row, tot = [], 0.0
            for m in TARGET_MODULES:
                v = cum_grad.get((L, m), 0.0)
                row.append(f"{v:7.3f}")
                tot += v
            f.write(f"  L{L:>2} | {' '.join(row)}    {tot:7.3f}\n")
        f.write("\n  Top-10 layers by total grad:\n")
        layer_totals = sorted(
            [(sum(cum_grad.get((L, m), 0.0) for m in TARGET_MODULES), L)
             for L in range(n_layers)], reverse=True,
        )
        for tot, L in layer_totals[:10]:
            f.write(f"    L{L:>2}: total={tot:.3f}\n")

        f.write("\n  Module-type totals:\n")
        module_totals = defaultdict(float)
        for (L, m), v in cum_grad.items():
            module_totals[m] += v
        for m in TARGET_MODULES:
            f.write(f"    {m:<10}: total={module_totals[m]:.3f}\n")

        # ---- top neurons + inventory ----
        f.write(f"\n\n================ Per-layer top-{TOP_K_NEURONS} neurons "
                f"by cum down_proj.lora_A grad ================\n")
        layer_max = {L: t.max().item() for L, t in cum_neuron_down.items()}
        f.write("\n  Layer ranking by max per-neuron cum-grad:\n")
        for L, mx in sorted(layer_max.items(), key=lambda kv: kv[1], reverse=True)[:15]:
            f.write(f"    L{L:>2}: max={mx:.4f}\n")

        for L in range(n_layers):
            if L not in cum_neuron_down:
                continue
            t = cum_neuron_down[L]
            topk = t.topk(TOP_K_NEURONS)
            f.write(f"\n  --- L{L} top-{TOP_K_NEURONS} (max={t.max().item():.4f}) ---\n")
            for i, (val, idx) in enumerate(zip(topk.values.tolist(),
                                                topk.indices.tolist())):
                key = (L, idx)
                inv_tag = ""
                if key in inventory:
                    inv = inventory[key]
                    inv_tag = (f"  [INV: {inv.get('tier','?')}, "
                               f"gain={inv.get('default_gain',0):+.1f}, "
                               f"region={inv.get('region','?')}] "
                               f"{inv.get('label','')[:40]}")
                f.write(f"    {i+1:>3}. L{L}#{idx:<5}  cum_grad={val:.4f}{inv_tag}\n")

        f.write(f"\n\n================ Inventory matches ================\n")
        total_matches = 0
        for L in range(n_layers):
            if L not in cum_neuron_down:
                continue
            t = cum_neuron_down[L]
            top_idx = set(t.topk(TOP_K_NEURONS).indices.tolist())
            matches = []
            for (li, ni), inv in inventory.items():
                if li == L and ni in top_idx:
                    rank = (-t).argsort().tolist().index(ni) + 1
                    matches.append((rank, ni, inv))
            if matches:
                matches.sort()
                f.write(f"\n  L{L:>2}: {len(matches)} matches\n")
                for rank, ni, inv in matches:
                    f.write(f"    rank {rank:>3}  L{L}#{ni:<5}  "
                            f"[{inv.get('tier','?')}, "
                            f"gain={inv.get('default_gain',0):+.1f}]  "
                            f"{inv.get('label','')[:50]}\n")
                total_matches += len(matches)
        f.write(f"\n  total matches: {total_matches}\n")

    print(f"  saved summary.txt", flush=True)
    print(f"\n[done] outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
