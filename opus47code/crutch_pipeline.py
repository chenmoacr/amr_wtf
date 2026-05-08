"""
Reusable crutch-removal SFT pipeline.

Usage:
  python crutch_pipeline.py configs/mode_D.yaml

The YAML config specifies:
  - mode_name + out_dir
  - which neurons to suppress (inventory + top-N from previous mode raw.pt files)
  - training hyperparameters
  - inference modes (suppress on / off, etc.)

To create a new mode (E, F, ...):
  1. Copy mode_D.yaml to mode_E.yaml
  2. Add the previous mode's raw.pt to suppression.add_recruits
  3. Update mode_name and out_dir
  4. Run: python crutch_pipeline.py configs/mode_E.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import yaml
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
DATA_DIR = ROOT / "data"
NEURONS_JSON = ROOT / "chat" / "neurons.json"
DEVICE = "cuda:0"


# ============================ helpers ============================

def parse_lora_param_name(name, target_modules):
    parts = name.split(".")
    layer_idx = None
    if "layers" in parts:
        i = parts.index("layers")
        try:
            layer_idx = int(parts[i + 1])
        except (ValueError, IndexError):
            pass
    module = None
    for m in target_modules:
        if m in parts:
            module = m
            break
    ab = "A" if "lora_A" in name else ("B" if "lora_B" in name else None)
    return layer_idx, module, ab


def find_substring_token_positions(target_strs, substrings):
    full = "".join(target_strs)
    ends = []
    cum = 0
    for s in target_strs:
        cum += len(s)
        ends.append(cum)
    out = set()
    for sub in substrings:
        sub = sub.strip()
        if not sub:
            continue
        start = 0
        while True:
            idx = full.find(sub, start)
            if idx < 0:
                break
            char_end = idx + len(sub)
            for i, end in enumerate(ends):
                tok_start = end - len(target_strs[i])
                if tok_start < char_end and end > idx:
                    out.add(i)
            start = idx + 1
    return out


def truncate_user_text(user_text, tokenizer, max_tokens):
    msgs = [{"role": "user", "content": user_text}]
    full_len = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )["input_ids"].shape[1]
    if full_len <= max_tokens:
        return user_text, full_len, False
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
    return user_text[-keep:], full_len, True


def build_qa(qa_id, qa_data, tokenizer, train_cfg):
    user_text = qa_data["input"]
    opus_answer = qa_data["output"]
    glue = qa_data.get("glue_sentences", [])
    concl_dict = qa_data.get("conclusion_analysis", {})
    concl_sentences = []
    for k, lst in concl_dict.items():
        if isinstance(lst, list):
            concl_sentences.extend(lst)

    user_text_t, raw_len, did_trunc = truncate_user_text(
        user_text, tokenizer, train_cfg["max_prompt_tokens"],
    )
    msgs_user = [{"role": "user", "content": user_text_t}]
    user_enc = tokenizer.apply_chat_template(
        msgs_user, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=False,
    )
    assistant_start = user_enc["input_ids"].shape[1]
    msgs_full = [
        {"role": "user", "content": user_text_t},
        {"role": "assistant", "content": opus_answer},
    ]
    enc = tokenizer.apply_chat_template(
        msgs_full, tokenize=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    )
    input_ids = enc["input_ids"]
    T = input_ids.shape[1]
    n_target = T - assistant_start

    target_ids = input_ids[0, assistant_start:].tolist()
    target_strs = [tokenizer.decode([t], skip_special_tokens=False) for t in target_ids]

    glue_pos = find_substring_token_positions(target_strs, glue)
    concl_pos = find_substring_token_positions(target_strs, concl_sentences)

    weights, n_glue, n_concl, n_filler = [], 0, 0, 0
    w = train_cfg["weights"]
    for i in range(n_target):
        if i in concl_pos:
            weights.append(w["concl"]); n_concl += 1
        elif i in glue_pos:
            weights.append(w["glue"]); n_glue += 1
        else:
            weights.append(w["filler"]); n_filler += 1

    return {
        "qa_id": qa_id, "user_text_t": user_text_t,
        "raw_prompt_tokens": raw_len, "did_truncate": did_trunc,
        "T": T, "assistant_start": assistant_start, "n_target": n_target,
        "input_ids": input_ids, "target_strs": target_strs,
        "weights": weights,
        "n_glue": n_glue, "n_concl": n_concl, "n_filler": n_filler,
    }


def build_suppression_set(cfg):
    """Returns suppress_dict {layer: set(neuron_ids)} and a listing for logging."""
    suppress = defaultdict(set)
    listing = []
    sup_cfg = cfg["suppression"]

    if sup_cfg.get("include_inventory", True):
        nj = json.loads(NEURONS_JSON.read_text(encoding="utf-8"))
        tiers = set(sup_cfg.get("inventory_tiers", []))
        regions = set(sup_cfg.get("inventory_regions", []))
        for n in nj["known_neurons"]:
            if n["tier"] in tiers and n.get("region", "answer") in regions:
                suppress[int(n["layer"])].add(int(n["index"]))
                listing.append(("inventory", n["id"], n["tier"], n.get("region")))

    for src in sup_cfg.get("add_recruits", []) or []:
        path = Path(src["source"])
        if not path.exists():
            raise FileNotFoundError(f"recruit source not found: {path}")
        prev = torch.load(path, weights_only=False)
        prev_sup = {int(k): set(v) for k, v in prev["suppress_dict"].items()}
        cum_neuron = prev["cum_neuron_down"]

        candidates = []
        for L, t in cum_neuron.items():
            for n in range(t.shape[0]):
                if n in prev_sup.get(L, set()):
                    continue
                candidates.append((t[n].item(), L, n))
        candidates.sort(reverse=True)

        added = 0
        top_n = src["top_n"]
        label = src.get("label", path.parent.name)
        for val, L, n in candidates:
            if added >= top_n:
                break
            if n in suppress[L]:
                continue
            suppress[L].add(n)
            listing.append((f"recruit:{label}", f"L{L}#{n}", f"prev_cum={val:.4f}"))
            added += 1

    return dict(suppress), listing


def install_suppression_hooks(layers, suppress_dict):
    handles = []
    for layer_idx, idxs in suppress_dict.items():
        if not idxs:
            continue
        idx_tensor = torch.tensor(sorted(idxs), dtype=torch.long)
        layer = layers[layer_idx]
        def make_hook(idx_t):
            def pre(module, inputs):
                x = inputs[0]
                idx = idx_t.to(x.device)
                x_new = x.clone()
                x_new[..., idx] = 0
                return (x_new,)
            return pre
        h = layer.mlp.down_proj.register_forward_pre_hook(make_hook(idx_tensor))
        handles.append(h)
    return handles


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


# ============================ main ============================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to YAML config")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # save resolved config for reproducibility
    (out_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8",
    )

    train_cfg = cfg["training"]
    infer_cfg = cfg["inference"]
    target_modules = train_cfg["target_modules"]

    print(f"[mode] {cfg['mode_name']}: {cfg.get('description', '')}", flush=True)
    print(f"[out_dir] {out_dir}", flush=True)

    # ---- model + LoRA ----
    print(f"\n[load model] {MODEL_PATH}", flush=True)
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
        r=train_cfg["lora_r"], lora_alpha=train_cfg["lora_alpha"],
        target_modules=target_modules,
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
    inner = model.base_model.model.model.language_model
    inner.config.use_cache = False
    if hasattr(inner, "gradient_checkpointing_enable"):
        inner.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for layer in inner.layers:
        if hasattr(layer, "gradient_checkpointing"):
            layer.gradient_checkpointing = True
    model.train()
    n_layers = len(inner.layers)
    layers_list = inner.layers

    # ---- build suppression ----
    suppress_dict, listing = build_suppression_set(cfg)
    n_suppressed = sum(len(s) for s in suppress_dict.values())
    print(f"\n[CRUTCH {cfg['mode_name']}] suppressing {n_suppressed} neurons "
          f"across {len(suppress_dict)} layers", flush=True)
    for L in sorted(suppress_dict.keys()):
        print(f"  L{L:>2} ({len(suppress_dict[L])}): {sorted(suppress_dict[L])}",
              flush=True)

    suppression_handles = install_suppression_hooks(layers_list, suppress_dict)
    print(f"[hooks] installed {len(suppression_handles)}", flush=True)

    # ---- load QAs ----
    qa_ids = train_cfg["qa_ids"]
    print(f"\n[data] loading {len(qa_ids)} QA files ...", flush=True)
    qas = []
    for i in qa_ids:
        path = DATA_DIR / f"claudeopusQA0{i}.json"
        qa_data = json.loads(path.read_text(encoding="utf-8"))
        qa = build_qa(i, qa_data, tokenizer, train_cfg)
        qas.append(qa)
        print(f"  QA0{i}: T={qa['T']} n_target={qa['n_target']} "
              f"glue={qa['n_glue']} concl={qa['n_concl']} filler={qa['n_filler']}",
              flush=True)

    # logits_to_keep probe
    qa = qas[0]
    K0 = qa["n_target"] + 1
    logits_kw = None
    inp = qa["input_ids"].to(DEVICE)
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                _p = model(input_ids=inp, use_cache=False, **{kw: K0})
            if _p.logits.shape[1] == K0:
                logits_kw = kw; del _p; torch.cuda.empty_cache(); break
            del _p
        except (TypeError, ValueError):
            continue
    print(f"[probe-fwd] using {logits_kw!r}", flush=True)
    del inp; torch.cuda.empty_cache()

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=train_cfg["lr"],
    )
    n_epochs = train_cfg["n_epochs"]
    total_steps = n_epochs * len(qas)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=total_steps, eta_min=1e-6,
    )

    history = []
    cum_grad = defaultdict(float)
    cum_neuron_down = {}
    per_qa_loss_curves = {i: [] for i in qa_ids}

    print(f"\n[train] {n_epochs} epochs × {len(qas)} = {total_steps} steps  "
          f"LR={train_cfg['lr']}", flush=True)

    step = 0
    try:
        for epoch in range(n_epochs):
            for qa in qas:
                t0 = time.time()
                inp = qa["input_ids"].to(DEVICE)
                astart = qa["assistant_start"]
                n_target = qa["n_target"]
                K = n_target + 1
                wtensor = torch.tensor(qa["weights"], dtype=torch.float32,
                                        device=DEVICE)

                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    fwk = {"input_ids": inp, "use_cache": False}
                    if logits_kw is not None:
                        fwk[logits_kw] = K
                    out = model(**fwk)

                if logits_kw is not None:
                    shift_logits = out.logits[:, :-1, :]
                else:
                    shift_logits = out.logits[:, astart - 1:-1, :]
                shift_labels = inp[:, astart:]

                ce_per_tok = F.cross_entropy(
                    shift_logits.float().view(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="none",
                )
                weighted = ce_per_tok * wtensor
                loss = weighted.sum() / wtensor.sum().clamp(min=1.0)
                plain = ce_per_tok.detach().mean().item()

                del out, shift_logits, shift_labels, inp
                torch.cuda.empty_cache()

                loss.backward()

                for n, p in model.named_parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    li, mod, ab = parse_lora_param_name(n, target_modules)
                    if li is None or mod is None or ab is None:
                        continue
                    gn = p.grad.float().norm().item()
                    cum_grad[(li, mod)] += gn
                    if mod == "down_proj" and ab == "A":
                        col = p.grad.float().norm(dim=0).cpu()
                        if li in cum_neuron_down:
                            cum_neuron_down[li] += col
                        else:
                            cum_neuron_down[li] = col.clone()

                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)

                peak = torch.cuda.max_memory_allocated(0) / 1e9
                torch.cuda.reset_peak_memory_stats(0)

                history.append({
                    "step": step, "epoch": epoch, "qa_id": qa["qa_id"],
                    "weighted_loss": loss.item(), "plain_loss": plain,
                })
                per_qa_loss_curves[qa["qa_id"]].append((plain, loss.item()))

                print(f"  step {step+1:>2}/{total_steps}  ep{epoch+1}/{n_epochs}  "
                      f"QA0{qa['qa_id']}  weighted={loss.item():.4f}  "
                      f"plain={plain:.4f}  {time.time()-t0:.1f}s  "
                      f"peak={peak:.2f}GB", flush=True)
                step += 1
                torch.cuda.empty_cache()

        # ---- save adapter ----
        adapter_dir = out_dir / "lora_adapter"
        model.save_pretrained(str(adapter_dir))
        print(f"\n[save] LoRA adapter → {adapter_dir}", flush=True)

        torch.save({
            "config": cfg,
            "suppress_dict": {L: list(s) for L, s in suppress_dict.items()},
            "listing": listing,
            "history": history,
            "per_qa_loss_curves": per_qa_loss_curves,
            "cum_grad": dict(cum_grad),
            "cum_neuron_down": cum_neuron_down,
        }, out_dir / "raw.pt")
        print(f"[save] raw.pt", flush=True)

        # ---- inference ----
        model.eval()
        eos_ids = get_eos_ids(tokenizer)

        infer_qa_id = infer_cfg["qa_id"]
        infer_qa = next((q for q in qas if q["qa_id"] == infer_qa_id), None)
        if infer_qa is None:
            print(f"[warn] inference qa_id={infer_qa_id} not in trained qa_ids; "
                  f"defaulting to first", flush=True)
            infer_qa = qas[0]

        user_text = infer_qa["user_text_t"]
        msgs = [{"role": "user", "content": user_text}]
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        input_ids_inf = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)
        prompt_len = input_ids_inf.shape[1]
        max_new = infer_cfg["max_new_tokens"]
        print(f"\n[infer] qa_id={infer_qa['qa_id']}  prompt_tokens={prompt_len}  "
              f"max_new={max_new}", flush=True)

        def gen(label):
            print(f"\n[{label}] generating ...", flush=True)
            t0 = time.time()
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                                                SDPBackend.MATH]):
                out = model.generate(
                    input_ids=input_ids_inf, attention_mask=attn_mask,
                    max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=eos_ids if eos_ids else None,
                    use_cache=True,
                )
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated(0) / 1e9
            torch.cuda.reset_peak_memory_stats(0)
            full = out[0].detach().cpu()
            gen_ids = full[prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=False).rstrip("<turn|>").rstrip()
            print(f"[{label}] {gen_ids.shape[0]}t  {dt:.1f}s  peak={peak:.2f}GB",
                  flush=True)
            print(f"[{label}] head: {text[:180].replace(chr(10), ' ')!r}", flush=True)
            return text, gen_ids.shape[0], dt, peak

        infer_results = []
        # collect modes; switch hooks accordingly
        for mode_spec in infer_cfg["modes"]:
            label = mode_spec["label"]
            hooks_active = mode_spec["hooks_active"]
            note = mode_spec.get("note", "")
            if hooks_active:
                # hooks already installed
                pass
            else:
                # remove hooks before this gen
                for h in suppression_handles:
                    h.remove()
                suppression_handles = []
            text, ntok, dt, peak = gen(label)
            infer_results.append({
                "label": label, "hooks_active": hooks_active, "note": note,
                "text": text, "ntok": ntok, "dt": dt, "peak": peak,
            })
            torch.cuda.empty_cache()

    finally:
        for h in suppression_handles:
            h.remove()

    for r in infer_results:
        path = out_dir / f"infer_{r['label']}.txt"
        path.write_text(
            f"================ INPUT (truncated to {len(user_text)}c) ================\n"
            + user_text
            + f"\n\n================ ANSWER ({r['label']}) ================\n"
            + r["text"]
            + "\n\n================ META ================\n"
            + f"label             = {r['label']}\n"
            + f"note              = {r['note']}\n"
            + f"hooks_active      = {r['hooks_active']}\n"
            + f"prompt_chars      = {len(user_text)}\n"
            + f"prompt_tokens     = {prompt_len}\n"
            + f"gen_tokens        = {r['ntok']}\n"
            + f"elapsed_s         = {r['dt']:.2f}\n"
            + f"peak_gpu_gb       = {r['peak']:.2f}\n"
            + f"answer_chars      = {len(r['text'])}\n"
            + f"suppression_set   = {n_suppressed} neurons / "
              f"{len(suppress_dict)} layers\n",
            encoding="utf-8",
        )
        print(f"[save] {path.name}  ({len(r['text'])}c)", flush=True)

    # ---- summary ----
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"================ Mode {cfg['mode_name']} ================\n\n")
        f.write(f"{cfg.get('description', '')}\n\n")
        f.write(f"Total suppressed: {n_suppressed} across {len(suppress_dict)} layers\n\n")

        f.write("================ Suppression listing ================\n")
        n_inv = sum(1 for x in listing if x[0] == "inventory")
        f.write(f"  inventory entries: {n_inv}\n")
        for src in cfg["suppression"].get("add_recruits", []) or []:
            label = src.get("label", "")
            count = sum(1 for x in listing if x[0].startswith(f"recruit:{label}"))
            f.write(f"  recruits from {label}: {count}\n")
        f.write("\n")

        f.write("================ Per-QA convergence ================\n")
        f.write(f"  qa   | epoch1 | epoch2 | epoch3 | epoch4 |  Δ\n")
        for qid, lc in per_qa_loss_curves.items():
            plains = [x[0] for x in lc]
            cells = "  ".join(f"{p:.3f}" for p in plains)
            f.write(f"  QA0{qid}  | {cells}  | {plains[0]-plains[-1]:+.3f}\n")

        # module totals
        mtot = defaultdict(float)
        for (L, m), v in cum_grad.items():
            mtot[m] += v
        f.write(f"\n================ Module totals ================\n")
        for m in target_modules:
            f.write(f"  {m:<10}: {mtot[m]:.2f}\n")

        # layer top-15
        f.write(f"\n================ Top-15 layers by cum-grad ================\n")
        ltot = sorted(
            [(L, sum(cum_grad.get((L, m), 0.0) for m in target_modules))
             for L in range(n_layers)], key=lambda kv: kv[1], reverse=True,
        )
        for rk, (L, v) in enumerate(ltot[:15], 1):
            f.write(f"  rank {rk:>2}  L{L:>2}  total={v:.2f}\n")

        # next-tier recruits
        candidates = []
        for L, t in cum_neuron_down.items():
            sup = suppress_dict.get(L, set())
            for n in range(t.shape[0]):
                if n in sup:
                    continue
                candidates.append((t[n].item(), L, n))
        candidates.sort(reverse=True)
        f.write(f"\n================ TOP 30 RECRUITS at this level ================\n")
        f.write(f"  (these are the candidates for next-mode suppression)\n")
        f.write(f"  rank | L#nrn       | cum_grad\n")
        for i, (val, L, n) in enumerate(candidates[:30], 1):
            f.write(f"  {i:>3}  | L{L}#{n:<6} | {val:.4f}\n")

    print(f"[save] summary.txt", flush=True)
    print(f"\n[done] outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
