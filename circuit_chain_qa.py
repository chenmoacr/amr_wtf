"""
Chain intervention test on QA datasets.

We have identified a 7-node functional chain on GB01:
  L09#357   (driver of L12)
  L09#420   (suppressor of L12)
  L09#6039  (FFI: drives L33, suppresses L12)
  L11#2089  (mid amplifier)
  L12#4638  (verified suppressor of L33)
  L24#5330  (verified mid hub)
  L33#9054  (terminal)

This script generates Gemma 4 E2B answers under 4 modes on 3 datasets:

  baseline    no intervention
  ablate      chain[i] := 0       for every chain node i
  amplify     chain[i] := 2 * chain[i]
  invert      chain[i] := -chain[i]

Datasets:
  gb01    LeetCode 233 (math/code; chain origin)
  qa02    Chinese short literary prompt (~1.4k chars)
  qa04    Chinese short literary prompt (~2.1k chars)

For each (dataset, mode) we run greedy generation with max_new_tokens=400
and dump full output to outputs/chain_qa/<dataset>_<mode>.txt; we also
write a summary.json with previews and metadata.
"""
from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("J:/amr/amr_wtf")
MODEL_PATH = "J:/amr/models/gemma-4-E2B-it"
OUT_DIR = ROOT / "outputs" / "chain_qa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
# Empirical: GB01 baseline ran 8753 tokens, KV peak ~2.5GB on top of 9.3GB model.
# 12GB GPU max. Cap at 12288 to keep peak below ~12GB while still letting
# typical responses finish naturally (anything not done by 12k tokens is in a loop).
MAX_NEW_TOKENS = 12288

# (layer_idx, neuron_idx, role)
CHAIN = [
    ( 9,  357, "L09#357  driver→L12"),
    ( 9,  420, "L09#420  suppressor→L12"),
    ( 9, 6039, "L09#6039 FFI(+L33,-L12)"),
    (11, 2089, "L11#2089 mid amp"),
    (12, 4638, "L12#4638 mid suppressor of L33"),
    (24, 5330, "L24#5330 mid hub"),
    (33, 9054, "L33#9054 terminal"),
]


# ---------- intervention hooks ----------
def make_chain_hook(neuron_idx, mode):
    """Return a forward-pre-hook that intervenes on column n of the
    down_proj input tensor."""
    def pre(module, inputs):
        x = inputs[0].clone()
        if mode == "ablate":
            x[..., neuron_idx] = 0
        elif mode == "amplify":
            x[..., neuron_idx] = x[..., neuron_idx] * 2
        elif mode == "invert":
            x[..., neuron_idx] = x[..., neuron_idx] * -1
        else:
            raise ValueError(mode)
        return (x,) + inputs[1:]
    return pre


def install_chain(layers, mode):
    if mode == "baseline":
        return []
    handles = []
    for li, ni, _ in CHAIN:
        h = layers[li].mlp.down_proj.register_forward_pre_hook(
            make_chain_hook(ni, mode)
        )
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ---------- prompt loading ----------
def strip_user_block(s: str) -> str:
    if "[用户]" in s:
        s = s.split("[用户]", 1)[1]
    if "[助手]" in s:
        s = s.split("[助手]", 1)[0]
    return s.strip("\n").strip()


def load_prompts():
    out = []
    # gb01: LeetCode 233 from item A's query
    gb01 = json.loads((ROOT / "data" / "gemma_TargetedDrugs" / "gemma_code_GB01.json").read_text(encoding="utf-8"))
    out.append(("gb01", strip_user_block(gb01["A"][0]["query"])))
    # qa02 / qa04
    for name in ("QA02", "QA04"):
        d = json.loads((ROOT / "data" / f"claudeopus{name}.json").read_text(encoding="utf-8"))
        out.append((name.lower(), d["input"]))
    return out


# ---------- generation ----------
def split_thought_response(text: str):
    """Gemma 4 chat template:
        <|channel>thought\n...content...\n<channel|>...response...<turn|>
    Return (thought_str, response_str). If the channel close marker is missing
    (e.g. generation cut off mid-thought), response_str is empty.
    """
    eoc_marker = "<channel|>"
    if eoc_marker in text:
        thought, response = text.split(eoc_marker, 1)
        # strip leading "<|channel>thought\n"
        for prefix in ("<|channel>thought\n", "<|channel>thought"):
            if thought.startswith(prefix):
                thought = thought[len(prefix):]
                break
        return thought.rstrip("\n"), response.rstrip("<turn|>").rstrip()
    return text, ""


def generate_one(model, tokenizer, prompt_text, label):
    msgs = [{"role": "user", "content": prompt_text}]
    enc = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, enable_thinking=True,
    )
    ids = enc["input_ids"].to(DEVICE)
    am = enc["attention_mask"].to(DEVICE)
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
    eos_ids = list({i for i in eos_ids if i is not None})
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=am,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                       # greedy == temperature 0
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids if eos_ids else None,
            use_cache=True,
        )
    full = out[0].detach().cpu()
    gen = full[ids.shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=False)
    dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"    [{label}] {gen.shape[0]} tokens in {dt:.1f}s  (peak GPU={peak_gb:.2f}GB)", flush=True)
    # Aggressive GPU memory release
    del out, ids, am
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    return text, gen.shape[0], dt


def main():
    print(f"[load] {MODEL_PATH}  (BF16)")
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
    print(f"  layers={len(layers)}, alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB")

    print("\n[chain] 7 nodes:")
    for li, ni, role in CHAIN:
        print(f"  L{li:02d}#{ni:<5d}  {role}")

    prompts = load_prompts()
    print(f"\n[prompts] {len(prompts)} datasets:")
    for name, txt in prompts:
        print(f"  {name:>6s}: {len(txt)} chars  head={txt[:60]!r}")

    summary = {
        "chain": [{"layer": li, "index": ni, "role": role} for li, ni, role in CHAIN],
        "max_new_tokens": MAX_NEW_TOKENS,
        "runs": [],
    }

    modes = ["baseline", "ablate", "amplify", "invert"]
    for ds_name, prompt_text in prompts:
        print(f"\n=== dataset: {ds_name} ===", flush=True)
        for mode in modes:
            torch.cuda.empty_cache()
            gc.collect()
            label = f"{ds_name}_{mode}"
            out_path = OUT_DIR / f"{label}.txt"
            # Resume: skip if file exists and looks complete (has META section)
            if out_path.exists():
                txt = out_path.read_text(encoding="utf-8")
                if "================ META" in txt:
                    print(f"  [skip] {label}  (already done)", flush=True)
                    continue
                else:
                    print(f"  [redo] {label}  (file incomplete, regenerating)", flush=True)
            handles = install_chain(layers, mode)
            try:
                text, n_tok, dt = generate_one(model, tokenizer, prompt_text, label)
            finally:
                remove_hooks(handles)

            thought, response = split_thought_response(text)
            # write the structured log (input + cot + answer)
            out_path.write_text(
                "================ INPUT ================\n"
                + prompt_text
                + "\n\n================ COT (thought) ================\n"
                + thought
                + "\n\n================ ANSWER (response) ================\n"
                + response
                + "\n\n================ META ================\n"
                + f"mode        = {mode}\n"
                + f"dataset     = {ds_name}\n"
                + f"n_gen_tokens= {n_tok}\n"
                + f"elapsed_s   = {dt:.2f}\n"
                + f"thought_len = {len(thought)}\n"
                + f"answer_len  = {len(response)}\n",
                encoding="utf-8",
            )
            print(f"      thought={len(thought)}c  answer={len(response)}c  "
                  f"answer_head={response[:120].replace(chr(10),' ')!r}")
            summary["runs"].append({
                "dataset": ds_name,
                "mode": mode,
                "n_tokens": n_tok,
                "elapsed_s": round(dt, 2),
                "thought_len": len(thought),
                "answer_len": len(response),
                "answer_preview": response[:400],
                "out_file": out_path.name,
            })

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[save] {OUT_DIR / 'summary.json'}")
    print(f"[save] {len(summary['runs'])} response txt files in {OUT_DIR}")


if __name__ == "__main__":
    main()
