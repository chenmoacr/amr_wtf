"""
OpenAI-compatible FastAPI server for the OpusSAE Gemma model.

Two load paths:
  1. If gemma7_E2b_OpusSAE/down_proj_biases.pt exists, load original model
     and replace down_proj with biased version (recommended; equivalent
     math, no per-step hook overhead).
  2. Otherwise, load model from MODEL_PATH and run baseline.

Endpoints:
  POST /v1/chat/completions       -- OpenAI-compatible chat
  POST /v1/completions            -- OpenAI-compatible completion (no chat template)
  GET  /v1/models                 -- model list

Optional body fields beyond OpenAI:
  enable_thinking : bool, default true
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from threading import Thread

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    AutoTokenizer,
    Gemma4ForConditionalGeneration,
    TextIteratorStreamer,
)

sys.stdout.reconfigure(encoding="utf-8")

MODEL_PATH = Path("J:/amr/amr_wtf/gemma7_E2b_OpusSAE")
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
MODEL_NAME = "opus-sae-gemma-4-E2B"

app = FastAPI()


def apply_bias_merge(model, biases_path):
    """Replace each affected down_proj with a biased copy.
    Equivalent math to the SAE clamp pre-hooks but zero runtime overhead."""
    biases = torch.load(biases_path, weights_only=True)
    layers = model.model.language_model.layers
    n_replaced = 0
    for li, bias_vec in biases.items():
        old = layers[li].mlp.down_proj
        # Skip if down_proj already has bias matching shape (model was saved merged)
        if old.bias is not None and old.bias.shape == bias_vec.shape:
            n_replaced += 1
            continue
        new = nn.Linear(old.in_features, old.out_features, bias=True,
                        device=old.weight.device, dtype=old.weight.dtype)
        new.weight.data = old.weight.data
        new.bias.data = bias_vec.to(old.weight.dtype).to(old.weight.device)
        layers[li].mlp.down_proj = new
        n_replaced += 1
    return n_replaced


# ---------- model load ----------
print(f"[load] tokenizer + model: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE, low_cpu_mem_usage=True,
)
model.eval()
for attr in ("vision_tower", "audio_tower", "multi_modal_projector",
             "embed_vision", "embed_audio"):
    if hasattr(model.model, attr):
        setattr(model.model, attr, None)

biases_path = MODEL_PATH / "down_proj_biases.pt"
if biases_path.exists():
    n = apply_bias_merge(model, biases_path)
    print(f"[merge] applied bias merge to {n} layers (zero hook overhead)")
else:
    print(f"[warn] {biases_path} not found, running baseline (unmerged)")

print(f"[ready] alloc={torch.cuda.memory_allocated(0)/1e9:.2f}GB on cuda:0")


# ---------- generation core ----------
def run_generate(input_ids, attention_mask, max_tokens, temperature, streamer=None):
    do_sample = temperature > 0
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if streamer is not None:
        gen_kwargs["streamer"] = streamer
    with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        out = model.generate(**gen_kwargs)
    return out


# ---------- API ----------
@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "created": int(time.time()),
                  "owned_by": "amr_wtf"}],
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))
    max_tokens = int(body.get("max_tokens", 2048))
    temperature = float(body.get("temperature", 0.0))
    enable_thinking = bool(body.get("enable_thinking", True))

    chat_input = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=enable_thinking,
    )
    input_ids = chat_input["input_ids"].to(DEVICE)
    attention_mask = chat_input["attention_mask"].to(DEVICE)
    prompt_len = int(input_ids.shape[1])

    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    if stream:
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=False,
        )
        gen_kwargs = dict(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            streamer=streamer,
        )

        def run_in_thread():
            with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                model.generate(**gen_kwargs)

        def event_stream():
            thread = Thread(target=run_in_thread, daemon=True)
            thread.start()
            try:
                for new_text in streamer:
                    chunk = {
                        "id": request_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_NAME,
                        "choices": [{"index": 0, "delta": {"content": new_text},
                                     "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            finally:
                thread.join()
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    out = run_generate(input_ids, attention_mask, max_tokens, temperature)
    gen_ids = out[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    return JSONResponse({
        "id": request_id, "object": "chat.completion",
        "created": created, "model": MODEL_NAME,
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "index": 0, "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": int(gen_ids.shape[0]),
            "total_tokens": prompt_len + int(gen_ids.shape[0]),
        },
    })


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    stream = bool(body.get("stream", False))
    max_tokens = int(body.get("max_tokens", 2048))
    temperature = float(body.get("temperature", 0.0))

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    prompt_len = int(input_ids.shape[1])

    request_id = f"cmpl-{uuid.uuid4()}"
    created = int(time.time())

    out = run_generate(input_ids, attention_mask, max_tokens, temperature)
    gen_ids = out[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    return JSONResponse({
        "id": request_id, "object": "text_completion",
        "created": created, "model": MODEL_NAME,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": int(gen_ids.shape[0]),
            "total_tokens": prompt_len + int(gen_ids.shape[0]),
        },
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
