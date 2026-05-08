"""
Lightweight FastAPI proxy in front of the vLLM Gemma 4 OpusSAE server.

Why: vLLM serves a single model id, so cherry studio (and any OpenAI client)
can only see one entry. We want two visible models that both forward to the
same vLLM backend, differing only in whether the chat template renders the
<|channel>thought ... <channel|> reasoning section.

Virtual models exposed:
  opus-sae-gemma-4-E2B          enable_thinking = false   (default mode)
  opus-sae-gemma-4-E2B-think    enable_thinking = true    (forces thought)

Run after vLLM is already serving on 127.0.0.1:8000:

  conda activate gemmapreview   # any env with fastapi + httpx + uvicorn
  python J:\\amr\\amr_wtf\\chat_proxy.py

Then point cherry studio at  http://<host>:8001/v1  with key 'x'.
"""
from __future__ import annotations

import os
import sys

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

sys.stdout.reconfigure(encoding="utf-8")

VLLM_BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8000/v1")
REAL_MODEL = os.environ.get("REAL_MODEL", "opus-sae-gemma-4-E2B")
PORT = int(os.environ.get("PROXY_PORT", "8001"))
TIMEOUT = httpx.Timeout(600.0, connect=30.0)

VIRTUAL_MODELS = [
    {"id": "opus-sae-gemma-4-E2B",       "enable_thinking": False},
    {"id": "opus-sae-gemma-4-E2B-think", "enable_thinking": True},
]
_BY_ID = {m["id"]: m for m in VIRTUAL_MODELS}

app = FastAPI(title="OpusSAE proxy", version="1.0")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "owned_by": "amr_wtf",
                "created": 0,
            }
            for m in VIRTUAL_MODELS
        ],
    }


def _rewrite(body: dict, inject_thinking: bool) -> dict:
    """Rewrite model id to the real backend; inject chat_template_kwargs.enable_thinking."""
    requested = body.get("model", "")
    flag = _BY_ID.get(requested, {}).get("enable_thinking", False)
    body["model"] = REAL_MODEL
    if inject_thinking:
        kwargs = dict(body.get("chat_template_kwargs") or {})
        # only set if not already specified by client (allow override)
        kwargs.setdefault("enable_thinking", flag)
        body["chat_template_kwargs"] = kwargs
    return body


async def _forward(path: str, request: Request, inject_thinking: bool):
    body = await request.json()
    body = _rewrite(body, inject_thinking)
    is_stream = bool(body.get("stream", False))
    url = f"{VLLM_BASE}{path}"
    headers = {"content-type": "application/json"}

    if is_stream:
        async def event_gen():
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        # surface the error body as a single SSE chunk
                        err = await resp.aread()
                        yield b"data: " + err + b"\n\n"
                        return
                    async for chunk in resp.aiter_raw():
                        yield chunk
        return StreamingResponse(event_gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(url, json=body, headers=headers)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    return await _forward("/chat/completions", request, inject_thinking=True)


@app.post("/v1/completions")
async def completions(request: Request):
    # /v1/completions does not use a chat template, so do not inject thinking flag
    return await _forward("/completions", request, inject_thinking=False)


@app.get("/healthz")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{VLLM_BASE}/models")
            return {"proxy": "ok", "vllm_status": r.status_code}
    except Exception as e:
        return {"proxy": "ok", "vllm_error": str(e)}


if __name__ == "__main__":
    print(f"[proxy] listening on 0.0.0.0:{PORT}", flush=True)
    print(f"[proxy] backend      = {VLLM_BASE}", flush=True)
    print(f"[proxy] real_model   = {REAL_MODEL}", flush=True)
    print(f"[proxy] virtual ids  = {[m['id'] for m in VIRTUAL_MODELS]}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
