"""Production inference service.

Endpoints:
  POST /v1/generate          — single-shot generation (JSON)
  GET  /v1/generate/stream   — token-by-token SSE stream
  POST /v1/tokenize          — tokenize text
  GET  /health               — liveness + model status
  GET  /metadata             — model config
  GET  /metrics              — Prometheus exposition (text/plain)

Model and tokenizer paths come from env vars (INDICLM_CHECKPOINT,
INDICLM_TOKENIZER) set by `indiclm serve`, so the module can be imported
in tests without a live model.

GPU slot management: a single asyncio.Semaphore(1) serialises GPU requests
to avoid OOM under concurrent load. fp16 autocast is applied automatically
when the model is on CUDA.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import sentencepiece as spm
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from indiclm.models.config import ModelConfig
from indiclm.models.transformer import DecoderOnlyTransformer
from indiclm.utils.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────

_state: dict = {
    "model": None,
    "tokenizer": None,
    "model_config": None,
    "checkpoint_path": None,
    "device": None,
}

# Serialise GPU requests; allows concurrent tokenise/health without blocking
_gpu_semaphore: asyncio.Semaphore = asyncio.Semaphore(1)

MAX_SEQ_LEN_CAP = 2048

# ── Prometheus metrics (optional) ─────────────────────────────────────────────

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _REQ_COUNT = Counter("indiclm_requests_total", "Requests by endpoint", ["endpoint", "status"])
    _REQ_LATENCY = Histogram(
        "indiclm_request_duration_seconds",
        "Request latency",
        ["endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )
    _TOKENS_GENERATED = Counter("indiclm_tokens_generated_total", "Tokens generated")
    _GPU_MEM = Gauge("indiclm_gpu_memory_allocated_bytes", "GPU memory allocated")
    _ACTIVE = Gauge("indiclm_active_requests", "Requests currently in flight")
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

# Simple fallback counter used when prometheus_client is not installed
_fallback_metrics: dict[str, int] = {
    "requests_total": 0,
    "generate_requests": 0,
    "stream_requests": 0,
    "tokenize_requests": 0,
    "errors_total": 0,
    "tokens_generated_total": 0,
}


def _inc(endpoint: str, status: str = "ok") -> None:
    _fallback_metrics["requests_total"] += 1
    if endpoint == "generate":
        _fallback_metrics["generate_requests"] += 1
    elif endpoint == "stream":
        _fallback_metrics["stream_requests"] += 1
    elif endpoint == "tokenize":
        _fallback_metrics["tokenize_requests"] += 1
    if status != "ok":
        _fallback_metrics["errors_total"] += 1
    if _PROM_AVAILABLE:
        _REQ_COUNT.labels(endpoint=endpoint, status=status).inc()


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def _load_model() -> None:
    checkpoint_path = os.environ.get("INDICLM_CHECKPOINT")
    tokenizer_path = os.environ.get("INDICLM_TOKENIZER")
    if not checkpoint_path or not tokenizer_path:
        log.warning("inference_not_configured", note="INDICLM_CHECKPOINT/INDICLM_TOKENIZER not set")
        return
    if not Path(checkpoint_path).exists():
        log.warning("checkpoint_not_found", path=checkpoint_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**payload["config"]["model_config"])
    model = DecoderOnlyTransformer(model_config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    # fp16 inference on CUDA for lower latency and memory
    if device.type == "cuda":
        model = model.half()

    _state["model"] = model
    _state["tokenizer"] = spm.SentencePieceProcessor(model_file=tokenizer_path)
    _state["model_config"] = model_config
    _state["checkpoint_path"] = checkpoint_path
    _state["device"] = device
    log.info(
        "model_loaded",
        checkpoint=checkpoint_path,
        device=str(device),
        params=model.num_parameters(),
        precision="fp16" if device.type == "cuda" else "fp32",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _load_model()
    yield
    log.info("inference_server_shutdown")


app = FastAPI(
    title="IndicLM Inference API",
    version="1.0.0",
    description="Multilingual Indic LM inference — 10 Indian languages + English",
    lifespan=lifespan,
)

# ── Request / response schemas ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(64, ge=1, le=512)
    temperature: float = Field(1.0, gt=0.0, le=2.0)
    top_k: int | None = Field(None, ge=1, le=1000)


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float
    device: str


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class TokenizeResponse(BaseModel):
    token_ids: list[int]
    tokens: list[str]
    count: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_model() -> tuple[DecoderOnlyTransformer, spm.SentencePieceProcessor]:
    if _state["model"] is None or _state["tokenizer"] is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Set INDICLM_CHECKPOINT/INDICLM_TOKENIZER and restart.",
        )
    return _state["model"], _state["tokenizer"]


def _encode_prompt(tokenizer: spm.SentencePieceProcessor, prompt: str, budget: int) -> list[int]:
    ids: list[int] = tokenizer.encode(prompt, out_type=int)
    if len(ids) >= budget:
        ids = ids[-(budget - 1):]
    return ids


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    model, tokenizer = _require_model()
    device: torch.device = _state["device"]
    max_seq_len = min(_state["model_config"].max_seq_len, MAX_SEQ_LEN_CAP)

    if _PROM_AVAILABLE:
        _ACTIVE.inc()
    start = time.perf_counter()

    async with _gpu_semaphore:
        try:
            input_ids = _encode_prompt(tokenizer, req.prompt, max_seq_len - req.max_new_tokens)
            x = torch.tensor([input_ids], dtype=torch.long, device=device)
            autocast_ctx = (
                torch.amp.autocast("cuda", dtype=torch.float16)
                if device.type == "cuda"
                else torch.amp.autocast("cpu", enabled=False)
            )
            with torch.no_grad(), autocast_ctx:
                out = model.generate(
                    x,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                )
            generated_ids = out[0, len(input_ids):].tolist()
            text = tokenizer.decode(generated_ids)
        except Exception as e:
            _inc("generate", "error")
            if _PROM_AVAILABLE:
                _ACTIVE.dec()
            raise HTTPException(status_code=500, detail=f"generation_failed: {e}") from e

    latency = time.perf_counter() - start
    _inc("generate")
    if _PROM_AVAILABLE:
        _REQ_LATENCY.labels(endpoint="generate").observe(latency)
        _TOKENS_GENERATED.inc(len(generated_ids))
        _ACTIVE.dec()
        if device.type == "cuda":
            _GPU_MEM.set(torch.cuda.memory_allocated())
    else:
        _fallback_metrics["tokens_generated_total"] += len(generated_ids)

    return GenerateResponse(
        text=text,
        prompt_tokens=len(input_ids),
        generated_tokens=len(generated_ids),
        latency_ms=round(latency * 1000, 2),
        device=str(device),
    )


@app.get("/v1/generate/stream")
async def generate_stream(
    prompt: str = Query(..., min_length=1, max_length=4000),
    max_new_tokens: int = Query(64, ge=1, le=512),
    temperature: float = Query(1.0, gt=0.0, le=2.0),
    top_k: int | None = Query(None, ge=1, le=1000),
) -> StreamingResponse:
    """Server-Sent Events stream: one JSON object per token, then [DONE]."""
    model, tokenizer = _require_model()
    device: torch.device = _state["device"]
    max_seq_len = min(_state["model_config"].max_seq_len, MAX_SEQ_LEN_CAP)

    async def _token_stream() -> AsyncIterator[str]:
        _inc("stream")
        input_ids = _encode_prompt(tokenizer, prompt, max_seq_len - max_new_tokens)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if device.type == "cuda"
            else torch.amp.autocast("cpu", enabled=False)
        )

        async with _gpu_semaphore:
            with torch.no_grad(), autocast_ctx:
                for _ in range(max_new_tokens):
                    cond = x[:, -max_seq_len:]
                    logits, _ = model(cond)
                    logits = logits[:, -1, :] / max(temperature, 1e-5)
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float("-inf")
                    next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                    x = torch.cat([x, next_token], dim=1)
                    token_text = tokenizer.decode([int(next_token.item())])
                    yield f"data: {json.dumps({'token': token_text, 'token_id': next_token.item()})}\n\n"
                    await asyncio.sleep(0)  # yield control to the event loop

        yield "data: [DONE]\n\n"

    return StreamingResponse(_token_stream(), media_type="text/event-stream")


@app.post("/v1/tokenize", response_model=TokenizeResponse)
async def tokenize(req: TokenizeRequest) -> TokenizeResponse:
    _inc("tokenize")
    _, tokenizer = _require_model()
    ids = tokenizer.encode(req.text, out_type=int)
    pieces = tokenizer.encode(req.text, out_type=str)
    return TokenizeResponse(token_ids=ids, tokens=pieces, count=len(ids))


@app.get("/health")
async def health() -> dict:
    loaded = _state["model"] is not None
    gpu_info: dict = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "memory_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        }
    return {
        "status": "ok" if loaded else "no_model_loaded",
        "checkpoint": _state["checkpoint_path"],
        **gpu_info,
    }


@app.get("/metadata")
async def metadata() -> dict:
    if _state["model_config"] is None:
        return {"model_loaded": False}
    cfg = _state["model_config"]
    return {
        "model_loaded": True,
        "checkpoint": _state["checkpoint_path"],
        "device": str(_state["device"]),
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "max_seq_len": cfg.max_seq_len,
        "use_moe": cfg.use_moe,
        "parameters": _state["model"].num_parameters(),
    }


@app.get("/metrics")
async def metrics() -> Response:
    if _PROM_AVAILABLE:
        if _state["device"] is not None and str(_state["device"]).startswith("cuda"):
            _GPU_MEM.set(torch.cuda.memory_allocated())
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    return Response(
        content=json.dumps(_fallback_metrics, indent=2),
        media_type="application/json",
    )
