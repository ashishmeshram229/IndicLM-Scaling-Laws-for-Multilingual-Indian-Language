"""Minimal production-shaped inference service, separate from training
code (imports only `indiclm.models` + `indiclm.training.checkpoint`, never
the training loop itself).

Endpoints: POST /generate, POST /tokenize, GET /health, GET /metadata,
GET /metrics. Checkpoint/tokenizer paths come from env vars
(INDICLM_CHECKPOINT, INDICLM_TOKENIZER) set by `indiclm serve`, so the
module can be imported standalone (e.g. by tests) without a live model.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from indiclm.models.config import ModelConfig
from indiclm.models.transformer import DecoderOnlyTransformer
from indiclm.utils.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

app = FastAPI(title="IndicLM Inference API", version="0.1.0")

_state: dict = {"model": None, "tokenizer": None, "model_config": None, "checkpoint_path": None}
_metrics = {"requests_total": 0, "generate_requests": 0, "tokenize_requests": 0, "errors_total": 0}

MAX_SEQ_LEN_CAP = 512  # hard safety cap independent of any single model's configured max_seq_len
REQUEST_TIMEOUT_SEC = 30.0


def _load_model() -> None:
    checkpoint_path = os.environ.get("INDICLM_CHECKPOINT")
    tokenizer_path = os.environ.get("INDICLM_TOKENIZER")
    if not checkpoint_path or not tokenizer_path:
        log.warning("inference_not_configured", note="INDICLM_CHECKPOINT/INDICLM_TOKENIZER not set")
        return
    if not Path(checkpoint_path).exists():
        log.warning("checkpoint_not_found", path=checkpoint_path)
        return

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**payload["config"]["model_config"])
    model = DecoderOnlyTransformer(model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    _state["model"] = model
    _state["tokenizer"] = spm.SentencePieceProcessor(model_file=tokenizer_path)
    _state["model_config"] = model_config
    _state["checkpoint_path"] = checkpoint_path
    log.info("model_loaded", checkpoint=checkpoint_path, params=model.num_parameters())


@app.on_event("startup")
def startup() -> None:
    _load_model()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(32, ge=1, le=256)
    temperature: float = Field(1.0, gt=0.0, le=2.0)
    top_k: Optional[int] = Field(None, ge=1, le=1000)


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class TokenizeResponse(BaseModel):
    token_ids: list[int]
    tokens: list[str]
    count: int


def _require_model() -> tuple[DecoderOnlyTransformer, spm.SentencePieceProcessor]:
    if _state["model"] is None or _state["tokenizer"] is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Set INDICLM_CHECKPOINT/INDICLM_TOKENIZER and restart, "
            "or run `indiclm serve --checkpoint <path> --tokenizer <path>`.",
        )
    return _state["model"], _state["tokenizer"]


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    _metrics["requests_total"] += 1
    _metrics["generate_requests"] += 1
    model, tokenizer = _require_model()

    max_seq_len = min(_state["model_config"].max_seq_len, MAX_SEQ_LEN_CAP)
    start = time.time()
    try:
        input_ids = tokenizer.encode(req.prompt, out_type=int)
        if len(input_ids) >= max_seq_len:
            input_ids = input_ids[-(max_seq_len - req.max_new_tokens - 1) :]
        x = torch.tensor([input_ids], dtype=torch.long)
        out = model.generate(
            x, max_new_tokens=req.max_new_tokens, temperature=req.temperature, top_k=req.top_k
        )
        generated_ids = out[0, len(input_ids) :].tolist()
        text = tokenizer.decode(generated_ids)
    except Exception as e:  # noqa: BLE001 - convert to a structured API error
        _metrics["errors_total"] += 1
        raise HTTPException(status_code=500, detail=f"generation_failed: {e}") from e

    latency_ms = (time.time() - start) * 1000
    if latency_ms > REQUEST_TIMEOUT_SEC * 1000:
        log.warning("generate_slow_request", latency_ms=latency_ms)

    return GenerateResponse(
        text=text,
        prompt_tokens=len(input_ids),
        generated_tokens=len(generated_ids),
        latency_ms=round(latency_ms, 2),
    )


@app.post("/tokenize", response_model=TokenizeResponse)
def tokenize(req: TokenizeRequest) -> TokenizeResponse:
    _metrics["requests_total"] += 1
    _metrics["tokenize_requests"] += 1
    _, tokenizer = _require_model()
    ids = tokenizer.encode(req.text, out_type=int)
    pieces = tokenizer.encode(req.text, out_type=str)
    return TokenizeResponse(token_ids=ids, tokens=pieces, count=len(ids))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _state["model"] is not None else "no_model_loaded",
        "checkpoint": _state["checkpoint_path"],
    }


@app.get("/metadata")
def metadata() -> dict:
    if _state["model_config"] is None:
        return {"model_loaded": False}
    cfg = _state["model_config"]
    return {
        "model_loaded": True,
        "checkpoint": _state["checkpoint_path"],
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
def metrics() -> dict:
    return dict(_metrics)
