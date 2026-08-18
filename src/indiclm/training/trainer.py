"""The real training engine: gradient accumulation, AdamW, cosine decay
with warmup, gradient clipping, checkpointing/resume, periodic evaluation,
and structured per-step metrics logging (loss, lr, grad_norm, tokens_seen,
tokens/sec, step_time, data_loading_time).

BF16/FP16: this milestone runs CPU-only (see docs/architecture.md); mixed
precision is wired via `torch.autocast` and only activates when `device`
is "cuda" and `precision` requests it, so the code path is exercised
honestly rather than claimed without hardware to validate it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from indiclm.models.config import ModelConfig
from indiclm.models.transformer import DecoderOnlyTransformer
from indiclm.monitoring.anomaly import AnomalyDetector
from indiclm.training.checkpoint import config_to_json, load_checkpoint, save_checkpoint
from indiclm.training.scheduler import CosineWarmupScheduler
from indiclm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrainingConfig:
    output_dir: Path
    max_steps: int
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 20
    grad_clip: float = 1.0
    eval_every: int = 50
    checkpoint_every: int = 100
    log_every: int = 10
    device: str = "cpu"
    precision: str = "fp32"  # "fp32" | "bf16" (bf16 only takes effect on cuda)
    seed: int = 0
    resume_from: Path | None = None


@dataclass
class TrainingResult:
    final_step: int
    final_train_loss: float
    final_val_loss: float | None
    tokens_seen: int
    total_train_time_sec: float
    mean_tokens_per_sec: float
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total**0.5


def train(
    model_config: ModelConfig,
    train_config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
) -> TrainingResult:
    torch.manual_seed(train_config.seed)
    device = torch.device(train_config.device)
    model = DecoderOnlyTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=train_config.warmup_steps, total_steps=train_config.max_steps
    )
    detector = AnomalyDetector()

    step = 0
    tokens_seen = 0
    if train_config.resume_from is not None and Path(train_config.resume_from).exists():
        state = load_checkpoint(train_config.resume_from, model, optimizer, scheduler)
        step = state["step"]
        tokens_seen = state["tokens_seen"]
        log.info("resumed_from_checkpoint", path=str(train_config.resume_from), step=step)

    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    checkpoints_dir = output_dir / "checkpoints"
    metrics_file = open(metrics_path, "a", encoding="utf-8")

    use_bf16 = train_config.precision == "bf16" and device.type == "cuda"
    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16)

    history: list[dict[str, Any]] = []
    train_iter = iter(train_loader)
    model.train()
    train_start = time.time()
    last_train_loss = float("nan")

    while step < train_config.max_steps:
        step_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        data_time = 0.0

        for micro_step in range(train_config.gradient_accumulation_steps):
            t0 = time.time()
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            data_time += time.time() - t0

            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast_ctx:
                _, loss = model(inputs, targets)
                loss = loss / train_config.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
            tokens_seen += inputs.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.grad_clip
        ).item()
        optimizer.step()
        scheduler.step()

        last_train_loss = accumulated_loss
        warnings = detector.check(accumulated_loss, grad_norm, step)
        for w in warnings:
            log.warning("training_anomaly_warning", message=w)

        step_time = time.time() - step_start
        tokens_per_sec = inputs.numel() * train_config.gradient_accumulation_steps / max(
            step_time, 1e-9
        )

        record = {
            "step": step,
            "loss": round(accumulated_loss, 6),
            "learning_rate": scheduler.get_last_lr()[0],
            "gradient_norm": round(grad_norm, 6),
            "tokens_seen": tokens_seen,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "step_time_sec": round(step_time, 4),
            "data_loading_time_sec": round(data_time, 4),
        }

        if val_loader is not None and (step % train_config.eval_every == 0 or step == train_config.max_steps - 1):
            record["val_loss"] = evaluate_loss(model, val_loader, device)
            model.train()

        history.append(record)
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

        if step % train_config.log_every == 0:
            log.info("train_step", **record)

        if train_config.checkpoint_every and step > 0 and step % train_config.checkpoint_every == 0:
            ckpt_path = checkpoints_dir / f"step_{step}.pt"
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler, step, tokens_seen,
                config_to_json({"model_config": model_config, "train_config": train_config}),
            )

        step += 1

    total_time = time.time() - train_start
    metrics_file.close()

    final_val_loss = None
    if val_loader is not None:
        final_val_loss = evaluate_loss(model, val_loader, device)

    final_ckpt = checkpoints_dir / "final.pt"
    save_checkpoint(
        final_ckpt, model, optimizer, scheduler, step, tokens_seen,
        config_to_json({"model_config": model_config, "train_config": train_config}),
    )

    mean_tps = tokens_seen / max(total_time, 1e-9)
    result = TrainingResult(
        final_step=step,
        final_train_loss=last_train_loss,
        final_val_loss=final_val_loss,
        tokens_seen=tokens_seen,
        total_train_time_sec=round(total_time, 3),
        mean_tokens_per_sec=round(mean_tps, 2),
        history=history,
    )
    (output_dir / "training_result.json").write_text(json.dumps(result.to_dict(), indent=2))
    log.info("training_complete", **{k: v for k, v in result.to_dict().items() if k != "history"})
    return result


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, loss = model(inputs, targets)
        total_loss += loss.item()
        total_batches += 1
    return round(total_loss / max(total_batches, 1), 6)
