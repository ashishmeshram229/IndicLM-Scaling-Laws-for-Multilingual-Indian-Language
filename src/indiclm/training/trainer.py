"""Training engine: gradient accumulation, AdamW, cosine decay with warmup,
gradient clipping, fp16 AMP (V100-safe, no bf16), DDP, checkpointing/resume,
periodic evaluation, W&B logging, and structured per-step metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from indiclm.distributed import get_local_rank, is_main_process, wrap_ddp
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
    # V100 is compute capability 7.0 — supports fp16/AMP but NOT bf16.
    # "fp16" activates torch.amp.autocast + GradScaler on CUDA; "fp32" is CPU-safe.
    precision: str = "fp32"  # "fp32" | "fp16"
    seed: int = 0
    resume_from: Path | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None


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


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, stripping DDP wrapper if present."""
    return getattr(model, "module", model)


def train(
    model_config: ModelConfig,
    train_config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
) -> TrainingResult:
    torch.manual_seed(train_config.seed)
    device = torch.device(train_config.device)

    model = DecoderOnlyTransformer(model_config).to(device)

    # DDP: wrap if WORLD_SIZE > 1 (set by torchrun / SLURM srun --ntasks-per-node)
    local_rank = get_local_rank()
    device_ids = [local_rank] if device.type == "cuda" else None
    model = wrap_ddp(model, device_ids=device_ids)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=train_config.warmup_steps, total_steps=train_config.max_steps
    )
    detector = AnomalyDetector()

    # fp16 AMP — GradScaler is a no-op when enabled=False (fp32 / CPU path)
    use_fp16 = train_config.precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    autocast_ctx = torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_fp16)

    # W&B (optional — only initialised on rank 0 to avoid duplicate runs)
    _wandb = None
    if train_config.wandb_project and is_main_process():
        try:
            import wandb as _wb
            _wandb = _wb
            _wb.init(
                project=train_config.wandb_project,
                name=train_config.wandb_run_name,
                config={
                    **{k: str(v) for k, v in model_config.__dict__.items()},
                    **{k: str(v) for k, v in train_config.__dict__.items()},
                },
            )
        except ImportError:
            log.warning("wandb_not_installed", note="pip install wandb to enable W&B logging")

    step = 0
    tokens_seen = 0
    if train_config.resume_from is not None and Path(train_config.resume_from).exists():
        state = load_checkpoint(
            train_config.resume_from, _unwrap(model), optimizer, scheduler, scaler=scaler
        )
        step = state["step"]
        tokens_seen = state["tokens_seen"]
        log.info("resumed_from_checkpoint", path=str(train_config.resume_from), step=step)

    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    checkpoints_dir = output_dir / "checkpoints"

    # Opened for the whole run (appended per logged step, closed at end)
    metrics_file = open(metrics_path, "a", encoding="utf-8")  # noqa: SIM115

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

        for _ in range(train_config.gradient_accumulation_steps):
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
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            tokens_seen += inputs.numel()

        # Unscale before clipping so clip operates on true gradient magnitudes
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.grad_clip
        ).item()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        last_train_loss = accumulated_loss
        warnings = detector.check(accumulated_loss, grad_norm, step)
        for w in warnings:
            log.warning("training_anomaly_warning", message=w)

        step_time = time.time() - step_start
        tokens_per_sec = inputs.numel() * train_config.gradient_accumulation_steps / max(
            step_time, 1e-9
        )

        record: dict[str, Any] = {
            "step": step,
            "loss": round(accumulated_loss, 6),
            "learning_rate": scheduler.get_last_lr()[0],
            "gradient_norm": round(grad_norm, 6),
            "tokens_seen": tokens_seen,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "step_time_sec": round(step_time, 4),
            "data_loading_time_sec": round(data_time, 4),
        }
        if use_fp16:
            record["loss_scale"] = scaler.get_scale()

        if val_loader is not None and (
            step % train_config.eval_every == 0 or step == train_config.max_steps - 1
        ):
            record["val_loss"] = evaluate_loss(model, val_loader, device)
            model.train()

        history.append(record)

        # Only rank 0 writes metrics/checkpoints to avoid duplicate I/O
        if is_main_process():
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()

        if step % train_config.log_every == 0:
            log.info("train_step", **record)

        if _wandb is not None and is_main_process():
            _wandb.log(record, step=step)
            if device.type == "cuda":
                _wandb.log({"gpu_memory_gb": torch.cuda.memory_allocated() / 1e9}, step=step)

        if (
            is_main_process()
            and train_config.checkpoint_every
            and step > 0
            and step % train_config.checkpoint_every == 0
        ):
            ckpt_path = checkpoints_dir / f"step_{step}.pt"
            save_checkpoint(
                ckpt_path,
                _unwrap(model),
                optimizer,
                scheduler,
                step,
                tokens_seen,
                config_to_json({"model_config": model_config, "train_config": train_config}),
                scaler=scaler,
            )

        step += 1

    total_time = time.time() - train_start
    metrics_file.close()

    final_val_loss = None
    if val_loader is not None:
        final_val_loss = evaluate_loss(model, val_loader, device)

    if is_main_process():
        final_ckpt = checkpoints_dir / "final.pt"
        save_checkpoint(
            final_ckpt,
            _unwrap(model),
            optimizer,
            scheduler,
            step,
            tokens_seen,
            config_to_json({"model_config": model_config, "train_config": train_config}),
            scaler=scaler,
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
    if is_main_process():
        (output_dir / "training_result.json").write_text(json.dumps(result.to_dict(), indent=2))
        log.info(
            "training_complete", **{k: v for k, v in result.to_dict().items() if k != "history"}
        )
    if _wandb is not None:
        _wandb.finish()
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
