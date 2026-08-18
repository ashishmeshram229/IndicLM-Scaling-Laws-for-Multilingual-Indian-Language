"""Unit tests for the LR scheduler and checkpoint save/load round-trip."""

from __future__ import annotations

from pathlib import Path

import torch

from indiclm.models.config import ModelConfig
from indiclm.models.transformer import DecoderOnlyTransformer
from indiclm.training.checkpoint import load_checkpoint, save_checkpoint
from indiclm.training.scheduler import CosineWarmupScheduler


def test_scheduler_warmup_then_decay() -> None:
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = CosineWarmupScheduler(opt, warmup_steps=10, total_steps=100)

    lrs = []
    for _ in range(100):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()

    assert lrs[0] < lrs[9]  # warming up
    assert lrs[9] > lrs[-1]  # decayed by the end
    assert min(lrs) >= 0


def test_checkpoint_roundtrip_restores_weights_and_step(tmp_path: Path) -> None:
    cfg = ModelConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=2)
    model = DecoderOnlyTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=5, total_steps=50)

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model, optimizer, scheduler, step=7, tokens_seen=1234, config={"x": 1})

    model2 = DecoderOnlyTransformer(cfg)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = CosineWarmupScheduler(optimizer2, warmup_steps=5, total_steps=50)
    state = load_checkpoint(ckpt_path, model2, optimizer2, scheduler2)

    assert state["step"] == 7
    assert state["tokens_seen"] == 1234
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
