"""Cosine learning-rate decay with linear warmup."""

from __future__ import annotations

import math

import torch


def cosine_with_warmup_lr(
    step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr_ratio: float = 0.1
) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    if step >= total_steps:
        return max_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max_lr * min_lr_ratio + coeff * max_lr * (1 - min_lr_ratio)


class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            if step >= total_steps:
                return min_lr_ratio
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + coeff * (1 - min_lr_ratio)

        super().__init__(optimizer, lr_lambda)
