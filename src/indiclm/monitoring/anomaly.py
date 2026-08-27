"""Lightweight training anomaly detection: NaN/Inf loss, exploding
gradients, and sudden loss spikes. Raises `TrainingAnomaly` so the
training loop can fail fast and loudly rather than silently continuing
with corrupted state.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


class TrainingAnomaly(RuntimeError):
    pass


@dataclass
class AnomalyDetector:
    grad_norm_threshold: float = 100.0
    loss_spike_multiplier: float = 5.0
    window_size: int = 20

    def __post_init__(self) -> None:
        self._recent_losses: deque[float] = deque(maxlen=self.window_size)

    def check(self, loss: float, grad_norm: float, step: int) -> list[str]:
        warnings: list[str] = []
        if math.isnan(loss):
            raise TrainingAnomaly(f"step {step}: loss is NaN")
        if loss in (float("inf"), float("-inf")):
            raise TrainingAnomaly(f"step {step}: loss is Inf")
        if math.isnan(grad_norm) or grad_norm in (float("inf"), float("-inf")):
            raise TrainingAnomaly(f"step {step}: gradient norm is NaN/Inf")
        if grad_norm > self.grad_norm_threshold:
            warnings.append(
                f"step {step}: gradient norm {grad_norm:.2f} exceeds threshold "
                f"{self.grad_norm_threshold} (exploding-gradient warning, not fatal)"
            )
        if self._recent_losses:
            baseline = sum(self._recent_losses) / len(self._recent_losses)
            if baseline > 0 and loss > baseline * self.loss_spike_multiplier:
                warnings.append(
                    f"step {step}: loss {loss:.4f} is {loss / baseline:.1f}x the recent "
                    f"average {baseline:.4f} (possible loss spike)"
                )
        self._recent_losses.append(loss)
        return warnings
