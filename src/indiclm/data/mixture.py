"""Data mixture engine: language/source -> token-budget allocation.

Implements the temperature-sampling formulation:

    p_i = n_i^alpha / sum_j n_j^alpha

where `n_i` is the available token count for group `i` and `alpha` is the
mixture temperature. alpha=1.0 reproduces proportional (natural) sampling;
alpha=0.0 reproduces uniform sampling across groups regardless of size;
alpha towards +inf collapses onto the single largest group; a capped
mixture additionally clips any single group's share.

Static, capped, and curriculum/annealing (alpha varying over training)
schedules are all built on top of `temperature_weights`.
"""

from __future__ import annotations

from dataclasses import dataclass


def temperature_weights(counts: dict[str, int], alpha: float) -> dict[str, float]:
    """p_i = n_i^alpha / sum_j n_j^alpha. Groups with n_i == 0 get p_i == 0."""
    if not counts:
        return {}
    weighted = {k: (float(v) ** alpha if v > 0 else 0.0) for k, v in counts.items()}
    total = sum(weighted.values())
    if total == 0:
        n = len(counts)
        return {k: 1.0 / n for k in counts}
    return {k: v / total for k, v in weighted.items()}


def static_mixture(weights: dict[str, float]) -> dict[str, float]:
    """Validate and renormalize a user-specified static mixture."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Static mixture weights must sum to a positive value.")
    return {k: v / total for k, v in weights.items()}


def capped_mixture(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Clip any single group's share to `cap`, redistributing the excess
    proportionally among the remaining (uncapped) groups. Iterates because
    redistribution can push another group over the cap."""
    weights = dict(weights)
    for _ in range(len(weights)):
        over = {k: v for k, v in weights.items() if v > cap}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for k in over:
            weights[k] = cap
        remaining = {k: v for k, v in weights.items() if k not in over}
        remaining_total = sum(remaining.values())
        if remaining_total == 0:
            break
        for k, v in remaining.items():
            weights[k] += excess * (v / remaining_total)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


@dataclass
class AnnealingSchedule:
    """Linearly interpolates alpha from `start_alpha` to `end_alpha` over
    training, letting a mixture start closer to natural proportions and
    anneal toward (or away from) uniform low-resource oversampling."""

    start_alpha: float
    end_alpha: float
    total_steps: int

    def alpha_at(self, step: int) -> float:
        if self.total_steps <= 0:
            return self.end_alpha
        frac = min(max(step / self.total_steps, 0.0), 1.0)
        return self.start_alpha + frac * (self.end_alpha - self.start_alpha)


def token_budget_allocation(weights: dict[str, float], total_tokens: int) -> dict[str, int]:
    """Convert a normalized mixture (weights summing to ~1.0) into an
    integer token budget per group, summing exactly to `total_tokens`."""
    raw = {k: v * total_tokens for k, v in weights.items()}
    floored = {k: int(v) for k, v in raw.items()}
    remainder = total_tokens - sum(floored.values())
    # Distribute leftover tokens (from flooring) to the largest fractional
    # remainders first, so the allocation sums exactly.
    fractional = sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)
    for k, _ in fractional[:remainder]:
        floored[k] += 1
    return floored
