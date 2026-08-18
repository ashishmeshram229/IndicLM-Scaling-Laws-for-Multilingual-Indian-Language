"""Unit tests for the data mixture engine's sampling math."""

from __future__ import annotations

import pytest

from indiclm.data.mixture import (
    AnnealingSchedule,
    capped_mixture,
    static_mixture,
    temperature_weights,
    token_budget_allocation,
)


def test_temperature_weights_sum_to_one() -> None:
    counts = {"eng": 1000, "hin": 500, "mar": 100}
    weights = temperature_weights(counts, alpha=1.0)
    assert pytest.approx(sum(weights.values()), abs=1e-9) == 1.0


def test_temperature_alpha_one_is_proportional() -> None:
    counts = {"eng": 1000, "hin": 500}
    weights = temperature_weights(counts, alpha=1.0)
    assert pytest.approx(weights["eng"], abs=1e-9) == 1000 / 1500
    assert pytest.approx(weights["hin"], abs=1e-9) == 500 / 1500


def test_temperature_alpha_zero_is_uniform() -> None:
    counts = {"eng": 1000, "hin": 500, "mar": 10}
    weights = temperature_weights(counts, alpha=0.0)
    for w in weights.values():
        assert pytest.approx(w, abs=1e-9) == 1 / 3


def test_temperature_zero_count_gets_zero_weight() -> None:
    counts = {"eng": 1000, "hin": 0}
    weights = temperature_weights(counts, alpha=0.5)
    assert weights["hin"] == 0.0
    assert weights["eng"] == 1.0


def test_static_mixture_normalizes() -> None:
    weights = static_mixture({"eng": 3, "hin": 1})
    assert pytest.approx(weights["eng"], abs=1e-9) == 0.75
    assert pytest.approx(weights["hin"], abs=1e-9) == 0.25


def test_static_mixture_rejects_nonpositive_total() -> None:
    with pytest.raises(ValueError):
        static_mixture({"eng": 0, "hin": 0})


def test_capped_mixture_respects_cap() -> None:
    weights = {"eng": 0.8, "hin": 0.1, "mar": 0.1}
    capped = capped_mixture(weights, cap=0.5)
    assert capped["eng"] <= 0.5 + 1e-9
    assert pytest.approx(sum(capped.values()), abs=1e-9) == 1.0


def test_token_budget_allocation_sums_exactly() -> None:
    weights = {"eng": 0.34, "hin": 0.33, "mar": 0.33}
    budget = token_budget_allocation(weights, total_tokens=1000)
    assert sum(budget.values()) == 1000


def test_annealing_schedule_interpolates() -> None:
    sched = AnnealingSchedule(start_alpha=1.0, end_alpha=0.0, total_steps=100)
    assert sched.alpha_at(0) == 1.0
    assert sched.alpha_at(100) == 0.0
    assert pytest.approx(sched.alpha_at(50), abs=1e-9) == 0.5
