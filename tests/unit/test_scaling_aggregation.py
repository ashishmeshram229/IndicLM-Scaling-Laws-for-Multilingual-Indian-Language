"""Unit tests for `aggregate_by_grid_point`, added alongside multi-seed
support for the scaling sweep. Pure function over `ScalingObservation`
records, so tested directly without any training."""

from __future__ import annotations

from indiclm.experiments.scaling import ScalingObservation, aggregate_by_grid_point


def _obs(n_params: int, d_tokens: int, loss: float, seed: int) -> ScalingObservation:
    return ScalingObservation(
        run_id=f"n{n_params}_d{d_tokens}_seed{seed}",
        n_params=n_params + 1000,
        n_params_non_embedding=n_params,
        d_tokens=d_tokens,
        final_val_loss=loss,
        mean_tokens_per_sec=1000.0,
        seed=seed,
    )


def test_aggregate_groups_by_n_and_d_across_seeds() -> None:
    observations = [
        _obs(100, 8000, 6.0, seed=0),
        _obs(100, 8000, 6.4, seed=1),
        _obs(100, 8000, 5.8, seed=2),
        _obs(200, 8000, 5.0, seed=0),
    ]
    agg = aggregate_by_grid_point(observations)
    assert len(agg) == 2  # two distinct (N, D) grid points

    point_100 = next(a for a in agg if a["n_params_non_embedding"] == 100)
    assert point_100["n_seeds"] == 3
    assert sorted(point_100["seeds"]) == [0, 1, 2]
    assert point_100["final_val_loss_mean"] == (6.0 + 6.4 + 5.8) / 3
    assert point_100["final_val_loss_std"] > 0

    point_200 = next(a for a in agg if a["n_params_non_embedding"] == 200)
    assert point_200["n_seeds"] == 1
    assert point_200["final_val_loss_std"] == 0.0  # single seed -> no spread to report


def test_aggregate_empty_input() -> None:
    assert aggregate_by_grid_point([]) == []
