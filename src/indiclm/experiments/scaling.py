"""Scaling-law experiment runner: sweeps model size (N) and/or token
budget (D), records validation loss (L), and fits the empirical relation

    L(N, D) ≈ A / N^alpha + B / D^beta + L_infinity

via nonlinear least squares (scipy.optimize.curve_fit). This is treated
as an empirical fit to whatever data points were actually run — not
asserted as a universal law — and includes parameter uncertainty
(standard errors from the covariance matrix) and the underlying
(N, D, L) observations, so a reviewer can judge the fit's honesty
themselves.

At this project's toy corpus scale, the resulting alpha/beta estimates
are a demonstration of the *methodology*, not a scientific claim about
Indic-language scaling laws — this is stated explicitly in every report
this module produces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader, random_split

from indiclm.models.config import ModelConfig
from indiclm.training.dataset import PackedTokenDataset
from indiclm.training.trainer import TrainingConfig, train
from indiclm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ScalingObservation:
    run_id: str
    n_params: int
    n_params_non_embedding: int
    d_tokens: int
    final_val_loss: float
    mean_tokens_per_sec: float
    seed: int = 0

    def to_dict(self) -> dict:
        return self.__dict__


def _scaling_law(
    nd: tuple[np.ndarray, np.ndarray], log_a: float, alpha: float, log_b: float, beta: float, l_inf: float, /
) -> np.ndarray:
    n, d = nd
    return np.exp(log_a) / (n**alpha) + np.exp(log_b) / (d**beta) + l_inf


def _make_scaling_law_fixed_linf(
    l_inf: float,
) -> Any:
    """Returns a 4-parameter version of _scaling_law with L_inf fixed."""
    def _fn(nd: tuple[np.ndarray, np.ndarray], log_a: float, alpha: float, log_b: float, beta: float) -> np.ndarray:
        n, d = nd
        return np.exp(log_a) / (n**alpha) + np.exp(log_b) / (d**beta) + l_inf
    return _fn


def _fit_fixed_linf(
    n: np.ndarray, d: np.ndarray, loss: np.ndarray, l_inf: float
) -> dict[str, Any]:
    """4-parameter refit with L_inf fixed to l_inf. Returns a sub-dict."""
    try:
        popt, pcov = curve_fit(
            _make_scaling_law_fixed_linf(l_inf), (n, d), loss,
            p0=[0.0, 0.3, 0.0, 0.3],
            bounds=([-10.0, 1e-3, -10.0, 1e-3], [30.0, 2.0, 30.0, 2.0]),
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
        log_a, alpha, log_b, beta = popt
        residuals = loss - _make_scaling_law_fixed_linf(l_inf)((n, d), *popt)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((loss - loss.mean()) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {
            "fit_status": "ok",
            "L_infinity_fixed": l_inf,
            "A": float(np.exp(log_a)),
            "alpha": float(alpha),
            "alpha_stderr": float(perr[1]),
            "B": float(np.exp(log_b)),
            "beta": float(beta),
            "beta_stderr": float(perr[3]),
            "r_squared": r_squared,
        }
    except RuntimeError as e:
        return {"fit_status": "fit_failed", "note": f"curve_fit did not converge: {e}"}


def fit_scaling_law(observations: list[ScalingObservation]) -> dict[str, Any]:
    """Fits L(N,D) via nonlinear least squares. Requires at least 5
    observations (5 free parameters) to be identifiable; with fewer, we
    report an honest "insufficient data" result rather than a fit.

    Always runs two fits and reports both:
    - 5-parameter free fit (L_inf free)
    - 4-parameter fixed fit (L_inf pinned to 0.99 × observed minimum),
      which tightens alpha/beta uncertainty at toy corpus scales where the
      asymptote is unidentifiable from the data alone.
    """
    if len(observations) < 5:
        return {
            "fit_status": "insufficient_data",
            "note": (
                f"Only {len(observations)} observations available; at least 5 are needed to "
                "fit the 5-parameter L(N,D) = A/N^alpha + B/D^beta + L_inf model. "
                "Reporting raw observations only."
            ),
            "observations": [o.to_dict() for o in observations],
        }

    n = np.array([o.n_params_non_embedding for o in observations], dtype=float)
    d = np.array([o.d_tokens for o in observations], dtype=float)
    loss = np.array([o.final_val_loss for o in observations], dtype=float)

    # 5-parameter free fit
    free_fit: dict[str, Any]
    try:
        # scipy-stubs models curve_fit's xdata as a single 1-D array; it
        # doesn't capture the (also-supported, and used here) multi-dimensional
        # xdata case of a tuple of arrays, so this is a stub gap, not a bug.
        popt, pcov = curve_fit(
            _scaling_law, (n, d), loss,  # type: ignore[arg-type]
            p0=[0.0, 0.3, 0.0, 0.3, min(loss) * 0.5],
            bounds=(
                [-10.0, 1e-3, -10.0, 1e-3, 0.0],
                [30.0, 2.0, 30.0, 2.0, max(min(loss) * 0.99, 1e-3)],
            ),
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
        log_a, alpha, log_b, beta, l_inf = popt
        residuals = loss - _scaling_law((n, d), *popt)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((loss - loss.mean()) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        free_fit = {
            "fit_status": "ok",
            "A": float(np.exp(log_a)),
            "alpha": float(alpha),
            "alpha_stderr": float(perr[1]),
            "B": float(np.exp(log_b)),
            "beta": float(beta),
            "beta_stderr": float(perr[3]),
            "L_infinity": float(l_inf),
            "L_infinity_stderr": float(perr[4]),
            "r_squared": r_squared,
        }
    except RuntimeError as e:
        free_fit = {"fit_status": "fit_failed", "note": f"curve_fit did not converge: {e}"}

    # 4-parameter fixed-L_inf fit (L_inf = 0.99 × observed minimum loss)
    l_inf_fixed = float(np.min(loss)) * 0.99
    fixed_fit = _fit_fixed_linf(n, d, loss, l_inf_fixed)

    result = {
        **free_fit,
        "n_observations": len(observations),
        "fit_free_linf": free_fit,
        "fit_fixed_linf": fixed_fit,
        "observations": [o.to_dict() for o in observations],
    }
    # Promote fit_status from free fit for backward compatibility
    result["fit_status"] = free_fit.get("fit_status", "fit_failed")
    return result


def run_scaling_sweep(
    model_sizes: list[dict[str, Any]],
    data_cfg: dict[str, Any],
    train_cfg_overrides: dict[str, Any],
    out_dir: Path,
    seed: int = 0,
) -> list[ScalingObservation]:
    """Trains one tiny model per entry in `model_sizes` (each a partial
    ModelConfig kwargs dict, e.g. {"d_model": 64, "n_layers": 2, ...}) on
    the same token budget, and records (N, D, L) triples."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    observations: list[ScalingObservation] = []

    dataset = PackedTokenDataset(
        shards_dir=Path(data_cfg["shards_dir"]),
        tokenizer_path=Path(data_cfg["tokenizer_path"]),
        seq_len=data_cfg["seq_len"],
        total_tokens=data_cfg["total_tokens"],
        alpha=data_cfg.get("alpha", 1.0),
        seed=seed,
    )
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    batch_size = train_cfg_overrides.get("micro_batch_size", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    for i, size_cfg in enumerate(model_sizes):
        run_id = size_cfg.get("run_id", f"size_{i}")
        model_config = ModelConfig(
            vocab_size=dataset.sp.get_piece_size(),
            max_seq_len=data_cfg["seq_len"],
            **{k: v for k, v in size_cfg.items() if k != "run_id"},
        )
        run_dir = out_dir / run_id
        train_cfg = TrainingConfig(output_dir=run_dir, seed=seed, **train_cfg_overrides)
        log.info(
            "scaling_sweep_run_start", run_id=run_id,
            n_params=model_config.num_parameters_estimate(),
        )
        result = train(model_config, train_cfg, train_loader, val_loader)

        # Instantiate once more to get the *actual* (non-embedding) param
        # count from the real module tree, not just the analytic estimate.
        from indiclm.models.transformer import DecoderOnlyTransformer

        actual_model = DecoderOnlyTransformer(model_config)
        n_params = actual_model.num_parameters()
        n_params_non_embed = actual_model.num_parameters(non_embedding=True)

        obs = ScalingObservation(
            run_id=run_id,
            n_params=n_params,
            n_params_non_embedding=n_params_non_embed,
            d_tokens=result.tokens_seen,
            final_val_loss=result.final_val_loss or result.final_train_loss,
            mean_tokens_per_sec=result.mean_tokens_per_sec,
            seed=seed,
        )
        observations.append(obs)

    (out_dir / "observations.json").write_text(
        json.dumps([o.to_dict() for o in observations], indent=2)
    )
    return observations


def aggregate_by_grid_point(observations: list[ScalingObservation]) -> list[dict[str, Any]]:
    """Groups multi-seed observations by grid point (same N, D) and
    reports mean/std/stderr of final_val_loss across seeds, so the
    scaling-law fit's honesty can be judged not just from the fit's own
    parameter uncertainty but from how noisy the underlying measurements
    actually are at each point. Grouped by (n_params_non_embedding,
    d_tokens) rather than by `run_id` string, since `run_id` embeds the
    seed and differs per observation."""
    groups: dict[tuple[int, int], list[ScalingObservation]] = {}
    for o in observations:
        key = (o.n_params_non_embedding, o.d_tokens)
        groups.setdefault(key, []).append(o)

    result = []
    for (n_params, d_tokens), obs_list in sorted(groups.items()):
        losses = np.array([o.final_val_loss for o in obs_list])
        result.append(
            {
                "n_params_non_embedding": n_params,
                "d_tokens": d_tokens,
                "n_seeds": len(obs_list),
                "seeds": [o.seed for o in obs_list],
                "final_val_loss_mean": float(losses.mean()),
                "final_val_loss_std": float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
                "final_val_loss_values": [float(loss_val) for loss_val in losses],
            }
        )
    return result


def plot_scaling_curves(observations: list[ScalingObservation], fit: dict[str, Any], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [o.n_params_non_embedding for o in observations]
    loss = [o.final_val_loss for o in observations]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    ax.scatter(n, loss, color="#2f6fed", zorder=3, label="measured runs")
    ax.set_xscale("log")
    ax.set_xlabel("Non-embedding parameters (N)")
    ax.set_ylabel("Final validation loss (L)")
    ax.set_title("IndicLM scaling sweep: loss vs. model size")
    ax.grid(True, which="both", alpha=0.3)

    if fit.get("fit_status") == "ok":
        n_grid = np.logspace(np.log10(min(n)), np.log10(max(n)), 100)
        d_fixed = np.mean([o.d_tokens for o in observations])
        l_pred = fit["A"] / n_grid ** fit["alpha"] + fit["B"] / d_fixed ** fit["beta"] + fit["L_infinity"]
        ax.plot(n_grid, l_pred, color="#d1495b", linestyle="--", label=f"fit (α={fit['alpha']:.3f})")

    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
