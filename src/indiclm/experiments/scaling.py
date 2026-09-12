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
from dataclasses import dataclass, field
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
    per_language_loss: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "n_params": self.n_params,
            "n_params_non_embedding": self.n_params_non_embedding,
            "d_tokens": self.d_tokens,
            "final_val_loss": self.final_val_loss,
            "mean_tokens_per_sec": self.mean_tokens_per_sec,
            "seed": self.seed,
            "per_language_loss": self.per_language_loss,
        }


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


def _fit_two_param(
    n: np.ndarray, d: np.ndarray, loss: np.ndarray
) -> dict[str, Any]:
    """2-parameter Chinchilla-style fit: L ≈ C / (N·D)^gamma.

    Log-linearises to log(L) = log(C) - gamma * log(N·D), then uses
    ordinary least squares. Cheaper than curve_fit and interpretable as a
    single scaling exponent over the compute budget N·D.
    """
    try:
        log_nd = np.log(n * d)
        log_loss = np.log(np.clip(loss, 1e-9, None))
        coeffs = np.polyfit(log_nd, log_loss, 1)
        gamma, log_c = -float(coeffs[0]), float(coeffs[1])
        predicted = np.exp(log_c) / (n * d) ** gamma
        ss_res = float(np.sum((loss - predicted) ** 2))
        ss_tot = float(np.sum((loss - loss.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {
            "fit_status": "ok",
            "C": float(np.exp(log_c)),
            "gamma": gamma,
            "r_squared": r_squared,
            "note": "L ≈ C / (N·D)^gamma — 2-parameter Chinchilla-style fit",
        }
    except Exception as e:  # noqa: BLE001
        return {"fit_status": "fit_failed", "note": str(e)}


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

    # 2-parameter Chinchilla-style fit: L ≈ C / (N·D)^gamma
    two_param_fit = _fit_two_param(n, d, loss)

    result = {
        **free_fit,
        "n_observations": len(observations),
        "fit_free_linf": free_fit,
        "fit_fixed_linf": fixed_fit,
        "fit_two_param": two_param_fit,
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

        per_language_loss: dict[str, float] = {}
        final_ckpt = run_dir / "checkpoints" / "final.pt"
        if final_ckpt.exists():
            try:
                from indiclm.evaluation.perplexity import evaluate_checkpoint as _eval_ckpt
                eval_rep = _eval_ckpt(
                    checkpoint_path=final_ckpt,
                    shards_dir=Path(data_cfg["shards_dir"]),
                    tokenizer_path=Path(data_cfg["tokenizer_path"]),
                    seq_len=data_cfg["seq_len"],
                    batch_size=batch_size,
                )
                per_language_loss = {
                    lang: lr.loss for lang, lr in eval_rep.per_language.items()
                }
            except Exception as exc:  # noqa: BLE001
                log.warning("per_language_eval_failed", run_id=run_id, error=str(exc))

        obs = ScalingObservation(
            run_id=run_id,
            n_params=n_params,
            n_params_non_embedding=n_params_non_embed,
            d_tokens=result.tokens_seen,
            final_val_loss=result.final_val_loss or result.final_train_loss,
            mean_tokens_per_sec=result.mean_tokens_per_sec,
            seed=seed,
            per_language_loss=per_language_loss,
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


def backfill_per_language_losses(
    observations: list[ScalingObservation],
    run_dirs: list[Path],
    shards_dir: Path,
    tokenizer_path: Path,
    seq_len: int,
    batch_size: int = 4,
) -> list[ScalingObservation]:
    """Evaluates perplexity per language for any observation that has an
    empty `per_language_loss`, using the checkpoint in the matching run_dir.
    Returns the same list with the field populated in-place."""
    from indiclm.evaluation.perplexity import evaluate_checkpoint as _eval_ckpt

    run_dir_by_id = {d.name: d for d in run_dirs}

    for obs in observations:
        if obs.per_language_loss:
            continue
        run_dir = run_dir_by_id.get(obs.run_id)
        if run_dir is None:
            log.warning("backfill_no_run_dir", run_id=obs.run_id)
            continue
        ckpt = run_dir / "checkpoints" / "final.pt"
        if not ckpt.exists():
            log.warning("backfill_no_checkpoint", run_id=obs.run_id, path=str(ckpt))
            continue
        try:
            eval_rep = _eval_ckpt(
                checkpoint_path=ckpt,
                shards_dir=shards_dir,
                tokenizer_path=tokenizer_path,
                seq_len=seq_len,
                batch_size=batch_size,
            )
            obs.per_language_loss = {lang: lr.loss for lang, lr in eval_rep.per_language.items()}
            log.info("backfill_done", run_id=obs.run_id, languages=list(obs.per_language_loss))
        except Exception as exc:  # noqa: BLE001
            log.warning("backfill_eval_failed", run_id=obs.run_id, error=str(exc))

    return observations


def fit_per_language_scaling_laws(
    observations: list[ScalingObservation],
) -> dict[str, Any]:
    """Fits a 2-param Chinchilla-style L ≈ C/(N·D)^gamma independently
    per language. Languages with fewer than 3 observations (or no data)
    are skipped with an explicit note. Returns a dict of:
        language → fit_result (same schema as fit_two_param).
    Also includes summary keys: all_languages, languages_fit, languages_skipped.
    """
    by_language: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for obs in observations:
        for lang, loss in obs.per_language_loss.items():
            if lang not in by_language:
                by_language[lang] = ([], [], [])
            by_language[lang][0].append(float(obs.n_params_non_embedding))
            by_language[lang][1].append(float(obs.d_tokens))
            by_language[lang][2].append(float(loss))

    fits: dict[str, Any] = {}
    languages_fit: list[str] = []
    languages_skipped: list[str] = []

    for lang in sorted(by_language):
        ns, ds, losses = by_language[lang]
        if len(ns) < 3:
            fits[lang] = {
                "fit_status": "insufficient_data",
                "n_observations": len(ns),
                "note": f"Only {len(ns)} observations; need ≥3 for 2-param fit.",
            }
            languages_skipped.append(lang)
            continue
        n_arr = np.array(ns, dtype=float)
        d_arr = np.array(ds, dtype=float)
        loss_arr = np.array(losses, dtype=float)
        result = _fit_two_param(n_arr, d_arr, loss_arr)
        result["n_observations"] = len(ns)
        result["loss_min"] = float(loss_arr.min())
        result["loss_max"] = float(loss_arr.max())
        result["loss_mean"] = float(loss_arr.mean())
        fits[lang] = result
        if result["fit_status"] == "ok":
            languages_fit.append(lang)
        else:
            languages_skipped.append(lang)

    return {
        "per_language": fits,
        "all_languages": sorted(by_language),
        "languages_fit": languages_fit,
        "languages_skipped": languages_skipped,
        "n_observations_total": len(observations),
    }


def plot_per_language_scaling(
    observations: list[ScalingObservation],
    per_language_fit: dict[str, Any],
    out_path: Path,
) -> None:
    """Loss vs. compute (N·D) scatter coloured by language, with one
    fitted line per language that converged."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    languages = sorted({
        lang for obs in observations for lang in obs.per_language_loss
    })
    if not languages:
        return

    palette = [
        "#4C6EF5", "#F76707", "#0CA678", "#E64980",
        "#7048E8", "#F59F00", "#1098AD", "#E03131",
        "#099268", "#862E9C",
    ]
    lang_color = {lang: palette[i % len(palette)] for i, lang in enumerate(languages)}

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for lang in languages:
        xs, ys = [], []
        for obs in observations:
            if lang in obs.per_language_loss:
                xs.append(obs.n_params_non_embedding * obs.d_tokens)
                ys.append(obs.per_language_loss[lang])
        if not xs:
            continue
        color = lang_color[lang]
        ax.scatter(xs, ys, color=color, alpha=0.55, s=22, zorder=3)

        # Fitted line
        fit = per_language_fit.get("per_language", {}).get(lang, {})
        if fit.get("fit_status") == "ok":
            x_grid = np.logspace(np.log10(min(xs)), np.log10(max(xs)), 80)
            y_pred = fit["C"] / x_grid ** fit["gamma"]
            ax.plot(x_grid, y_pred, color=color, linewidth=1.5, label=lang)
        else:
            ax.scatter([], [], color=color, label=lang)

    ax.set_xscale("log")
    ax.set_xlabel("Compute budget  N · D  (params × tokens)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Per-language scaling: loss vs. compute budget")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
