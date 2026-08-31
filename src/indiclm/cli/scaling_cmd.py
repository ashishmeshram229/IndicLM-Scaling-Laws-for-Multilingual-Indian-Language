"""`indiclm experiment scaling-sweep`: runs the model-size x token-budget
grid, fits the empirical scaling law, and writes individual per-size
manifests (EXP-001/002/003 baselines) plus the full grid + fit under
EXP-012 (compute-optimal scaling experiment).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from indiclm.experiments.manifest import build_manifest, write_manifest
from indiclm.experiments.scaling import (
    ScalingObservation,
    aggregate_by_grid_point,
    fit_scaling_law,
    plot_scaling_curves,
    run_scaling_sweep,
)
from indiclm.utils.logging import configure_logging

app = typer.Typer(help="Model-size x token-budget scaling sweep (EXP-001/002/003/012).")
console = Console()


@app.command("scaling-sweep")
def scaling_sweep(
    shards_dir: Path = typer.Option(Path("data/processed")),
    tokenizer_path: Path = typer.Option(Path("data/tokenizer_v1/indiclm_tokenizer.model")),
    out_dir: Path = typer.Option(Path("experiments/manifests/EXP-012")),
    seq_len: int = typer.Option(64),
    seeds: list[int] = typer.Option(
        [0, 1, 2],
        help=(
            "One run per (model size, token budget, seed). Multiple seeds "
            "let the scaling-law fit report actual measurement noise per "
            "grid point (see aggregate_by_grid_point), not just the fit's "
            "own parameter uncertainty -- pass a single seed to reproduce "
            "the original single-seed behavior."
        ),
    ),
) -> None:
    configure_logging()

    model_sizes = [
        {"run_id": "n_tiny", "d_model": 32, "n_layers": 2, "n_heads": 2, "n_kv_heads": 1},
        {"run_id": "n_small", "d_model": 48, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2},
        {"run_id": "n_medium", "d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2},
        {"run_id": "n_large", "d_model": 96, "n_layers": 3, "n_heads": 4, "n_kv_heads": 2},
    ]
    # D (training tokens actually consumed) = max_steps * micro_batch_size *
    # gradient_accumulation_steps * seq_len. Varying the *dataset's* pool
    # size does NOT vary D on its own, since the training loop cycles the
    # dataloader for exactly max_steps regardless of pool size — so D is
    # controlled here via max_steps, not via data_cfg.total_tokens.
    micro_batch_size, grad_accum = 4, 2
    tokens_per_step = micro_batch_size * grad_accum * seq_len
    target_token_budgets = [8000, 24000]

    data_cfg = {
        "shards_dir": str(shards_dir), "tokenizer_path": str(tokenizer_path),
        "seq_len": seq_len, "total_tokens": 20000, "alpha": 0.7,
    }

    all_observations: list[ScalingObservation] = []
    for d_tokens_target in target_token_budgets:
        max_steps = max(1, round(d_tokens_target / tokens_per_step))
        for seed in seeds:
            obs = run_scaling_sweep(
                model_sizes=[
                    {**m, "run_id": f"{m['run_id']}_d{d_tokens_target}_seed{seed}"} for m in model_sizes
                ],
                data_cfg=data_cfg,
                train_cfg_overrides={
                    "max_steps": max_steps, "micro_batch_size": micro_batch_size,
                    "gradient_accumulation_steps": grad_accum,
                    "warmup_steps": max(2, max_steps // 6), "eval_every": max(5, max_steps // 3),
                    "checkpoint_every": max_steps, "log_every": max(5, max_steps // 3),
                },
                out_dir=out_dir / f"tokens_{d_tokens_target}" / f"seed_{seed}",
                seed=seed,
            )
            all_observations.extend(obs)

    # Grouped by (N, D) across seeds *before* fitting, so the fit is
    # judged against per-point measurement noise, not just its own
    # parameter covariance -- see aggregate_by_grid_point's docstring.
    grid_aggregation = aggregate_by_grid_point(all_observations)
    (out_dir / "seed_aggregation.json").write_text(json.dumps(grid_aggregation, indent=2))

    fit = fit_scaling_law(all_observations)
    fit["seeds_used"] = seeds
    fit["grid_point_aggregation"] = grid_aggregation
    (out_dir / "scaling_law_fit.json").write_text(json.dumps(fit, indent=2))
    plot_scaling_curves(all_observations, fit, out_dir / "loss_vs_params.png")

    manifest = build_manifest(
        experiment_id="EXP-012",
        config={
            "model_sizes": model_sizes, "target_token_budgets": target_token_budgets, "seeds": seeds,
        },
        dataset_version="v1", tokenizer_version="bpe_v1", seed=seeds[0],
    )
    manifest.evaluation_metrics = fit
    write_manifest(manifest, out_dir)

    # EXP-001/002/003: the three canonical baseline sizes at the larger
    # token budget, each gets its own manifest pointing back at the shared
    # sweep run for provenance. final_val_loss is the mean across seeds,
    # with per-seed spread recorded in evaluation_metrics so a reviewer
    # can see how much a single-seed number could have varied.
    baseline_map = {"EXP-001": "n_tiny", "EXP-002": "n_small", "EXP-003": "n_medium"}
    max_d = max(target_token_budgets)
    for exp_id, run_id in baseline_map.items():
        matched_runs = [
            o for o in all_observations
            if o.run_id.startswith(f"{run_id}_d{max_d}_seed") and o.d_tokens > 0
        ]
        if not matched_runs:
            continue
        losses = [o.final_val_loss for o in matched_runs]
        mean_loss = sum(losses) / len(losses)
        std_loss = (
            (sum((loss_val - mean_loss) ** 2 for loss_val in losses) / (len(losses) - 1)) ** 0.5
            if len(losses) > 1
            else 0.0
        )
        m = build_manifest(
            experiment_id=exp_id,
            config={"model_size": run_id, "token_budget": max_d, "source_sweep": "EXP-012", "seeds": seeds},
            dataset_version="v1", tokenizer_version="bpe_v1", seed=seeds[0],
        )
        m.training_tokens = matched_runs[0].d_tokens
        m.final_val_loss = mean_loss
        m.evaluation_metrics = {
            "final_val_loss_mean": mean_loss,
            "final_val_loss_std": std_loss,
            "final_val_loss_per_seed": {o.seed: o.final_val_loss for o in matched_runs},
            "n_seeds": len(matched_runs),
        }
        write_manifest(m, out_dir.parent / exp_id)

    console.print(f"[green]Scaling sweep complete.[/green] Fit status: {fit.get('fit_status')}")
    if fit.get("fit_status") == "ok":
        console.print(f"alpha={fit['alpha']:.4f} +/- {fit['alpha_stderr']:.4f}, R^2={fit['r_squared']:.4f}")
    console.print(f"Seeds used: {seeds}")
    console.print(f"Plot: {out_dir / 'loss_vs_params.png'}")
