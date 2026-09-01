"""`indiclm experiment ...` subcommands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from indiclm.experiments.runner import compare_experiments, run_experiment
from indiclm.utils.logging import configure_logging

app = typer.Typer(help="Run and compare experiments.")
console = Console()


@app.command()
def run(
    config: Path = typer.Option(..., help="Path to an experiment YAML config."),
    seeds: list[int] = typer.Option(
        [0],
        help=(
            "Seeds to run. Pass multiple (e.g. --seeds 0 --seeds 1 --seeds 2) to run "
            "the experiment once per seed and aggregate final_val_loss across seeds, "
            "writing a multi_seed_summary.json alongside the per-seed manifests."
        ),
    ),
    experiments_root: Path = typer.Option(Path("experiments/manifests")),
) -> None:
    configure_logging()
    cfg = yaml.safe_load(Path(config).read_text())
    exp_id = cfg["experiment_id"]

    if len(seeds) == 1:
        cfg["seed"] = seeds[0]
        result = run_experiment(cfg, experiments_root)
        console.print(f"[green]Experiment {result['experiment_id']} complete.[/green]")
        console.print(f"Final val loss: {result['result']['final_val_loss']}")
        console.print(f"Manifest: {result['manifest_path']}")
        return

    # Multi-seed run: each seed writes to experiments_root/exp_id/seed_{s}/
    base_dir = experiments_root / exp_id
    seed_results = []
    for seed in seeds:
        seed_cfg = {**cfg, "seed": seed}
        out_dir = base_dir / f"seed_{seed}"
        console.print(f"Running {exp_id} seed={seed} → {out_dir}")
        result = run_experiment(seed_cfg, experiments_root, out_dir=out_dir)
        seed_results.append((seed, result))

    losses = [r["result"]["final_val_loss"] for _, r in seed_results if r["result"].get("final_val_loss") is not None]
    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    std_loss = (
        (sum((lo - mean_loss) ** 2 for lo in losses) / (len(losses) - 1)) ** 0.5
        if len(losses) > 1 else 0.0
    )
    summary = {
        "experiment_id": exp_id,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "final_val_loss_mean": mean_loss,
        "final_val_loss_std": std_loss,
        "final_val_loss_per_seed": {s: r["result"]["final_val_loss"] for s, r in seed_results},
    }
    summary_path = base_dir / "multi_seed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    console.print(f"[green]{exp_id} multi-seed complete.[/green]  "
                  f"mean_loss={mean_loss:.4f}  std={std_loss:.4f}  seeds={seeds}")
    console.print(f"Summary: {summary_path}")


@app.command()
def compare(
    experiments: list[str] = typer.Option(..., "--experiments", help="Experiment IDs to compare."),
    experiments_root: Path = typer.Option(Path("experiments/manifests")),
) -> None:
    configure_logging()
    comparison = compare_experiments(experiments, experiments_root)
    table = Table(title="Experiment Comparison")
    table.add_column("Experiment")
    table.add_column("Status")
    table.add_column("Final val loss")
    table.add_column("Overall PPL")
    for exp_id, manifest in comparison.items():
        if manifest.get("status") == "not_run":
            table.add_row(exp_id, "not run", "-", "-")
        else:
            table.add_row(
                exp_id, "complete", str(manifest.get("final_val_loss")),
                str(manifest.get("evaluation_metrics", {}).get("overall_perplexity")),
            )
    console.print(table)


@app.command()
def list_experiments(experiments_root: Path = typer.Option(Path("experiments/configs"))) -> None:
    """List experiment configs registered under experiments/configs/."""
    for path in sorted(experiments_root.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        console.print(f"{cfg.get('experiment_id', path.stem)}: {cfg.get('hypothesis', '(no hypothesis)')[:80]}")
