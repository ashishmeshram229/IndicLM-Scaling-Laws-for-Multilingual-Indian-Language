"""`indiclm experiment ...` subcommands."""

from __future__ import annotations

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
def run(config: Path = typer.Option(..., help="Path to an experiment YAML config.")) -> None:
    configure_logging()
    cfg = yaml.safe_load(Path(config).read_text())
    result = run_experiment(cfg)
    console.print(f"[green]Experiment {result['experiment_id']} complete.[/green]")
    console.print(f"Final val loss: {result['result']['final_val_loss']}")
    console.print(f"Manifest: {result['manifest_path']}")


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
