"""`indiclm data ...` subcommands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from indiclm.data.pipeline import DataPipelineConfig, run_pipeline
from indiclm.utils.logging import configure_logging

app = typer.Typer(help="Data ingestion, cleaning, and mixture preparation.")
console = Console()


@app.command()
def prepare(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Directory of raw .txt sources."),
    output_dir: Path = typer.Option(Path("data/processed"), help="Where to write shards + stats."),
    dataset_version: str = typer.Option("v1"),
    min_quality_score: float = typer.Option(0.5),
    enable_quality_filter: bool = typer.Option(True),
    enable_exact_dedup: bool = typer.Option(True),
    enable_near_dedup: bool = typer.Option(True),
) -> None:
    """Run the full pipeline: ingest -> langid -> quality -> dedup -> shard."""
    configure_logging()
    cfg = DataPipelineConfig(
        raw_dir=raw_dir, output_dir=output_dir, dataset_version=dataset_version,
        min_quality_score=min_quality_score, enable_quality_filter=enable_quality_filter,
        enable_exact_dedup=enable_exact_dedup, enable_near_dedup=enable_near_dedup,
    )
    stats = run_pipeline(cfg)
    console.print(f"[green]Pipeline complete.[/green] Stats written to {output_dir}/pipeline_stats.json")
    console.print(stats.to_dict())


@app.command()
def stats(output_dir: Path = typer.Option(Path("data/processed"))) -> None:
    """Print the dataset statistics from the most recent `data prepare` run."""
    import json

    stats_path = output_dir / "pipeline_stats.json"
    if not stats_path.exists():
        console.print(f"[red]No stats found at {stats_path}. Run `indiclm data prepare` first.[/red]")
        raise typer.Exit(1)
    data = json.loads(stats_path.read_text())

    table = Table(title="Dataset Statistics")
    table.add_column("Metric")
    table.add_column("Value")
    for key in ["total_documents", "accepted_documents", "rejected_documents", "duplicate_rate", "near_duplicate_rate", "mean_quality_score"]:
        table.add_row(key, str(data.get(key)))
    console.print(table)

    lang_table = Table(title="Language Distribution")
    lang_table.add_column("Language")
    lang_table.add_column("Accepted Documents")
    for lang, count in sorted(data.get("language_distribution", {}).items()):
        lang_table.add_row(lang, str(count))
    console.print(lang_table)

    rej_table = Table(title="Rejection Reasons")
    rej_table.add_column("Reason")
    rej_table.add_column("Count")
    for reason, count in sorted(data.get("rejection_reasons", {}).items()):
        rej_table.add_row(reason, str(count))
    console.print(rej_table)


@app.command()
def inspect(raw_dir: Path = typer.Option(Path("data/raw"))) -> None:
    """List raw source files under `raw_dir` without processing them."""
    for path in sorted(raw_dir.rglob("*.txt")):
        n_lines = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        console.print(f"{path}: {n_lines} lines")
