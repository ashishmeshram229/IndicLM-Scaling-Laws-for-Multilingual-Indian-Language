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
    raw_dir: Path = typer.Option(Path("data/raw"), help="Directory of raw .txt sources (txt mode)."),
    output_dir: Path = typer.Option(Path("data/processed"), help="Where to write shards + stats."),
    dataset_version: str = typer.Option("v1"),
    min_quality_score: float = typer.Option(0.5),
    enable_quality_filter: bool = typer.Option(True),
    enable_exact_dedup: bool = typer.Option(True),
    enable_near_dedup: bool = typer.Option(True),
    source_format: str = typer.Option(
        "txt",
        help="Source format: 'txt' (bootstrap) or 'wikipedia-parquet' (production).",
    ),
    raw_wiki_dir: Path | None = typer.Option(
        None,
        help="Path to raw_wiki/ directory from download_corpus.py (wikipedia-parquet mode).",
    ),
    languages: str = typer.Option(
        "hi,mr,bn,ta,te,kn,ml,gu,pa,en",
        help="Comma-separated language codes to process (wikipedia-parquet mode).",
    ),
    max_docs_per_language: int | None = typer.Option(
        None,
        help="Cap docs per language. Defaults: en=500K, others=unlimited. Set 0 for no cap.",
    ),
) -> None:
    """Run the full pipeline: ingest -> langid -> quality -> dedup -> shard."""
    configure_logging()

    if source_format == "wikipedia-parquet" and raw_wiki_dir is None:
        console.print("[red]--raw-wiki-dir is required when --source-format=wikipedia-parquet[/red]")
        raise typer.Exit(1)

    # max_docs=0 means "no cap" (override the default English cap)
    effective_max_docs = None if max_docs_per_language == 0 else max_docs_per_language

    cfg = DataPipelineConfig(
        raw_dir=raw_dir,
        output_dir=output_dir,
        dataset_version=dataset_version,
        min_quality_score=min_quality_score,
        enable_quality_filter=enable_quality_filter,
        enable_exact_dedup=enable_exact_dedup,
        enable_near_dedup=enable_near_dedup,
        source_format=source_format,
        raw_wiki_dir=raw_wiki_dir,
        languages=[l.strip() for l in languages.split(",")],
        max_docs_per_language=effective_max_docs,
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
