"""`indiclm tokenizer ...` subcommands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from indiclm.tokenizer.benchmark import benchmark_tokenizer, write_report
from indiclm.tokenizer.train import TokenizerTrainConfig, train_tokenizer
from indiclm.utils.logging import configure_logging

app = typer.Typer(help="Tokenizer training and benchmarking.")
console = Console()


@app.command()
def train(
    input_dir: Path = typer.Option(Path("data/processed")),
    output_dir: Path = typer.Option(Path("data/tokenizer_v1")),
    vocab_size: int = typer.Option(2000),
    model_type: str = typer.Option("bpe", help="bpe | unigram"),
) -> None:
    configure_logging()
    cfg = TokenizerTrainConfig(
        input_dir=input_dir, output_dir=output_dir, vocab_size=vocab_size, model_type=model_type
    )
    model_path = train_tokenizer(cfg)
    console.print(f"[green]Tokenizer trained:[/green] {model_path}")


@app.command()
def benchmark(
    model_path: Path = typer.Option(Path("data/tokenizer_v1/indiclm_tokenizer.model")),
    shards_dir: Path = typer.Option(Path("data/processed")),
    output_path: Path = typer.Option(Path("data/tokenizer_v1/benchmark_report.json")),
) -> None:
    configure_logging()
    reports = benchmark_tokenizer(model_path, shards_dir)
    write_report(reports, output_path)

    table = Table(title="Tokenizer Benchmark (per language)")
    for col in ["Language", "Tokens/char", "Tokens/word", "Compression", "UNK rate", "Mean seq len"]:
        table.add_column(col)
    for lang, r in sorted(reports.items()):
        table.add_row(
            lang, str(r.tokens_per_char), str(r.tokens_per_word), str(r.compression_ratio),
            str(r.unk_rate), str(r.mean_sequence_length),
        )
    console.print(table)
    console.print(f"[green]Report written to[/green] {output_path}")
