"""IndicLM CLI entry point.

Only `doctor` is implemented in this milestone (repository foundation).
Subsequent milestones add: data, tokenizer, train, evaluate, experiment,
report, serve — each as its own Typer sub-app registered here, so the CLI
grows without this file becoming a dumping ground.
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from indiclm.cli.ablation_cmd import app as ablation_app
from indiclm.cli.data_cmd import app as data_app
from indiclm.cli.experiment_cmd import app as experiment_app
from indiclm.cli.report_cmd import app as report_app
from indiclm.cli.scaling_cmd import app as scaling_app
from indiclm.cli.tokenizer_cmd import app as tokenizer_app
from indiclm.cli.train_cmd import app as train_app
from indiclm.utils.hardware import detect_hardware
from indiclm.utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="indiclm",
    help="IndicLM: multilingual Indian-language model research platform.",
    no_args_is_help=True,
)
console = Console()

app.add_typer(data_app, name="data")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(train_app, name="")  # exposes `train` and `evaluate` at top level

# `experiment_app`, `scaling_app`, and `ablation_app` are three separate
# Typer sub-apps (split across files so each stays focused: run/compare,
# the scaling sweep, and the two ablations) that all logically belong
# under one `indiclm experiment ...` namespace. Click's command registry
# is a flat dict keyed by name, so three `app.add_typer(..., name=
# "experiment")` calls here don't merge -- each one silently overwrites
# the last, leaving only the final call's commands reachable (this was
# discovered as a real bug: `experiment run`, `experiment compare`,
# `experiment list-experiments`, and `experiment scaling-sweep` were all
# unreachable via the CLI, though the underlying functions work fine
# when called directly). Merging the registered commands onto one Typer
# app before mounting it once is the fix.
experiment_app.registered_commands += scaling_app.registered_commands
experiment_app.registered_commands += ablation_app.registered_commands
app.add_typer(experiment_app, name="experiment")
app.add_typer(report_app, name="report")


@app.command()
def serve(
    checkpoint: str = typer.Option(..., help="Path to a training checkpoint (.pt)."),
    tokenizer: str = typer.Option("data/tokenizer_v1/indiclm_tokenizer.model"),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
) -> None:
    """Serve a trained checkpoint over the inference API (FastAPI/uvicorn)."""
    import os

    import uvicorn

    os.environ["INDICLM_CHECKPOINT"] = checkpoint
    os.environ["INDICLM_TOKENIZER"] = tokenizer
    uvicorn.run("indiclm.inference.api:app", host=host, port=port)


@app.command()
def doctor() -> None:
    """Diagnose Python, PyTorch, CUDA, GPU, disk, memory, and dependencies."""
    configure_logging()
    log = get_logger(__name__)

    table = Table(title="IndicLM Environment Diagnostics")
    table.add_column("Check")
    table.add_column("Result")

    table.add_row("Python", sys.version.split()[0])

    try:
        import torch

        table.add_row("PyTorch", torch.__version__)
        table.add_row("CUDA available", str(torch.cuda.is_available()))
    except ImportError:
        table.add_row("PyTorch", "[red]not installed[/red]")
        table.add_row("CUDA available", "n/a")

    hw = detect_hardware()
    table.add_row("Device type", hw.device_type)
    table.add_row("GPU count", str(hw.num_gpus))
    if hw.gpu_names:
        table.add_row("GPU(s)", ", ".join(hw.gpu_names))
        table.add_row("GPU memory (GB)", ", ".join(f"{m:.1f}" for m in hw.gpu_memory_gb))
    table.add_row("CPU cores", str(hw.cpu_count))
    table.add_row("RAM (GB)", str(hw.total_ram_gb))
    table.add_row("Recommended profile", hw.recommended_profile)

    console.print(table)
    log.info("doctor_check_complete", profile=hw.recommended_profile)


if __name__ == "__main__":
    app()
