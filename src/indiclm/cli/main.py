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

from indiclm.utils.hardware import detect_hardware
from indiclm.utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="indiclm",
    help="IndicLM: multilingual Indian-language model research platform.",
    no_args_is_help=True,
)
console = Console()


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
