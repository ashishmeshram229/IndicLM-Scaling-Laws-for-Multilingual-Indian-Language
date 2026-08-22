"""`indiclm report generate`: assembles the top-level technical report
(docs/research_report.md) from whatever experiments have actually been
run under experiments/manifests/. Experiments with no manifest are
explicitly marked "not available" rather than omitted or faked.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from indiclm.experiments.dashboard import generate_dashboard

app = typer.Typer(help="Generate the aggregated research report from run experiments.")
console = Console()

REGISTRY = [
    ("EXP-001", "Baseline tiny model (smallest size in scaling sweep)"),
    ("EXP-002", "Baseline mid-size model (scaling sweep)"),
    ("EXP-003", "Baseline larger model (scaling sweep)"),
    ("EXP-004", "Tokenizer comparison (BPE vs Unigram)"),
    ("EXP-005", "English-heavy mixture"),
    ("EXP-006", "Indic-balanced mixture"),
    ("EXP-007", "Temperature sampling mixture"),
    ("EXP-008", "Data filtering ablation"),
    ("EXP-009", "Deduplication ablation"),
    ("EXP-010", "Low-resource oversampling"),
    ("EXP-011", "Code-mixed training"),
    ("EXP-012", "Compute-optimal scaling sweep"),
]


@app.command()
def generate(
    experiments_root: Path = typer.Option(Path("experiments/manifests")),
    out_path: Path = typer.Option(Path("docs/research_report.md")),
) -> None:
    sections = []
    for exp_id, description in REGISTRY:
        manifest_path = experiments_root / exp_id / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("final_val_loss") is not None:
                summary = (
                    f"Final val loss: {manifest.get('final_val_loss')}. Overall perplexity: "
                    f"{manifest.get('evaluation_metrics', {}).get('overall_perplexity')}."
                )
            else:
                # Multi-variant comparison (EXP-004/008/009) or a sweep+fit
                # (EXP-012): evaluation_metrics holds a dict of results rather
                # than one scalar, so summarize it directly instead of "None".
                summary = f"Comparison results: {json.dumps(manifest.get('evaluation_metrics', {}))}"
            sections.append(
                f"### {exp_id}: {description}\n\n**Status: run.** {summary} "
                f"See `experiments/manifests/{exp_id}/report.md` for the full writeup.\n"
            )
        else:
            sections.append(
                f"### {exp_id}: {description}\n\n**Status: not available** — not run in this "
                f"environment. See `docs/reproducibility.md` for how to run it.\n"
            )
    body = "\n".join(sections)
    Path(out_path).write_text(body, encoding="utf-8")
    console.print(f"[green]Report generated:[/green] {out_path}")


@app.command()
def dashboard(
    experiments_root: Path = typer.Option(Path("experiments/manifests")),
    data_dir: Path = typer.Option(Path("data")),
    out_path: Path = typer.Option(Path("docs/dashboard.html")),
) -> None:
    """Generate the static HTML research dashboard (dataset, tokenizer,
    per-experiment results, ablations, scaling) from whatever artifacts
    are currently on disk under `experiments/manifests/` and `data/`."""
    path = generate_dashboard(experiments_root=experiments_root, data_dir=data_dir, out_path=out_path)
    console.print(f"[green]Dashboard generated:[/green] {path}")
