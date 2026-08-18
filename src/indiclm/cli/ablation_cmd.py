"""`indiclm experiment tokenizer-ablation` and `... data-ablation`: pairwise
(or N-way) comparisons that a single `run_experiment` call doesn't capture
on its own — each variant is a full run via `run_experiment`, and this
module adds a synthesizing manifest + comparison note at the parent
experiment ID (EXP-004, EXP-008, EXP-009).
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from indiclm.experiments.manifest import build_manifest, write_manifest
from indiclm.experiments.runner import run_experiment
from indiclm.utils.logging import configure_logging

app = typer.Typer(help="Multi-variant ablation experiments (EXP-004, EXP-008, EXP-009).")
console = Console()

_BASE_MODEL = {"d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "dropout": 0.0}
_BASE_TRAINING = {
    "max_steps": 60, "micro_batch_size": 4, "gradient_accumulation_steps": 2,
    "learning_rate": 3e-4, "warmup_steps": 10, "eval_every": 15, "checkpoint_every": 60,
    "log_every": 15, "device": "cpu",
}


def _write_comparison_manifest(parent_id: str, variants: dict[str, dict], hypothesis: str) -> None:
    out_dir = Path("experiments/manifests") / parent_id
    manifest = build_manifest(
        experiment_id=parent_id,
        config={"variants": list(variants.keys()), "hypothesis": hypothesis},
        dataset_version="v1", tokenizer_version="v1", seed=0,
    )
    manifest.evaluation_metrics = {
        name: {
            "final_val_loss": r["result"]["final_val_loss"],
            "overall_perplexity": r["evaluation"]["overall_perplexity"],
            "macro_avg_perplexity": r["evaluation"]["macro_avg_perplexity"],
        }
        for name, r in variants.items()
    }
    write_manifest(manifest, out_dir)

    rows = "\n".join(
        f"| {name} | {r['result']['final_val_loss']} | {r['evaluation']['overall_perplexity']} | "
        f"{r['evaluation']['macro_avg_perplexity']} |"
        for name, r in variants.items()
    )
    report = f"""# Ablation: {parent_id}

## Hypothesis

{hypothesis}

## Results

| Variant | Final val loss | Overall perplexity | Macro-avg perplexity |
|---|---|---|---|
{rows}

Each variant's full report is at `experiments/manifests/{parent_id}__<variant>/report.md`.

## Limitations

Single seed per variant, tiny hand-authored corpus (see docs/data_pipeline.md).
Differences below roughly one perplexity point at this scale should be read
as noise, not a robust effect — see docs/reproducibility.md.
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")


@app.command("tokenizer-ablation")
def tokenizer_ablation(
    shards_dir: Path = typer.Option(Path("data/processed")),
    seq_len: int = typer.Option(64),
    total_tokens: int = typer.Option(20000),
) -> None:
    """EXP-004: identical tiny model trained with the BPE vs Unigram tokenizer."""
    configure_logging()
    variants = {}
    for name, tok_path in [
        ("bpe", "data/tokenizer_v1/indiclm_tokenizer.model"),
        ("unigram", "data/tokenizer_unigram/indiclm_tokenizer.model"),
    ]:
        cfg = {
            "experiment_id": f"EXP-004__{name}",
            "data": {
                "shards_dir": str(shards_dir), "tokenizer_path": tok_path,
                "seq_len": seq_len, "total_tokens": total_tokens, "alpha": 0.7,
                "dataset_version": "v1", "tokenizer_version": name,
            },
            "model": _BASE_MODEL,
            "training": _BASE_TRAINING,
        }
        variants[name] = run_experiment(cfg)

    _write_comparison_manifest(
        "EXP-004", variants,
        "Does tokenizer efficiency (tokens/char, compression) predict actual model "
        "quality (validation perplexity) when model architecture and token budget "
        "are held constant? Compares BPE vs Unigram SentencePiece tokenizers trained "
        "on the same corpus (see data/tokenizer_v1 and data/tokenizer_unigram "
        "benchmark_report.json for the efficiency numbers being correlated against).",
    )
    console.print("[green]EXP-004 (tokenizer ablation) complete.[/green]")


@app.command("data-ablation")
def data_ablation(seq_len: int = typer.Option(64), total_tokens: int = typer.Option(20000)) -> None:
    """EXP-008 (filtering) + EXP-009 (dedup): raw vs filtered vs filtered+dedup."""
    configure_logging()
    tokenizer_path = "data/tokenizer_v1/indiclm_tokenizer.model"
    variant_dirs = {
        "raw": "data/processed_raw",
        "filtered": "data/processed_filtered",
        "filtered_dedup": "data/processed_dedup",
    }
    variants = {}
    for name, shards_dir in variant_dirs.items():
        cfg = {
            "experiment_id": f"EXP-008__{name}",
            "data": {
                "shards_dir": shards_dir, "tokenizer_path": tokenizer_path,
                "seq_len": seq_len, "total_tokens": total_tokens, "alpha": 0.7,
                "dataset_version": name, "tokenizer_version": "bpe_v1",
            },
            "model": _BASE_MODEL,
            "training": _BASE_TRAINING,
        }
        variants[name] = run_experiment(cfg)

    _write_comparison_manifest(
        "EXP-008", variants,
        "Does quality filtering (raw -> filtered) and/or deduplication "
        "(filtered -> filtered_dedup) improve validation loss at a fixed token "
        "budget and fixed model size? (Q7, Q8)",
    )
    # EXP-009 is the dedup-specific slice of the same three runs.
    _write_comparison_manifest(
        "EXP-009",
        {k: v for k, v in variants.items() if k in ("filtered", "filtered_dedup")},
        "Isolating deduplication's effect: filtered (no dedup) vs filtered+dedup, "
        "same quality thresholds, same token budget.",
    )
    console.print("[green]EXP-008 and EXP-009 (data ablations) complete.[/green]")
