"""Renders the per-experiment `report.md` — a lab-notebook-style writeup,
not a Kaggle-style results dump. Every number in the report is read back
from the manifest/result/eval objects the run actually produced; nothing
here is templated with placeholder numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from indiclm.evaluation.perplexity import EvaluationReport
from indiclm.experiments.manifest import ExperimentManifest
from indiclm.training.dataset import PackedDatasetStats
from indiclm.training.trainer import TrainingResult


def render_report(
    experiment_id: str,
    config: dict[str, Any],
    result: TrainingResult,
    eval_report: EvaluationReport,
    dataset_stats: PackedDatasetStats,
    manifest: ExperimentManifest,
    out_path: Path,
) -> None:
    lang_rows = "\n".join(
        f"| {lang} | {r.loss} | {r.perplexity} | {r.tokens_evaluated} |"
        for lang, r in sorted(eval_report.per_language.items())
    )
    hypothesis = config.get("hypothesis", "(not specified in experiment config)")
    next_experiment = config.get("next_experiment", "(not specified in experiment config)")

    md = f"""# Experiment Report: {experiment_id}

## 1. Hypothesis

{hypothesis}

## 2. Experimental setup

- Git commit: `{manifest.git_commit}`
- Config hash: `{manifest.config_hash}`
- Seed: `{manifest.seed}`
- Hardware: `{manifest.hardware.get("device_type")}`, {manifest.hardware.get("cpu_count")} CPU cores,
  {manifest.hardware.get("total_ram_gb")} GB RAM, {manifest.hardware.get("num_gpus")} GPU(s)
- Software: Python {manifest.software_versions.get("python")}, PyTorch {manifest.software_versions.get("torch")}

## 3. Dataset

- Dataset version: `{manifest.dataset_version}`
- Tokenizer version: `{manifest.tokenizer_version}`
- Sequence length: {dataset_stats.seq_len}
- Total packed tokens: {dataset_stats.total_tokens}
- Tokens per language: {dataset_stats.tokens_per_language}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {dataset_stats.epochs_per_language}
- Padding ratio: {dataset_stats.padding_ratio}

## 4. Model

```
{config.get("model", {})}
```

## 5. Compute

- Training tokens seen: {result.tokens_seen}
- Total training wall-clock time: {result.total_train_time_sec} s
- Mean tokens/sec (measured, CPU): {result.mean_tokens_per_sec}
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{config.get("training", {})}
```

## 7. Results

- Final training loss: {result.final_train_loss}
- Final validation loss: {result.final_val_loss}
- Overall perplexity: {eval_report.overall_perplexity}
- Macro-average per-language perplexity: {eval_report.macro_avg_perplexity}
- Weighted-average per-language perplexity: {eval_report.weighted_avg_perplexity}

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
{lang_rows}

## 8. Statistical uncertainty

Single-seed run; no confidence interval is computed for this individual
experiment. See `experiments/manifests/scaling_law/` for the
multi-point sweep used to estimate scaling-law parameter uncertainty.

## 9. Failure cases

Not applicable to this run's scope, or none observed. Anomaly-detector
warnings (if any) are recorded in `metrics.jsonl` under `training_anomaly_warning`.

## 10. Interpretation

At this corpus scale (a hand-authored bootstrap corpus of a few hundred
short documents per language — see `docs/data_pipeline.md`), loss values
demonstrate the pipeline is wired correctly end-to-end. They should not be
read as evidence about model quality on real multilingual data at scale.

## 11. Limitations

- Training corpus is a small, hand-authored sample corpus, not a
  production-scale Indic corpus (see `docs/reproducibility.md`).
- CPU-only training; throughput numbers do not reflect GPU performance.
- Single run per configuration unless this is part of an explicit sweep.

## 12. Next experiment

{next_experiment}
"""
    Path(out_path).write_text(md, encoding="utf-8")
