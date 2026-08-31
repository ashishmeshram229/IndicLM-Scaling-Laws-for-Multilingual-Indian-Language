# Ablation: EXP-009

## Hypothesis

Isolating deduplication's effect: filtered (no dedup) vs filtered+dedup, same quality thresholds, same token budget.

## Results

| Variant | Final val loss | Overall perplexity | Macro-avg perplexity |
|---|---|---|---|
| filtered | 6.813816 | 903.4774 | 905.4335 |
| filtered_dedup | 6.814866 | 901.9895 | 903.9558 |

Each variant's full report is at `experiments/manifests/EXP-009__<variant>/report.md`.

## Limitations

Single seed per variant, tiny hand-authored corpus (see docs/data_pipeline.md).
Differences below roughly one perplexity point at this scale should be read
as noise, not a robust effect — see docs/reproducibility.md.
