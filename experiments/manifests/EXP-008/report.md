# Ablation: EXP-008

## Hypothesis

Does quality filtering (raw -> filtered) and/or deduplication (filtered -> filtered_dedup) improve validation loss at a fixed token budget and fixed model size? (Q7, Q8)

## Results

| Variant | Final val loss | Overall perplexity | Macro-avg perplexity |
|---|---|---|---|
| raw | 6.609044 | 734.025 | 734.2832 |
| filtered | 6.609044 | 734.025 | 734.2832 |
| filtered_dedup | 6.606203 | 733.2537 | 733.596 |

Each variant's full report is at `experiments/manifests/EXP-008__<variant>/report.md`.

## Limitations

Single seed per variant, tiny hand-authored corpus (see docs/data_pipeline.md).
Differences below roughly one perplexity point at this scale should be read
as noise, not a robust effect — see docs/reproducibility.md.
