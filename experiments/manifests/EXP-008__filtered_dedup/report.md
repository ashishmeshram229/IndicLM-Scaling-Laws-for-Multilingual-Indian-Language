# Experiment Report: EXP-008__filtered_dedup

## 1. Hypothesis

(not specified in experiment config)

## 2. Experimental setup

- Git commit: `unknown (not a git repository or git unavailable)`
- Config hash: `6ea28ec3efbfb69b`
- Seed: `0`
- Hardware: `cpu`, 2 CPU cores,
  7.84 GB RAM, 0 GPU(s)
- Software: Python 3.11.15, PyTorch 2.13.0+cpu

## 3. Dataset

- Dataset version: `filtered_dedup`
- Tokenizer version: `bpe_v1`
- Sequence length: 64
- Total packed tokens: 19955
- Tokens per language: {'ben': 1812, 'eng': 3522, 'guj': 1823, 'hin': 2218, 'kan': 1905, 'mal': 2050, 'mar': 1035, 'pan': 1820, 'tam': 1875, 'tel': 1940}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'ben': 2.288, 'eng': 1.721, 'guj': 2.282, 'hin': 2.098, 'kan': 2.239, 'mal': 2.169, 'mar': 2.907, 'pan': 2.284, 'tam': 2.254, 'tel': 2.222}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 30528
- Total training wall-clock time: 1.693 s
- Mean tokens/sec (measured, CPU): 18032.44
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.599850416183472
- Final validation loss: 6.606203
- Overall perplexity: 733.2537
- Macro-average per-language perplexity: 733.596
- Weighted-average per-language perplexity: 733.596

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 6.597795 | 733.476 | 2944 |
| eng | 6.665875 | 785.1504 | 2944 |
| guj | 6.553451 | 701.6613 | 2944 |
| hin | 6.578663 | 719.5765 | 2944 |
| kan | 6.617557 | 748.1154 | 2944 |
| mal | 6.608365 | 741.2699 | 2944 |
| mar | 6.574229 | 716.3928 | 2944 |
| pan | 6.583597 | 723.1357 | 2944 |
| tam | 6.620453 | 750.2846 | 2944 |
| tel | 6.574933 | 716.8976 | 2944 |

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

(not specified in experiment config)
