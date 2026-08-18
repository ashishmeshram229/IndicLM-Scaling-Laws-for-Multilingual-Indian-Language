# Experiment Report: EXP-004__unigram

## 1. Hypothesis

(not specified in experiment config)

## 2. Experimental setup

- Git commit: `unknown (not a git repository or git unavailable)`
- Config hash: `3b1f74480fe90815`
- Seed: `0`
- Hardware: `cpu`, 2 CPU cores,
  7.84 GB RAM, 0 GPU(s)
- Software: Python 3.11.15, PyTorch 2.13.0+cpu

## 3. Dataset

- Dataset version: `v1`
- Tokenizer version: `unigram`
- Sequence length: 64
- Total packed tokens: 19955
- Tokens per language: {'ben': 1793, 'eng': 3458, 'guj': 1837, 'hin': 2295, 'kan': 1898, 'mal': 2003, 'mar': 994, 'pan': 1846, 'tam': 1905, 'tel': 1971}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'ben': 2.384, 'eng': 1.799, 'guj': 2.358, 'hin': 2.145, 'kan': 2.326, 'mal': 2.274, 'mar': 3.068, 'pan': 2.355, 'tam': 2.323, 'tel': 2.289}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 30528
- Total training wall-clock time: 2.045 s
- Mean tokens/sec (measured, CPU): 14924.66
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.504048109054565
- Final validation loss: 6.500119
- Overall perplexity: 660.7111
- Macro-average per-language perplexity: 661.3335
- Weighted-average per-language perplexity: 661.3335

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 6.544796 | 695.6146 | 2944 |
| eng | 6.427234 | 618.4608 | 2944 |
| guj | 6.4833 | 654.1258 | 2944 |
| hin | 6.440415 | 626.6666 | 2944 |
| kan | 6.564239 | 709.2718 | 2944 |
| mal | 6.511239 | 672.6591 | 2944 |
| mar | 6.441067 | 627.0757 | 2944 |
| pan | 6.520092 | 678.6405 | 2944 |
| tam | 6.50507 | 668.5223 | 2944 |
| tel | 6.495715 | 662.2977 | 2944 |

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
