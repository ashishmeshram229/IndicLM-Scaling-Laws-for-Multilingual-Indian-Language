# Experiment Report: EXP-004__unigram

## 1. Hypothesis

(not specified in experiment config)

## 2. Experimental setup

- Git commit: `unknown (not a git repository or git unavailable)`
- Config hash: `3b1f74480fe90815`
- Seed: `0`
- Hardware: `cpu`, 2 CPU cores,
  7.84 GB RAM, 0 GPU(s)
- Software: Python 3.11.15, PyTorch 2.13.0+cu130

## 3. Dataset

- Dataset version: `v1`
- Tokenizer version: `unigram`
- Sequence length: 64
- Total packed tokens: 19955
- Tokens per language: {'ben': 2567, 'eng': 2357, 'guj': 1935, 'hin': 2206, 'kan': 1799, 'mal': 1858, 'mar': 1714, 'pan': 1973, 'tam': 1592, 'tel': 1999}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'ben': 0.03, 'eng': 0.031, 'guj': 0.034, 'hin': 0.032, 'kan': 0.035, 'mal': 0.034, 'mar': 0.035, 'pan': 0.033, 'tam': 0.037, 'tel': 0.033}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 30528
- Total training wall-clock time: 1.698 s
- Mean tokens/sec (measured, CPU): 17980.53
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.360205888748169
- Final validation loss: 6.363677
- Overall perplexity: 597.4121
- Macro-average per-language perplexity: 598.7749
- Weighted-average per-language perplexity: 598.7749

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 6.392332 | 597.2478 | 2944 |
| eng | 6.335931 | 564.4947 | 2944 |
| guj | 6.391885 | 596.9807 | 2944 |
| hin | 6.283076 | 535.4329 | 2944 |
| kan | 6.481847 | 653.176 | 2944 |
| mal | 6.471241 | 646.285 | 2944 |
| mar | 6.301881 | 545.5971 | 2944 |
| pan | 6.372795 | 585.6923 | 2944 |
| tam | 6.47694 | 649.9787 | 2944 |
| tel | 6.418144 | 612.8643 | 2944 |

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
