# Experiment Report: EXP-007

## 1. Hypothesis

Temperature sampling (alpha=0.5, p_i = n_i^alpha / sum_j n_j^alpha over each language's measured available token count) will land between the natural-proportion mixture (alpha=1.0) and uniform oversampling (alpha=0.0), giving a macro-average Indic perplexity better than EXP-005 (English-heavy) without the extreme oversampling of EXP-010.


## 2. Experimental setup

- Git commit: `b7fa0f9da9fca38ddbd554a57b6c9ff1d4e41670`
- Config hash: `8df06bee8e3b8633`
- Seed: `0`
- Hardware: `mps`, 10 CPU cores,
  16.0 GB RAM, 0 GPU(s)
- Software: Python 3.13.5, PyTorch 2.9.1

## 3. Dataset

- Dataset version: `v1`
- Tokenizer version: `bpe_v1`
- Sequence length: 64
- Total packed tokens: 19955
- Tokens per language: {'ben': 2320, 'eng': 2270, 'guj': 1971, 'hin': 2081, 'kan': 1926, 'mal': 1935, 'mar': 1763, 'pan': 1969, 'tam': 1745, 'tel': 2020}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'ben': 0.036, 'eng': 0.037, 'guj': 0.042, 'hin': 0.04, 'kan': 0.043, 'mal': 0.043, 'mar': 0.047, 'pan': 0.042, 'tam': 0.048, 'tel': 0.041}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 30528
- Total training wall-clock time: 0.36 s
- Mean tokens/sec (measured, CPU): 84841.04
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.8683977127075195
- Final validation loss: 6.896834
- Overall perplexity: 988.3124
- Macro-average per-language perplexity: 989.004
- Weighted-average per-language perplexity: 989.004

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 6.940352 | 1033.134 | 2944 |
| eng | 6.896816 | 989.1201 | 2944 |
| guj | 6.863479 | 956.6899 | 2944 |
| hin | 6.847981 | 941.9776 | 2944 |
| kan | 6.916318 | 1008.5995 | 2944 |
| mal | 6.913918 | 1006.1813 | 2944 |
| mar | 6.831287 | 926.3826 | 2944 |
| pan | 6.912842 | 1005.1 | 2944 |
| tam | 6.883256 | 975.7984 | 2944 |
| tel | 6.953739 | 1047.0569 | 2944 |

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

Sweep alpha in {0.0, 0.3, 0.5, 0.7, 1.0} and plot macro-average perplexity vs alpha.
