# Experiment Report: EXP-005

## 1. Hypothesis

An English-heavy static mixture (60% English, remaining 40% split evenly across the nine Indic languages) will show lower validation loss on English and worse (or no better) loss on Indic languages compared to a temperature-sampled or Indic-balanced mixture at the same total token budget — demonstrating "how much performance is lost by heavily English-dominated training" (Q9).


## 2. Experimental setup

- Git commit: `b7fa0f9da9fca38ddbd554a57b6c9ff1d4e41670`
- Config hash: `3e6cd408067ed4d2`
- Seed: `0`
- Hardware: `mps`, 10 CPU cores,
  16.0 GB RAM, 0 GPU(s)
- Software: Python 3.13.5, PyTorch 2.9.1

## 3. Dataset

- Dataset version: `v1`
- Tokenizer version: `bpe_v1`
- Sequence length: 64
- Total packed tokens: 19955
- Tokens per language: {'ben': 884, 'eng': 12048, 'guj': 884, 'hin': 884, 'kan': 884, 'mal': 884, 'mar': 883, 'pan': 883, 'tam': 883, 'tel': 883}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'ben': 0.014, 'eng': 0.195, 'guj': 0.019, 'hin': 0.017, 'kan': 0.02, 'mal': 0.02, 'mar': 0.024, 'pan': 0.019, 'tam': 0.024, 'tel': 0.018}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 30528
- Total training wall-clock time: 0.347 s
- Mean tokens/sec (measured, CPU): 88057.32
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.832165956497192
- Final validation loss: 6.683803
- Overall perplexity: 1040.9503
- Macro-average per-language perplexity: 1052.1425
- Weighted-average per-language perplexity: 1052.1425

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 7.04678 | 1149.1522 | 2944 |
| eng | 6.491524 | 659.5278 | 2944 |
| guj | 6.992279 | 1088.1986 | 2944 |
| hin | 6.968186 | 1062.2945 | 2944 |
| kan | 7.021769 | 1120.7672 | 2944 |
| mal | 6.98445 | 1079.7119 | 2944 |
| mar | 6.948829 | 1041.9291 | 2944 |
| pan | 7.047199 | 1149.6347 | 2944 |
| tam | 6.957533 | 1051.0372 | 2944 |
| tel | 7.020344 | 1119.1718 | 2944 |

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

Compare against EXP-006 (Indic-balanced) and EXP-007 (temperature sampling) at the same token budget.
