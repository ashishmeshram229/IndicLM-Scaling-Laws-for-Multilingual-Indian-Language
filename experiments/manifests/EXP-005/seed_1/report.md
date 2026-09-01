# Experiment Report: EXP-005

## 1. Hypothesis

An English-heavy static mixture (60% English, remaining 40% split evenly across the nine Indic languages) will show lower validation loss on English and worse (or no better) loss on Indic languages compared to a temperature-sampled or Indic-balanced mixture at the same total token budget — demonstrating "how much performance is lost by heavily English-dominated training" (Q9).


## 2. Experimental setup

- Git commit: `b7fa0f9da9fca38ddbd554a57b6c9ff1d4e41670`
- Config hash: `9b78d3338554b87c`
- Seed: `1`
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
- Total training wall-clock time: 0.315 s
- Mean tokens/sec (measured, CPU): 96846.68
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.721871376037598
- Final validation loss: 6.768578
- Overall perplexity: 1045.7944
- Macro-average per-language perplexity: 1056.1018
- Weighted-average per-language perplexity: 1056.1018

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 7.072908 | 1179.573 | 2944 |
| eng | 6.516636 | 676.2998 | 2944 |
| guj | 6.985806 | 1081.1773 | 2944 |
| hin | 6.982481 | 1077.5884 | 2944 |
| kan | 6.994363 | 1090.4693 | 2944 |
| mal | 6.975575 | 1070.1725 | 2944 |
| mar | 6.960301 | 1053.9507 | 2944 |
| pan | 7.016608 | 1114.9979 | 2944 |
| tam | 6.983996 | 1079.2228 | 2944 |
| tel | 7.036647 | 1137.5664 | 2944 |

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
