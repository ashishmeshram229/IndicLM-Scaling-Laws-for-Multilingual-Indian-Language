# Experiment Report: EXP-006

## 1. Hypothesis

An Indic-balanced static mixture (equal weight across the nine Indic languages, small residual share for English) will improve macro-average Indic perplexity relative to the English-heavy mixture (EXP-005) at the same total token budget, at some cost to English perplexity.


## 2. Experimental setup

- Git commit: `b7fa0f9da9fca38ddbd554a57b6c9ff1d4e41670`
- Config hash: `8b932e95e0de3394`
- Seed: `1`
- Hardware: `mps`, 10 CPU cores,
  16.0 GB RAM, 0 GPU(s)
- Software: Python 3.13.5, PyTorch 2.9.1

## 3. Dataset

- Dataset version: `v1`
- Tokenizer version: `bpe_v1`
- Sequence length: 64
- Total packed tokens: 19955
- Tokens per language: {'ben': 2000, 'eng': 2000, 'guj': 2000, 'hin': 2000, 'kan': 2000, 'mal': 2000, 'mar': 2000, 'pan': 2000, 'tam': 2000, 'tel': 2000}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'ben': 0.031, 'eng': 0.032, 'guj': 0.043, 'hin': 0.039, 'kan': 0.045, 'mal': 0.045, 'mar': 0.054, 'pan': 0.043, 'tam': 0.055, 'tel': 0.041}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 30528
- Total training wall-clock time: 0.371 s
- Mean tokens/sec (measured, CPU): 82360.54
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.912670373916626
- Final validation loss: 6.865801
- Overall perplexity: 980.1087
- Macro-average per-language perplexity: 981.759
- Weighted-average per-language perplexity: 981.759

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 7.007467 | 1104.852 | 2944 |
| eng | 6.965305 | 1059.2383 | 2944 |
| guj | 6.869829 | 962.7837 | 2944 |
| hin | 6.831979 | 927.0235 | 2944 |
| kan | 6.886778 | 979.2414 | 2944 |
| mal | 6.877888 | 970.5744 | 2944 |
| mar | 6.794455 | 892.8824 | 2944 |
| pan | 6.896012 | 988.3258 | 2944 |
| tam | 6.863569 | 956.7755 | 2944 |
| tel | 6.883353 | 975.8934 | 2944 |

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

Compare macro-average Indic perplexity against EXP-005 and EXP-007 via `indiclm experiment compare`.
