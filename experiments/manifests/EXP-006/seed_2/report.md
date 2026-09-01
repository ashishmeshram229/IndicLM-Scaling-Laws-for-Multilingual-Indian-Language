# Experiment Report: EXP-006

## 1. Hypothesis

An Indic-balanced static mixture (equal weight across the nine Indic languages, small residual share for English) will improve macro-average Indic perplexity relative to the English-heavy mixture (EXP-005) at the same total token budget, at some cost to English perplexity.


## 2. Experimental setup

- Git commit: `b7fa0f9da9fca38ddbd554a57b6c9ff1d4e41670`
- Config hash: `8894dc8f37782b29`
- Seed: `2`
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
- Total training wall-clock time: 0.386 s
- Mean tokens/sec (measured, CPU): 79093.42
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.8575098514556885
- Final validation loss: 6.899093
- Overall perplexity: 986.3616
- Macro-average per-language perplexity: 987.7571
- Weighted-average per-language perplexity: 987.7571

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 6.99759 | 1093.9932 | 2944 |
| eng | 6.892954 | 985.3078 | 2944 |
| guj | 6.872415 | 965.2765 | 2944 |
| hin | 6.849939 | 943.8236 | 2944 |
| kan | 6.908481 | 1000.7259 | 2944 |
| mal | 6.920154 | 1012.476 | 2944 |
| mar | 6.786074 | 885.4305 | 2944 |
| pan | 6.901256 | 993.5217 | 2944 |
| tam | 6.871655 | 964.5432 | 2944 |
| tel | 6.939712 | 1032.4726 | 2944 |

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
