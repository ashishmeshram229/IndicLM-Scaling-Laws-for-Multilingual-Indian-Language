# Experiment Report: EXP-005

## 1. Hypothesis

An English-heavy static mixture (60% English, remaining 40% split evenly across the nine Indic languages) will show lower validation loss on English and worse (or no better) loss on Indic languages compared to a temperature-sampled or Indic-balanced mixture at the same total token budget — demonstrating "how much performance is lost by heavily English-dominated training" (Q9).


## 2. Experimental setup

- Git commit: `unknown (not a git repository or git unavailable)`
- Config hash: `3e6cd408067ed4d2`
- Seed: `0`
- Hardware: `cpu`, 2 CPU cores,
  7.84 GB RAM, 0 GPU(s)
- Software: Python 3.11.15, PyTorch 2.13.0+cpu

## 3. Dataset

- Dataset version: `v1`
- Tokenizer version: `bpe_v1`
- Sequence length: 64
- Total packed tokens: 19955
- Tokens per language: {'ben': 884, 'eng': 12048, 'guj': 884, 'hin': 884, 'kan': 884, 'mal': 884, 'mar': 883, 'pan': 883, 'tam': 883, 'tel': 883}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'ben': 1.116, 'eng': 5.886, 'guj': 1.106, 'hin': 0.836, 'kan': 1.039, 'mal': 0.935, 'mar': 2.48, 'pan': 1.108, 'tam': 1.061, 'tel': 1.011}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 30528
- Total training wall-clock time: 2.078 s
- Mean tokens/sec (measured, CPU): 14692.27
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.527769565582275
- Final validation loss: 6.548765
- Overall perplexity: 789.1673
- Macro-average per-language perplexity: 792.4607
- Weighted-average per-language perplexity: 792.4607

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 6.696329 | 809.429 | 2944 |
| eng | 6.402652 | 603.4433 | 2944 |
| guj | 6.712696 | 822.7857 | 2944 |
| hin | 6.703235 | 815.038 | 2944 |
| kan | 6.66617 | 785.3821 | 2944 |
| mal | 6.717824 | 827.0158 | 2944 |
| mar | 6.654172 | 776.0152 | 2944 |
| pan | 6.735716 | 841.9459 | 2944 |
| tam | 6.754626 | 858.0187 | 2944 |
| tel | 6.666363 | 785.5331 | 2944 |

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
