# Experiment Report: EXP-004__bpe

## 1. Hypothesis

(not specified in experiment config)

## 2. Experimental setup

- Git commit: `unknown (not a git repository or git unavailable)`
- Config hash: `442e0fa98ba759cf`
- Seed: `0`
- Hardware: `cpu`, 2 CPU cores,
  7.84 GB RAM, 0 GPU(s)
- Software: Python 3.11.15, PyTorch 2.13.0+cu130

## 3. Dataset

- Dataset version: `v1`
- Tokenizer version: `bpe`
- Sequence length: 64
- Total packed tokens: 19955
- Tokens per language: {'ben': 2457, 'eng': 2383, 'guj': 1956, 'hin': 2109, 'kan': 1892, 'mal': 1906, 'mar': 1673, 'pan': 1952, 'tam': 1649, 'tel': 2023}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'ben': 0.038, 'eng': 0.039, 'guj': 0.042, 'hin': 0.041, 'kan': 0.043, 'mal': 0.042, 'mar': 0.045, 'pan': 0.042, 'tam': 0.045, 'tel': 0.041}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 30528
- Total training wall-clock time: 2.359 s
- Mean tokens/sec (measured, CPU): 12939.15
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.8793370723724365
- Final validation loss: 6.88879
- Overall perplexity: 991.8323
- Macro-average per-language perplexity: 992.3379
- Weighted-average per-language perplexity: 992.3379

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 6.925919 | 1018.3301 | 2944 |
| eng | 6.880438 | 973.0524 | 2944 |
| guj | 6.855158 | 948.7618 | 2944 |
| hin | 6.8743 | 967.0984 | 2944 |
| kan | 6.92733 | 1019.7678 | 2944 |
| mal | 6.916113 | 1008.3923 | 2944 |
| mar | 6.859689 | 953.0704 | 2944 |
| pan | 6.890737 | 983.1258 | 2944 |
| tam | 6.90509 | 997.3384 | 2944 |
| tel | 6.960766 | 1054.4412 | 2944 |

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
