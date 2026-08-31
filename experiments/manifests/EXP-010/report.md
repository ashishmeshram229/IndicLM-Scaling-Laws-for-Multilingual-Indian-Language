# Experiment Report: EXP-010

## 1. Hypothesis

Uniform sampling (alpha=0.0) across all ten languages regardless of available token count will oversample the lowest-resource language in this corpus (Marathi, ~1000 available tokens vs. English's ~3500) by a larger factor than any other language, testing whether that oversampling helps Marathi perplexity or causes negative transfer (more repetition of a small pool without new information) relative to natural-proportion sampling (Q6).


## 2. Experimental setup

- Git commit: `unknown (not a git repository or git unavailable)`
- Config hash: `8652c4fb7350d414`
- Seed: `0`
- Hardware: `cpu`, 2 CPU cores,
  7.84 GB RAM, 0 GPU(s)
- Software: Python 3.11.15, PyTorch 2.13.0+cu130

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
- Total training wall-clock time: 2.045 s
- Mean tokens/sec (measured, CPU): 14927.48
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.877647638320923
- Final validation loss: 6.882474
- Overall perplexity: 991.6731
- Macro-average per-language perplexity: 992.6504
- Weighted-average per-language perplexity: 992.6504

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 6.973952 | 1068.4369 | 2944 |
| eng | 6.90544 | 997.6873 | 2944 |
| guj | 6.879576 | 972.2136 | 2944 |
| hin | 6.846433 | 940.5205 | 2944 |
| kan | 6.918303 | 1010.6038 | 2944 |
| mal | 6.91973 | 1012.0471 | 2944 |
| mar | 6.83204 | 927.0798 | 2944 |
| pan | 6.899671 | 991.9482 | 2944 |
| tam | 6.857631 | 951.1113 | 2944 |
| tel | 6.961159 | 1054.8558 | 2944 |

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

Compare Marathi-specific perplexity against EXP-007 (alpha=0.5) and a natural-proportion (alpha=1.0) run.
