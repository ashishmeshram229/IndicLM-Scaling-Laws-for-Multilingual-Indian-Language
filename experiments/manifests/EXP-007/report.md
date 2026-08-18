# Experiment Report: EXP-007

## 1. Hypothesis

Temperature sampling (alpha=0.5, p_i = n_i^alpha / sum_j n_j^alpha over each language's measured available token count) will land between the natural-proportion mixture (alpha=1.0) and uniform oversampling (alpha=0.0), giving a macro-average Indic perplexity better than EXP-005 (English-heavy) without the extreme oversampling of EXP-010.


## 2. Experimental setup

- Git commit: `unknown (not a git repository or git unavailable)`
- Config hash: `8df06bee8e3b8633`
- Seed: `0`
- Hardware: `cpu`, 2 CPU cores,
  7.84 GB RAM, 0 GPU(s)
- Software: Python 3.11.15, PyTorch 2.13.0+cpu

## 3. Dataset

- Dataset version: `v1`
- Tokenizer version: `bpe_v1`
- Sequence length: 64
- Total packed tokens: 19955
- Tokens per language: {'ben': 1879, 'eng': 3021, 'guj': 1887, 'hin': 2170, 'kan': 1947, 'mal': 2052, 'mar': 1260, 'pan': 1885, 'tam': 1926, 'tel': 1973}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'ben': 2.372, 'eng': 1.476, 'guj': 2.362, 'hin': 2.053, 'kan': 2.288, 'mal': 2.171, 'mar': 3.539, 'pan': 2.365, 'tam': 2.315, 'tel': 2.26}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 30528
- Total training wall-clock time: 1.892 s
- Mean tokens/sec (measured, CPU): 16131.06
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.583242177963257
- Final validation loss: 6.616812
- Overall perplexity: 733.1068
- Macro-average per-language perplexity: 733.4844
- Weighted-average per-language perplexity: 733.4844

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 6.588233 | 726.4964 | 2944 |
| eng | 6.673458 | 791.1268 | 2944 |
| guj | 6.545092 | 695.8204 | 2944 |
| hin | 6.587206 | 725.7499 | 2944 |
| kan | 6.599459 | 734.6978 | 2944 |
| mal | 6.606814 | 740.121 | 2944 |
| mar | 6.570803 | 713.9429 | 2944 |
| pan | 6.60327 | 737.5031 | 2944 |
| tam | 6.617345 | 747.9568 | 2944 |
| tel | 6.581234 | 721.4287 | 2944 |

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
