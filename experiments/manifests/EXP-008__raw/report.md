# Experiment Report: EXP-008__raw

## 1. Hypothesis

(not specified in experiment config)

## 2. Experimental setup

- Git commit: `unknown (not a git repository or git unavailable)`
- Config hash: `c87e1533385169fb`
- Seed: `0`
- Hardware: `cpu`, 2 CPU cores,
  7.84 GB RAM, 0 GPU(s)
- Software: Python 3.11.15, PyTorch 2.13.0+cpu

## 3. Dataset

- Dataset version: `raw`
- Tokenizer version: `bpe_v1`
- Sequence length: 64
- Total packed tokens: 19955
- Tokens per language: {'ben': 1809, 'eng': 3545, 'guj': 1821, 'hin': 2214, 'kan': 1903, 'mal': 2047, 'mar': 1034, 'pan': 1817, 'tam': 1873, 'tel': 1937}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'ben': 2.284, 'eng': 1.713, 'guj': 2.279, 'hin': 2.095, 'kan': 2.236, 'mal': 2.166, 'mar': 2.904, 'pan': 2.28, 'tam': 2.251, 'tel': 2.219}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 30528
- Total training wall-clock time: 1.717 s
- Mean tokens/sec (measured, CPU): 17777.48
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.604960203170776
- Final validation loss: 6.609044
- Overall perplexity: 734.025
- Macro-average per-language perplexity: 734.2832
- Weighted-average per-language perplexity: 734.2832

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 6.597271 | 733.092 | 2944 |
| eng | 6.655835 | 777.3068 | 2944 |
| guj | 6.550493 | 699.5889 | 2944 |
| hin | 6.5935 | 730.3325 | 2944 |
| kan | 6.620595 | 750.3912 | 2944 |
| mal | 6.601626 | 736.2916 | 2944 |
| mar | 6.588432 | 726.6409 | 2944 |
| pan | 6.58765 | 726.0723 | 2944 |
| tam | 6.613018 | 744.7273 | 2944 |
| tel | 6.577011 | 718.3886 | 2944 |

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
