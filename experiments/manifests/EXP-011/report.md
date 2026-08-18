# Experiment Report: EXP-011

## 1. Hypothesis

A tiny model trained only on the code-mixed sample corpus (Hindi-English, Marathi-English, Tamil-English) will reach a reasonable training loss despite the code-mixed distribution being different from any single monolingual corpus, demonstrating the pipeline's ability to identify, isolate, and train on code-mixed data end-to-end (data/raw/codemixed_sample -> data/processed_codemixed).


## 2. Experimental setup

- Git commit: `unknown (not a git repository or git unavailable)`
- Config hash: `9012705c2a0378cd`
- Seed: `0`
- Hardware: `cpu`, 2 CPU cores,
  7.84 GB RAM, 0 GPU(s)
- Software: Python 3.11.15, PyTorch 2.13.0+cpu

## 3. Dataset

- Dataset version: `codemixed_v1`
- Tokenizer version: `bpe_v1`
- Sequence length: 48
- Total packed tokens: 3969
- Tokens per language: {'eng': 4000}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'eng': 3.868}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 7392
- Total training wall-clock time: 0.669 s
- Mean tokens/sec (measured, CPU): 11053.26
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 40, 'micro_batch_size': 4, 'gradient_accumulation_steps': 1, 'learning_rate': 0.0003, 'warmup_steps': 8, 'eval_every': 10, 'checkpoint_every': 40, 'log_every': 10, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.327212810516357
- Final validation loss: 6.37383
- Overall perplexity: 566.7175
- Macro-average per-language perplexity: 566.7178
- Weighted-average per-language perplexity: 566.7178

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| eng | 6.339861 | 566.7178 | 2928 |

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

Evaluate a model trained on monolingual + code-mixed data jointly against one trained on monolingual data alone, on a held-out code-mixed set.
