# Experiment Report: EXP-008__filtered

## 1. Hypothesis

(not specified in experiment config)

## 2. Experimental setup

- Git commit: `unknown (not a git repository or git unavailable)`
- Config hash: `2f6820357bb217eb`
- Seed: `0`
- Hardware: `cpu`, 2 CPU cores,
  7.84 GB RAM, 0 GPU(s)
- Software: Python 3.11.15, PyTorch 2.13.0+cu130

## 3. Dataset

- Dataset version: `filtered`
- Tokenizer version: `bpe_v1`
- Sequence length: 64
- Total packed tokens: 19955
- Tokens per language: {'ben': 1660, 'eng': 3951, 'guj': 1732, 'hin': 2187, 'kan': 1892, 'mal': 2009, 'mar': 1058, 'pan': 1732, 'tam': 1882, 'tel': 1897}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'ben': 2.309, 'eng': 1.592, 'guj': 2.267, 'hin': 2.052, 'kan': 2.182, 'mal': 2.128, 'mar': 2.799, 'pan': 2.267, 'tam': 2.188, 'tel': 2.18}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 30528
- Total training wall-clock time: 2.143 s
- Mean tokens/sec (measured, CPU): 14244.9
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.798642158508301
- Final validation loss: 6.813816
- Overall perplexity: 903.4774
- Macro-average per-language perplexity: 905.4335
- Weighted-average per-language perplexity: 905.4335

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 6.9425 | 1035.3553 | 2944 |
| eng | 6.708305 | 819.1809 | 2944 |
| guj | 6.793461 | 891.9958 | 2944 |
| hin | 6.788392 | 887.4851 | 2944 |
| kan | 6.798075 | 896.1203 | 2944 |
| mal | 6.782098 | 881.9171 | 2944 |
| mar | 6.760341 | 862.936 | 2944 |
| pan | 6.86349 | 956.6999 | 2944 |
| tam | 6.750098 | 854.1424 | 2944 |
| tel | 6.875751 | 968.5023 | 2944 |

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
