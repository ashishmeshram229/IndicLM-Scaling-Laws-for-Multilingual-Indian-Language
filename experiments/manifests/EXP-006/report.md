# Experiment Report: EXP-006

## 1. Hypothesis

An Indic-balanced static mixture (equal weight across the nine Indic languages, small residual share for English) will improve macro-average Indic perplexity relative to the English-heavy mixture (EXP-005) at the same total token budget, at some cost to English perplexity.


## 2. Experimental setup

- Git commit: `unknown (not a git repository or git unavailable)`
- Config hash: `55b92382bb9fa79f`
- Seed: `0`
- Hardware: `cpu`, 2 CPU cores,
  7.84 GB RAM, 0 GPU(s)
- Software: Python 3.11.15, PyTorch 2.13.0+cpu

## 3. Dataset

- Dataset version: `v1`
- Tokenizer version: `bpe_v1`
- Sequence length: 64
- Total packed tokens: 19955
- Tokens per language: {'ben': 2000, 'eng': 2000, 'guj': 2000, 'hin': 2000, 'kan': 2000, 'mal': 2000, 'mar': 2000, 'pan': 2000, 'tam': 2000, 'tel': 2000}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'ben': 2.525, 'eng': 0.977, 'guj': 2.503, 'hin': 1.892, 'kan': 2.35, 'mal': 2.116, 'mar': 5.618, 'pan': 2.509, 'tam': 2.404, 'tel': 2.291}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 30528
- Total training wall-clock time: 1.886 s
- Mean tokens/sec (measured, CPU): 16190.27
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.597186803817749
- Final validation loss: 6.593474
- Overall perplexity: 726.0022
- Macro-average per-language perplexity: 727.1128
- Weighted-average per-language perplexity: 727.1128

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 6.570102 | 713.4429 | 2944 |
| eng | 6.725191 | 833.1312 | 2944 |
| guj | 6.537273 | 690.4012 | 2944 |
| hin | 6.559644 | 706.0206 | 2944 |
| kan | 6.601808 | 736.4252 | 2944 |
| mal | 6.601137 | 735.9312 | 2944 |
| mar | 6.51422 | 674.6674 | 2944 |
| pan | 6.587458 | 725.9333 | 2944 |
| tam | 6.617117 | 747.7862 | 2944 |
| tel | 6.56158 | 707.3886 | 2944 |

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
