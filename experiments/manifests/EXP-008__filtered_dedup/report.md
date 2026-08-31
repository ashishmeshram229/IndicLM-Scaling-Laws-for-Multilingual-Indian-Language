# Experiment Report: EXP-008__filtered_dedup

## 1. Hypothesis

(not specified in experiment config)

## 2. Experimental setup

- Git commit: `unknown (not a git repository or git unavailable)`
- Config hash: `6ea28ec3efbfb69b`
- Seed: `0`
- Hardware: `cpu`, 2 CPU cores,
  7.84 GB RAM, 0 GPU(s)
- Software: Python 3.11.15, PyTorch 2.13.0+cu130

## 3. Dataset

- Dataset version: `filtered_dedup`
- Tokenizer version: `bpe_v1`
- Sequence length: 64
- Total packed tokens: 19955
- Tokens per language: {'ben': 1663, 'eng': 3924, 'guj': 1735, 'hin': 2190, 'kan': 1896, 'mal': 2012, 'mar': 1060, 'pan': 1735, 'tam': 1885, 'tel': 1900}
- Epochs per language (token budget / available tokens; >1 means the
  small bootstrap corpus was repeated to reach the budget): {'ben': 2.313, 'eng': 1.6, 'guj': 2.271, 'hin': 2.054, 'kan': 2.187, 'mal': 2.131, 'mar': 2.804, 'pan': 2.271, 'tam': 2.192, 'tel': 2.184}
- Padding ratio: 0.0

## 4. Model

```
{'d_model': 64, 'n_layers': 2, 'n_heads': 4, 'n_kv_heads': 2, 'dropout': 0.0}
```

## 5. Compute

- Training tokens seen: 30528
- Total training wall-clock time: 1.765 s
- Mean tokens/sec (measured, CPU): 17291.85
- FLOPs estimate: not computed in this report (see `docs/scaling_laws.md`
  compute-accounting note); wall-clock and tokens/sec above are measured,
  not estimated.

## 6. Variables

```
{'max_steps': 60, 'micro_batch_size': 4, 'gradient_accumulation_steps': 2, 'learning_rate': 0.0003, 'warmup_steps': 10, 'eval_every': 15, 'checkpoint_every': 60, 'log_every': 15, 'device': 'cpu'}
```

## 7. Results

- Final training loss: 6.7970921993255615
- Final validation loss: 6.814866
- Overall perplexity: 901.9895
- Macro-average per-language perplexity: 903.9558
- Weighted-average per-language perplexity: 903.9558

| Language | Loss | Perplexity | Tokens evaluated |
|---|---|---|---|
| ben | 6.939447 | 1032.1995 | 2944 |
| eng | 6.707908 | 818.8558 | 2944 |
| guj | 6.788484 | 887.5672 | 2944 |
| hin | 6.782088 | 881.9086 | 2944 |
| kan | 6.800922 | 898.6751 | 2944 |
| mal | 6.774989 | 875.6698 | 2944 |
| mar | 6.750673 | 854.6338 | 2944 |
| pan | 6.867737 | 960.772 | 2944 |
| tam | 6.75964 | 862.3318 | 2944 |
| tel | 6.874141 | 966.9447 | 2944 |

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
