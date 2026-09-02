# IndicLM Architecture

## Status

This document reflects the current working state of the repository (all 12
milestones complete). Every module listed below has real, tested code unless
explicitly noted otherwise.

## System diagram

```
Data -> Tokenizer -> Model -> Training -> Evaluation
                       |
                 Experiment Hub
                       |
                Scaling Analysis
```

## Environment

Recorded here for honesty/reproducibility, not as a hardware requirement:

- Python 3.11.15
- No CUDA GPU (CPU-only training throughout all experiments)
- 2 CPU cores, ~7.8 GB RAM, ~30 GB writable disk
- Recommended profile: `configs/profiles/cpu.yaml`

All scaling experiments (EXP-001 through EXP-012) were run on this hardware.
Model sizes range from 12K to 554K non-embedding parameters (~43× range).
The resulting scaling-law exponents are a demonstration of methodology, not
scientific claims about Indic-language scaling laws — this is stated explicitly
in every report this codebase produces.

## Module map

- `src/indiclm/utils/` — `structlog`-based logging (JSON in non-interactive
  contexts), hardware detection, `get_logger`. **[implemented]**
- `src/indiclm/cli/` — Typer-based CLI; subcommands: `doctor`, `data`,
  `tokenizer`, `train`, `evaluate`, `experiment`, `ablation`, `scaling`,
  `report`, `serve`. **[implemented]**
- `src/indiclm/data/` — ingestion, Unicode normalization, language ID,
  rule-based and hybrid quality filtering, exact + MinHash near-dedup +
  TF-IDF semantic dedup, contamination detection, mixture engine, sharding.
  **[implemented]**
- `src/indiclm/tokenizer/` — BPE / SentencePiece-Unigram / byte-level
  training and benchmarking. **[implemented]**
- `src/indiclm/models/` — decoder-only Transformer from scratch: RoPE,
  RMSNorm, SwiGLU, GQA, optional MoE; `ModelConfig` includes analytic
  parameter count, FLOPs-per-forward, and FLOPs-per-token. **[implemented]**
- `src/indiclm/training/` — training loop, gradient accumulation, AdamW,
  cosine-warmup scheduler, gradient clipping, checkpointing/resume, mixed
  precision (bf16 on CUDA, fp32 on CPU), anomaly detection. **[implemented]**
- `src/indiclm/distributed/` — DDP setup helpers (`init_distributed`,
  `cleanup_distributed`, `wrap_ddp`, rank/world-size queries). Safe to
  import on single-process runs; activates only when `WORLD_SIZE > 1`.
  **[implemented — requires multi-GPU environment to activate DDP]**
- `src/indiclm/evaluation/` — perplexity evaluation, sentiment downstream
  task, contamination scanning (64-doc probe per experiment run). **[implemented]**
- `src/indiclm/experiments/` — experiment tracking, manifest generation,
  multi-seed runner, scaling sweeps (24-point grid × 3 seeds = 72 runs),
  three scaling-law fits: 5-param free, 4-param fixed-L_inf, 2-param
  Chinchilla-style `L ≈ C/(N·D)^γ`. Dashboard and report generation.
  **[implemented]**
- `src/indiclm/inference/` — FastAPI serving layer, separate from training
  code. **[implemented]**
- `src/indiclm/monitoring/` — anomaly detection (NaN loss, exploding
  gradients, stalled dataloader). **[implemented]**

## Configuration system

`configs/` holds YAML for `model/`, `tokenizer/`, `data/`, `training/`,
`evaluation/`, `experiments/`, `infrastructure/`, and `profiles/`
(hardware-aware resource profiles). No experiment parameter is hard-coded in
application code.

## Milestone plan

0. Inspect repo/environment — **done**
1. Repository foundation: packaging, config, logging, hardware detection,
   CLI skeleton, CI, first tests — **done**
2. Data pipeline — **done**
3. Tokenizer — **done**
4. Transformer model — **done**
5. Single-GPU training — **done** (CPU; bf16 code path wired for CUDA)
6. Evaluation — **done**
7. Experiment tracking — **done**
8. Scaling experiments (EXP-001 – EXP-012) — **done**
9. Distributed training — **done** (stub; activates with `WORLD_SIZE > 1`)
10. Inference API — **done**
11. Dashboard / report generation — **done**
12. Research report + cleanup — **done**
