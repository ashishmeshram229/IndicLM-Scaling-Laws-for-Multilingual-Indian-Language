# IndicLM Architecture

## Status

This document reflects Milestone 1 (repository foundation). Sections
marked **[planned]** describe the target design and will be updated to
**[implemented]** as each milestone lands. Nothing in this document
should be read as a claim that the corresponding code exists yet unless
marked implemented.

## System diagram

```
Data -> Tokenizer -> Model -> Training -> Evaluation
                       |
                 Experiment Hub
                       |
                Scaling Analysis
```

## Environment this repository was scaffolded in

Recorded here for honesty/reproducibility, not as a hardware requirement:

- Python 3.11.15
- No CUDA GPU detected (`torch` not yet installed at scaffold time)
- 2 CPU cores, ~7.8 GB RAM, ~30 GB writable disk available
- Recommended profile: `configs/profiles/cpu.yaml`

Because of this, early milestones target **correctness on CPU with tiny
(≈5M param) models**, not throughput or scale. Larger-scale claims must
be backed by an actual run recorded under `experiments/manifests/`.

## Module map [implemented / planned]

- `src/indiclm/utils/` — logging (`structlog`-based, JSON in
  non-interactive contexts), hardware detection. **[implemented, partial]**
- `src/indiclm/cli/` — Typer-based CLI; `doctor` implemented, remaining
  subcommands (`data`, `tokenizer`, `train`, `evaluate`, `experiment`,
  `report`, `serve`) **[planned]**.
- `src/indiclm/data/` — ingestion, normalization, language ID, quality
  filtering, dedup, contamination detection, mixture engine, sharding.
  **[planned — Milestone 2]**
- `src/indiclm/tokenizer/` — BPE / SentencePiece-Unigram / byte-level
  training and benchmarking. **[planned — Milestone 3]**
- `src/indiclm/models/` — decoder-only Transformer implemented from
  scratch (RoPE, RMSNorm, SwiGLU, GQA, optional MoE). **[planned —
  Milestone 4]**
- `src/indiclm/training/` — training loop, checkpointing, mixed
  precision, gradient accumulation. **[planned — Milestone 5]**
- `src/indiclm/distributed/` — DDP/FSDP, Slurm launcher. **[planned —
  Milestone 9]**
- `src/indiclm/evaluation/` — perplexity, downstream tasks, code-mixed
  eval, contamination scanning. **[planned — Milestone 6]**
- `src/indiclm/experiments/` — experiment tracking abstraction, scaling
  sweeps, manifest/report generation. **[planned — Milestones 7-8]**
- `src/indiclm/inference/` — FastAPI serving layer, separate from
  training code. **[planned — Milestone 10]**
- `src/indiclm/monitoring/` — anomaly detection (NaN loss, exploding
  gradients, stalled dataloader). **[planned]**

## Configuration system

`configs/` holds YAML for `model/`, `tokenizer/`, `data/`,
`training/`, `evaluation/`, `experiments/`, `infrastructure/`, and
`profiles/` (hardware-aware resource profiles, **implemented** as
static YAML in this milestone; a Pydantic/OmegaConf-backed loader with
validation is **planned** for Milestone 1 completion). No experiment
parameter is hard-coded in application code.

## Milestone plan

0. Inspect repo/environment — **done** (this document).
1. Repository foundation: packaging, config, logging, hardware
   detection, CLI skeleton, CI, first tests — **in progress**.
2. Data pipeline. 3. Tokenizer. 4. Transformer model. 5. Single-GPU
   training. 6. Evaluation. 7. Experiment tracking. 8. Scaling
   experiments. 9. Distributed training. 10. Inference API.
   11. Dashboard/report generation. 12. Research report + cleanup.

Each milestone will only be marked complete after its tests pass.
