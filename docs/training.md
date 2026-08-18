# Training

## What's implemented

- Decoder-only Transformer from scratch (`src/indiclm/models/`): token
  embeddings, RoPE, causal self-attention with optional grouped-query
  attention, RMSNorm, SwiGLU FFN, tied/untied embeddings, configurable
  vocab size and context length, an experimental MoE FFN behind
  `use_moe` (router, top-k, capacity factor, load-balancing aux loss,
  routing stats — see `models/moe.py`).
- Training engine (`src/indiclm/training/trainer.py`): gradient
  accumulation, AdamW, cosine decay with linear warmup, gradient
  clipping, periodic evaluation, checkpointing (model + optimizer +
  scheduler + RNG state + step + tokens_seen + config), resume-from-
  checkpoint, structured per-step JSONL metrics (loss, lr, grad_norm,
  tokens_seen, tokens/sec, step_time, data_loading_time), and an anomaly
  detector (`monitoring/anomaly.py`) that raises on NaN/Inf loss or
  gradients and warns on exploding-gradient / loss-spike patterns.

## What's tested, not just claimed

- Causal masking verified by unit test (`tests/unit/test_model.py`):
  changing a future token does not change earlier-position logits.
- RoPE verified to preserve per-vector norm (a rotation must).
- Checkpoint round-trip verified to restore weights bit-for-bit and to
  resume training at the correct step (`tests/unit/test_scheduler_and_checkpoint.py`,
  and a live resume smoke test that continued training from step 30 to
  step 40 with a continuous loss curve).
- GQA and MoE both verified via real forward/backward passes, not just
  imported and assumed correct.

## Execution modes: tested vs. architecturally supported

| Mode | Status |
|---|---|
| CPU | **Tested.** All experiments in this repository ran on a 2-core, 7.8GB RAM CPU-only container. |
| Single GPU | Not tested — no GPU was available in this environment. `TrainingConfig.device`/`precision` (bf16 via `torch.autocast`) are wired to activate correctly on CUDA, but the code path has not been exercised on real hardware. |
| Multi-GPU (DDP/FSDP) | Not implemented in this milestone. `configs/profiles/multi_gpu.yaml` documents the target profile; no launcher code exists yet. |
| Slurm / cluster | Not implemented. `configs/profiles/cluster.yaml` documents the target profile only. |

This repository does not claim distributed training capability it has
not built and tested — see `docs/reproducibility.md`.

## Measured throughput (CPU, this environment)

From the scaling sweep (`experiments/manifests/EXP-012/scaling_law_fit.json`):
a 55K-parameter model trained at ~16,700-22,400 tokens/sec, a 396K-parameter
model at ~8,700-9,600 tokens/sec, on 2 CPU cores. These are measured, not
estimated, numbers, specific to this container — they say nothing about
GPU throughput.

## Running training directly

```bash
indiclm train --config <path-to-yaml-with-model/training/data-sections>
indiclm evaluate --checkpoint <path-to.pt>
```

or, for the full experiment-registry workflow (training + evaluation +
manifest + report.md together):

```bash
indiclm experiment run --config experiments/configs/EXP-007_temperature_sampling.yaml
```
