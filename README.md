# IndicLM

A research platform for studying how model scale, training-token budget,
tokenizer efficiency, data quality, and multilingual data composition
interact when training language models for Indian languages.

**Status: Milestone 1 (repository foundation) — data pipeline, tokenizer,
model, training engine, evaluation, and experiment tooling are not yet
implemented.** This README will be updated as each milestone lands; see
`docs/` for the living architecture and `experiments/` for what has
actually been run (nothing yet — no results in this repo are fabricated).

## Motivation

Training a competent multilingual model is not just "write a Transformer
and scale it up." It requires controlled experiments across data quality,
tokenization, mixture ratios, and compute. IndicLM is built to run and
document those experiments end-to-end and reproducibly, for at least:
English, Hindi, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam,
Gujarati, and Punjabi — with new languages addable without touching core
code.

## Research question

How do model scale, training-token budget, tokenizer efficiency, data
quality, and multilingual data composition interact when training
language models for Indian languages? See `docs/scaling_laws.md` (stub)
and the experiment registry in `experiments/configs/` for the specific
questions this is designed to answer.

## Architecture

```
Data -> Tokenizer -> Model -> Training -> Evaluation
                       |
                 Experiment Hub
                       |
                Scaling Analysis
```

Full breakdown in `docs/architecture.md`.

## Quick start

```bash
git clone <this-repo>
cd indiclm
make dev-install     # pip install -e ".[dev]" + pre-commit
make test            # unit tests (CPU-only, no GPU required)
indiclm doctor        # inspect Python/PyTorch/CUDA/GPU/disk/memory
```

Data preparation, tokenizer training, model training, evaluation, and
experiment commands will be documented here as each milestone lands
(see Roadmap).

## Hardware requirements

`indiclm doctor` detects your hardware and recommends a profile from
`configs/profiles/` (`cpu`, `8gb_gpu`, `16gb_gpu`, `24gb_gpu`,
`multi_gpu`, `cluster`). This repository does not claim training scale
beyond what has actually been run and logged under `experiments/`.

## Reproducibility

Every experiment (once the experiment runner lands) will record: git
commit, full config, seed, dataset version, tokenizer version, model
config, training tokens, hardware, software versions, checkpoint, and
evaluation results, as a machine-readable manifest alongside a
human-readable report. See `docs/reproducibility.md`.

## Limitations (current)

- No GPU was available in the environment this repository was scaffolded
  in; all early validation targets CPU-only correctness, not throughput.
- No data pipeline, tokenizer, model, training engine, or evaluation
  code exists yet — this milestone is repository foundation only
  (config system, logging, hardware detection, CLI skeleton, CI, tests).

## Roadmap

See the milestone breakdown in `docs/architecture.md`. In order: data
pipeline -> tokenizer -> Transformer model -> single-GPU training ->
evaluation -> experiment tracking -> scaling experiments -> distributed
training -> inference API -> dashboard/report generation -> final
research report.
