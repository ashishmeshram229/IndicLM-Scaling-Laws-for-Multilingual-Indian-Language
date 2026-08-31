# IndicLM

[![CI](https://github.com/ashishmeshram229/IndicLM-Scaling-Laws-for-Multilingual-Indian-Language/actions/workflows/ci.yml/badge.svg)](https://github.com/ashishmeshram229/IndicLM-Scaling-Laws-for-Multilingual-Indian-Language/actions/workflows/ci.yml)

A research platform for studying how model scale, training-token budget,
tokenizer efficiency, data quality, and multilingual data composition
interact when training language models for Indian languages.

**Status: end-to-end pipeline implemented and run** — data pipeline,
tokenizer training/benchmarking, a from-scratch decoder-only Transformer,
a real training engine (checkpointing, resume, mixed precision path,
anomaly detection), standalone evaluation (perplexity + a zero-shot
sentiment downstream task, see `docs/evaluation.md`), an
experiment-tracking + manifest + report system, a scaling-law sweep with
a fitted (if underdetermined) power law, a static HTML research
dashboard, a FastAPI inference service, Docker/Compose files, CI, and 40
passing unit/integration tests. All 12 registry experiments (EXP-001
through EXP-012) have been run — see `docs/research_report.md` and
`experiments/manifests/`.

**Read this before trusting any number below:** every experiment in this
repository trains on ~1,800 real Wikipedia paragraphs across 10
languages (see `docs/data_pipeline.md` for exactly how that corpus was
fetched and licensed), on CPU only, with tiny (55K-400K parameter)
models. The text is real, not fabricated, but the scale still isn't —
the *infrastructure* is real and reusable; the *numbers* are a
methodology demonstration, not evidence about real Indic-language model
quality at production scale. Every doc in `docs/` says this again in
context so it isn't missed.

## Motivation

Training a competent multilingual model is not just "write a Transformer
and scale it up." It requires controlled experiments across data quality,
tokenization, mixture ratios, and compute. IndicLM runs and documents
those experiments end-to-end and reproducibly, for English, Hindi,
Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, and
Punjabi — with new languages addable without touching core code (the
pipeline is script- and language-agnostic; see `src/indiclm/data/langid.py`).

## Research question

How do model scale, training-token budget, tokenizer efficiency, data
quality, and multilingual data composition interact when training
language models for Indian languages? `docs/scaling_laws.md` and
`docs/research_report.md` report what was actually measured toward each
of the ten sub-questions in the original spec; `experiments/configs/`
holds every experiment's config, version-controlled.

## Architecture

```
Data -> Tokenizer -> Model -> Training -> Evaluation
                       |
                 Experiment Hub
                       |
                Scaling Analysis
```

Full breakdown, including what's tested vs. architecturally-supported-
but-unvalidated (GPU, multi-GPU, Slurm), in `docs/architecture.md`.

## Quick start

```bash
git clone <this-repo> && cd indiclm
make dev-install                 # pip install -e ".[dev]" + pre-commit
make test                        # 40 unit + integration tests, CPU-only
indiclm doctor                    # inspect Python/PyTorch/CUDA/GPU/disk/memory

indiclm data prepare              # data/raw -> data/processed (+ stats)
indiclm data stats
indiclm tokenizer train --vocab-size 1200
indiclm tokenizer benchmark

indiclm experiment run --config experiments/configs/EXP-007_temperature_sampling.yaml
indiclm experiment scaling-sweep  # EXP-001/002/003/012
indiclm experiment tokenizer-ablation   # EXP-004
indiclm experiment data-ablation        # EXP-008, EXP-009
indiclm experiment compare --experiments EXP-005 EXP-006 EXP-007
indiclm evaluate-downstream --checkpoint experiments/manifests/EXP-007/checkpoints/final.pt \
  --out-path experiments/manifests/EXP-007/downstream_evaluation.json   # zero-shot sentiment eval
indiclm report generate            # docs/research_report.md
indiclm report dashboard            # docs/dashboard.html (static, no deps)

indiclm serve --checkpoint experiments/manifests/EXP-003/checkpoints/final.pt
# then: curl localhost:8000/health, POST /generate, POST /tokenize
```

## Hardware requirements

`indiclm doctor` detects your hardware and recommends a profile from
`configs/profiles/` (`cpu`, `8gb_gpu`, `16gb_gpu`, `24gb_gpu`,
`multi_gpu`, `cluster`). Every result in this repo was produced on the
`cpu` profile (2 cores, 7.8GB RAM, no GPU) — see `docs/training.md` for
exactly what's tested vs. supported-but-unvalidated on other profiles.

## Reproducibility

Every `indiclm experiment run` records git commit, config hash, seed,
dataset version, tokenizer version, model config, training tokens,
hardware, software versions, checkpoint path, and evaluation metrics in
`experiments/manifests/<id>/manifest.json`, alongside a human-readable
`report.md`. See `docs/reproducibility.md` for the full reproduction
recipe and an explicit list of what "tested" means in this repo.

## Limitations

See `docs/reproducibility.md` "Limitations that affect every result" —
in short: real but bootstrap-scale corpus (~1,800 Wikipedia paragraphs,
not a production pretraining corpus), CPU-only, an underdetermined
scaling-law fit (methodology real, exponents not trustworthy at 8 grid
points — now confirmed via 3-seed reruns to be a structural grid-coarseness
issue, not sampling noise), single seed everywhere except the scaling
sweep, and Docker build unverified (no Docker daemon in the build
environment).

## Roadmap

Implemented: data pipeline, tokenizer, model, training, evaluation
(perplexity + a zero-shot sentiment downstream task), experiment
tracking, scaling experiments, inference API, a static HTML research
dashboard, CI, tests. Not yet implemented: multi-GPU/Slurm execution
(architecture documented, untested), model-based quality/contamination
scoring, downstream tasks beyond sentiment classification (QA,
translation, summarization) against public Indic benchmarks. See
`docs/architecture.md` for the full map.
