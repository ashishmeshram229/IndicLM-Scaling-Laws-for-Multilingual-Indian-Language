# IndicLM

[![CI](https://github.com/ashishmeshram229/IndicLM-Scaling-Laws-for-Multilingual-Indian-Language/actions/workflows/ci.yml/badge.svg)](https://github.com/ashishmeshram229/IndicLM-Scaling-Laws-for-Multilingual-Indian-Language/actions/workflows/ci.yml)

A research platform for studying how model scale, training-token budget,
tokenizer efficiency, data quality, and multilingual data composition
interact when training language models for Indian languages.

**Status: V100 GPU scaling sweep complete.** 48 training runs
(EXP-157–EXP-204) across 4 model sizes × 4 data scales × 3 seeds were
executed on NVIDIA V100 GPUs. A Chinchilla-style power law was fitted to
all 48 points (R²=0.984). Full results and the plot are in
`experiments/scaling_results_v100/`. Earlier infrastructure work
(data pipeline, tokenizer, Transformer, training engine, evaluation,
inference API, Docker, CI, 40 tests) was validated on CPU with the 12
registry experiments EXP-001–EXP-012.

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
`docs/research_report.md` report what was measured toward each of the
ten sub-questions in the original spec; `experiments/configs/` holds
every experiment config, version-controlled.

## Corpus

Wikipedia dumps for 10 languages were fetched and deduplicated. The
token counter in `experiments/scaling_results_v100/corpus_token_count.json`
gives the precise numbers; key figures:

| Language | Documents | Est. tokens |
|----------|----------:|------------:|
| Bengali  | 139,279   | 100M        |
| English  | 505,163   | 472M        |
| Gujarati |  15,157   |  17M        |
| Hindi    | 121,707   |  72M        |
| Kannada  |  28,721   |  48M        |
| Malayalam|  83,197   |  56M        |
| Marathi  |  54,135   |  25M        |
| Punjabi  |  46,353   |  30M        |
| Tamil    | 153,188   |  75M        |
| Telugu   |  82,126   |  72M        |
| **Total**|**1,229,026**|**~968M**  |

Training split (90%): ~871M tokens. The 1B-token data-scale point in the
sweep required ≈1.15 epochs over the training corpus.

## V100 Scaling Sweep (EXP-157–EXP-204)

### Grid

| Dimension | Values |
|-----------|--------|
| Model sizes (total / non-emb params) | 5M/923K · 18M/10M · 65M/49M · 148M/123M |
| Data scales (tokens) | 50M · 200M · 500M · 1B |
| Seeds per cell | 3 (seeds 0, 1, 2) |
| **Total runs** | **48** |
| Hardware | NVIDIA V100 (32 GB) |

### Scaling law fit

```
L(N, D) = A / N^α  +  B / D^β
```

| Parameter | Value | Std err |
|-----------|------:|--------:|
| A         | 18.38 | —       |
| α (model) | 0.113 | ±0.048  |
| B         | 32,189| —       |
| β (data)  | 0.533 | ±0.072  |
| R²        | 0.984 | —       |
| n points  | 48/48 | —       |

The data exponent (β≈0.53) is well-determined; the model exponent
(α≈0.11) has wider uncertainty because the parameter range (5M–148M)
is narrow relative to production-scale sweeps. See
`experiments/scaling_results_v100/scaling_law_fit.json` for the full
fit object including fixed-L∞ and 2-parameter Chinchilla-style
alternatives.

### Seed stability

Across all 16 (N, D) cells the per-seed standard deviation in
validation loss is below 0.025 nats, confirming the sweep measures
the scaling signal rather than initialization noise.

### Best validation loss per model size (at 1B tokens)

| Total params | Best val loss |
|-------------:|:-------------:|
| 5M           | 4.122         |
| 18M          | 3.494         |
| 65M          | 3.093         |
| 148M         | 2.895         |

Detailed per-run numbers: `experiments/scaling_results_v100/observations.json`  
Seed-aggregated table: `experiments/scaling_results_v100/seed_aggregation.json`  
Loss-vs-params plot: `experiments/scaling_results_v100/loss_vs_params.png`

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
git clone <this-repo> && cd indiclm
make dev-install                 # pip install -e ".[dev]" + pre-commit
make test                        # 40 unit + integration tests, CPU-only
indiclm doctor                   # inspect Python/PyTorch/CUDA/GPU/disk/memory

indiclm data prepare             # data/raw -> data/processed (+ stats)
indiclm data stats
indiclm tokenizer train --vocab-size 1200
indiclm tokenizer benchmark

indiclm experiment run --config experiments/configs/EXP-007_temperature_sampling.yaml
indiclm experiment scaling-sweep          # EXP-001/002/003/012
indiclm experiment tokenizer-ablation     # EXP-004
indiclm experiment data-ablation          # EXP-008, EXP-009
indiclm experiment compare --experiments EXP-005 EXP-006 EXP-007
indiclm evaluate-downstream \
  --checkpoint experiments/manifests/EXP-007/checkpoints/final.pt \
  --out-path experiments/manifests/EXP-007/downstream_evaluation.json

indiclm report generate          # docs/research_report.md
indiclm report dashboard         # docs/dashboard.html (static, no deps)

indiclm serve --checkpoint experiments/manifests/EXP-003/checkpoints/final.pt
# then: curl localhost:8000/health, POST /generate, POST /tokenize
```

### Fetching cluster results

If you re-run the V100 sweep on a SLURM cluster and need to pull the
output files locally:

```bash
# rsync (preferred — incremental, safe to re-run)
rsync -avz --progress \
  <user>@<cluster-host>:/path/to/indiclm/experiments/scaling_results_v100/ \
  experiments/scaling_results_v100/

# scp alternative
scp -r <user>@<cluster-host>:/path/to/indiclm/experiments/scaling_results_v100/ \
  experiments/
```

Replace `<user>`, `<cluster-host>`, and the remote path to match your
cluster. The SLURM sweep scripts live in `scripts/` (see
`scripts/scaling_sweep.sh`).

## Hardware requirements

`indiclm doctor` detects your hardware and recommends a profile from
`configs/profiles/` (`cpu`, `8gb_gpu`, `16gb_gpu`, `24gb_gpu`,
`multi_gpu`, `cluster`).

- **CPU profile** — used for EXP-001–EXP-012 and all CI tests.
- **V100 (32 GB) cluster** — used for the EXP-157–EXP-204 scaling sweep
  via `scripts/scaling_sweep.sh` on a SLURM cluster.

Multi-GPU / DDP code is implemented (fp16/AMP, `torch.nn.parallel.DistributedDataParallel`)
and wired into the SLURM scripts but the 48 sweep runs above used single-GPU
per job. See `docs/training.md` for the full capability map.

## Reproducibility

Every `indiclm experiment run` records git commit, config hash, seed,
dataset version, tokenizer version, model config, training tokens,
hardware, software versions, checkpoint path, and evaluation metrics in
`experiments/manifests/<id>/manifest.json`, alongside a human-readable
`report.md`. See `docs/reproducibility.md` for the full reproduction
recipe.

## Limitations

- **Corpus scale**: ~968M Wikipedia tokens across 10 languages. Real
  production pretraining corpora are 10–100× larger. Scaling exponents
  measured here reflect this data regime; extrapolation beyond it is
  speculative.
- **Model scale**: 5M–148M parameters. The model-scaling exponent (α)
  has wider uncertainty than the data exponent (β) because the
  parameter range is narrow. A sweep to 1B+ params would tighten it.
- **Languages**: 10 languages. Exponents may shift as more low-resource
  Indic languages are added with fewer Wikipedia documents.
- **Downstream tasks**: evaluation is perplexity + one zero-shot
  sentiment task. QA, translation, and summarization benchmarks against
  public Indic datasets are not yet implemented.
- **Docker**: Dockerfile and Compose files are written and reviewed but
  the image was not built in CI (no Docker daemon in the CI environment).

## Roadmap

Done: data pipeline, tokenizer, model, training (fp16/AMP, DDP),
evaluation (perplexity + zero-shot sentiment), experiment tracking,
48-run V100 scaling sweep (EXP-157–EXP-204), scaling law fit (R²=0.984),
inference API, static HTML dashboard, CI, 40 tests.

Not yet implemented: multi-GPU DDP runs (code written, not benchmarked),
model-based quality/contamination scoring, downstream tasks beyond
sentiment classification (QA, translation, summarization against public
Indic benchmarks), Docker image CI build. See `docs/architecture.md`
for the full map.
