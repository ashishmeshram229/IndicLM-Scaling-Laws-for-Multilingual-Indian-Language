# Reproducibility

## What every experiment records

`indiclm experiment run` writes, under `experiments/manifests/<experiment_id>/`:

- `manifest.json` — git commit, config hash (sha256 of the sorted config),
  full config, dataset version, tokenizer version, seed, hardware
  (`indiclm doctor`'s detection output), software versions (Python,
  PyTorch, platform), training tokens, final train/val loss, evaluation
  metrics, checkpoint path.
- `metrics.jsonl` — per-step loss, learning rate, gradient norm,
  tokens_seen, tokens/sec, step time, data-loading time.
- `tracked_metrics.jsonl` / `params.json` — the same run through the
  `ExperimentTracker` abstraction (`experiments/tracking.py`), which is
  always-on locally; an optional MLflow adapter exists behind the same
  interface but is not configured by default, per "never rely solely on
  a hosted dashboard."
- `checkpoints/final.pt` (and periodic `step_N.pt`) — model, optimizer,
  scheduler, RNG state, step, tokens_seen, config.
- `evaluation.json` — overall + per-language loss/perplexity.
- `report.md` — the lab-notebook-style writeup (hypothesis, setup,
  dataset, model, compute, variables, results, uncertainty, failure
  cases, interpretation, limitations, next experiment), generated from
  the objects above, not hand-written per run.

## Reproducing a specific result

```bash
git checkout <git_commit from manifest.json>
indiclm data prepare --raw-dir data/raw --output-dir data/processed
indiclm tokenizer train --input-dir data/processed --output-dir data/tokenizer_v1
indiclm experiment run --config experiments/configs/<matching-config>.yaml
```

Given the same config, seed, and corpus, results should match up to
floating-point nondeterminism in PyTorch's CPU kernels (not pinned to
bit-exact reproducibility in this milestone — no
`torch.use_deterministic_algorithms(True)` pass has been validated).

## What "tested" means in this repository

Tested = actually run in this environment and checked (via assertions or
manual inspection of output) to behave as claimed. Sections above and in
`docs/architecture.md`/`docs/training.md` mark each capability as tested,
architecturally-supported-but-unvalidated, or not-implemented. Docker
build was **not** verified in this environment — the sandbox this
repository was built in has the `docker` CLI installed but no reachable
Docker daemon (`/var/run/docker.sock` does not exist), so `Dockerfile`/
`docker-compose.yml` are written to the same conventions as the rest of
the repo but have not been build-tested. Verify with `docker build -t
indiclm .` before relying on them.

## Known environment this repository was built and evaluated in

- Python 3.11.15, PyTorch 2.13.0+cpu, no CUDA GPU, 2 CPU cores, 7.8GB RAM.
- All 40 unit + integration tests pass (`pytest -q`) in this environment.
- All 12 experiments in the registry (EXP-001 through EXP-012) were run;
  none are marked "not available" in `docs/research_report.md`.

## Limitations that affect every result in this repository

1. The training corpus (`data/raw/wiki_sample/`) is ~1,800 real
   Wikipedia paragraphs across 10 languages — real text, not
   hand-authored placeholders, but still a small bootstrap-scale sample,
   not a production pretraining corpus. See `docs/data_pipeline.md` for
   exactly how it was sourced and what license it carries. No result
   here should be read as evidence about real Indic-language model
   quality at production scale.
2. All training is CPU-only; throughput numbers do not reflect GPU
   performance, and no GPU/multi-GPU/cluster code path has been
   exercised (see `docs/training.md`).
3. The scaling-law fit (`docs/scaling_laws.md`) is underdetermined given
   only 8 (N, D) grid points — the methodology is real, the fitted
   exponents are not trustworthy point estimates. This is now backed by
   3 seeds per grid point (24 observations), which confirmed the
   underdetermination is structural (a coarse 4x2 grid), not sampling
   noise — see `docs/scaling_laws.md`'s "Fit result" for the reasoning.
4. Single seed per experiment everywhere except the scaling sweep
   (EXP-001/002/003/012, which now run 3 seeds per grid point via
   `indiclm experiment scaling-sweep --seeds`). The mixture experiments
   (EXP-005 through EXP-011) and tokenizer/data ablations (EXP-004,
   EXP-008/009) remain single-seed; no confidence intervals across seeds
   for those.
