# Scaling-Law Experiments

## Method

`indiclm experiment scaling-sweep` trains a grid of model sizes x token
budgets on the same corpus/mixture and fits

```
L(N, D) ≈ A / N^alpha + B / D^beta + L_infinity
```

via `scipy.optimize.curve_fit` (nonlinear least squares, bounded), where
`N` is non-embedding parameters and `D` is tokens actually consumed
during training (`max_steps * micro_batch_size * grad_accum * seq_len`).
The fit is **empirical** — it describes whatever (N, D, L) points were
actually measured, with standard errors from the parameter covariance
matrix, not an assumed universal law. Fewer than 5 observations (the
number of free parameters) is reported as `insufficient_data` rather than
fit.

## What was actually run (EXP-001/002/003/012)

4 model sizes (`n_tiny` 55K params, `n_small` 96K, `n_medium` 159K,
`n_large` 396K, all non-embedding-param counts about half those figures)
x 2 token budgets (~8.2K and ~23.9K tokens actually consumed) x **3
seeds each (0, 1, 2)** — 24 runs total, all on CPU, all on the real
Wikipedia-sourced corpus described in `docs/data_pipeline.md`, alpha=0.7
temperature mixture. Full observations and the per-grid-point seed
aggregation: `experiments/manifests/EXP-012/scaling_law_fit.json` and
`experiments/manifests/EXP-012/seed_aggregation.json`.

| N (non-embed) | D (tokens) | mean val loss | std across 3 seeds |
|---|---|---|---|
| 24,736 | 8,192 | 7.0856 | 0.0039 |
| 24,736 | 23,872 | 7.0519 | 0.0062 |
| 50,928 | 8,192 | 7.0765 | 0.0087 |
| 50,928 | 23,872 | 7.0101 | 0.0126 |
| 98,624 | 8,192 | 7.0571 | 0.0026 |
| 98,624 | 23,872 | 6.9416 | 0.0040 |
| 304,800 | 8,192 | 7.0235 | 0.0048 |
| 304,800 | 23,872 | 6.8295 | 0.0177 |

Loss decreases monotonically with both model size and token budget at
this scale — the qualitative direction scaling laws predict. Seed-to-seed
standard deviation is small (0.003-0.018 nats, well under 1% of the loss
values themselves) at every grid point — training itself is not the
noisy part of this experiment; see "Fit result" for what is.

### Fit result

```
fit_status: ok
alpha = 0.008 ± 0.350     (statistically indistinguishable from 0 — see below)
beta  = 1.872 ± 0.074     (near the fit's upper bound of 2.0)
R^2   = 0.845
n_observations = 24 (8 grid points x 3 seeds)
```

**This fit should still not be trusted as an estimate of a real scaling
exponent, but multi-seed reruns pin down *why* more precisely than the
single-seed version could.** Going from 1 seed (8 points) to 3 seeds (24
points) tightened `alpha`'s standard error from 0.75 to 0.35 — real
information gain — but `alpha`'s point estimate (0.008) barely moved and
stayed far smaller than its own uncertainty. Combined with the very low
per-grid-point seed variance above, this rules out "the single-seed run
was just an unlucky draw" as the explanation: the flat, unidentifiable
`alpha` is a genuine property of this grid, not sampling noise. The
actual cause is structural, as before — only 4 N-values and 2 D-values
(a 3x spread) is a narrow, coarse grid for a 5-parameter model to
identify two exponents independently from, and `beta` sitting near its
search bound is the classic symptom of the D-term absorbing variance the
N-term couldn't. The methodology (grid sweep -> multi-seed -> nonlinear
fit -> honest uncertainty reporting) is real and reusable; the numeric
alpha/beta are not evidence about Indic-language scaling behavior.
Widening the grid (more N-values, a wider D range) is the direct fix for
the fit itself — see "Next experiment" below; multi-seeding was the fix
for "is the noise coming from training stochasticity or the grid" and
that question is now answered.

### Plot

`experiments/manifests/EXP-012/loss_vs_params.png` — loss vs. parameters
on a log-x axis, measured points only (the fit line is meaningful only
insofar as the caveats above are kept in mind).

## Compute accounting

Only measured wall-clock and measured tokens/sec are reported (see table
above) — no FLOPs/MFU estimate is computed in this milestone, since doing
so honestly requires accounting for the exact attention/FFN op count per
architecture variant (GQA, MoE) and is not yet implemented. This is
tracked as a gap, not silently omitted: `indiclm experiment run`'s
`report.md` explicitly states "FLOPs estimate: not computed."

## Next experiment

Widen the sweep: 6-8 model sizes across a 20x parameter range, 4+ token
budgets across a 10x range, to properly identify both exponents
independently. This also requires a corpus large enough that larger
token budgets don't force many repeated epochs over the same documents
— `data/raw/wiki_sample/` is ~1,800 real paragraphs now (see
`docs/data_pipeline.md`), better than the original hand-authored sample
but still not large relative to the D values a wider budget sweep would
need. Multi-seeding (`--seeds 0 1 2 3 4`, or any list) is already wired
up via `indiclm experiment scaling-sweep`'s `seeds` option and
`aggregate_by_grid_point`; a wider grid should reuse it rather than
reintroduce a single-seed run.
