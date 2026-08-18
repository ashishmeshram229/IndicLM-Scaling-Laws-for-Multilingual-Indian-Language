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
x 2 token budgets (~8.2K and ~23.9K tokens actually consumed), all on
CPU, all on the bootstrap corpus described in `docs/data_pipeline.md`,
alpha=0.7 temperature mixture. Full observations:
`experiments/manifests/EXP-012/scaling_law_fit.json`.

| run_id | N (non-embed) | D (tokens) | final val loss | tokens/sec |
|---|---|---|---|---|
| n_tiny_d8000 | 24,736 | 8,192 | 6.836 | 16,711 |
| n_small_d8000 | 50,928 | 8,192 | 6.826 | 16,646 |
| n_medium_d8000 | 98,624 | 8,192 | 6.796 | 14,726 |
| n_large_d8000 | 304,800 | 8,192 | 6.750 | 8,726 |
| n_tiny_d24000 | 24,736 | 23,872 | 6.792 | 22,421 |
| n_small_d24000 | 50,928 | 23,872 | 6.739 | 18,456 |
| n_medium_d24000 | 98,624 | 23,872 | 6.661 | 16,194 |
| n_large_d24000 | 304,800 | 23,872 | 6.533 | 9,580 |

Loss decreases monotonically with both model size and token budget at
this scale — the qualitative direction scaling laws predict.

### Fit result

```
fit_status: ok
alpha = 0.011 ± 0.753     (essentially unconstrained — see below)
beta  = 1.997 ± 0.122     (pinned near the fit's upper bound of 2.0)
R^2   = 0.879
```

**This fit should not be trusted as an estimate of a real scaling
exponent.** With only 4 N-values and 2 D-values (8 points total, 5 free
parameters), the fit is underdetermined: `alpha`'s standard error (0.75)
is larger than its point estimate, and `beta` sits at the search bound,
which is the classic symptom of the D-dependence term absorbing residual
variance the N-dependence term could not identify. Two D-values 3x apart
is also a narrow range for isolating a power law. The methodology (grid
sweep -> nonlinear fit -> honest uncertainty reporting) is real and
reusable; the numeric alpha/beta are not evidence about Indic-language
scaling behavior. Running more model sizes and a wider spread of token
budgets is the direct fix — see "Next experiment" below.

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
independently. At real (non-bootstrap) corpus scale this also requires a
corpus large enough that larger token budgets don't force many repeated
epochs over the same few hundred documents.
