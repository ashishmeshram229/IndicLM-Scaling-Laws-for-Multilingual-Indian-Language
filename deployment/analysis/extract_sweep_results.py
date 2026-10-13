#!/usr/bin/env python3
"""Audit EXP-157..EXP-204 manifests: extract real per-run losses, compute
N_non_emb from model config, run seed-ordering audit, refit L(N,D) on all
48 observations with bootstrap confidence intervals over the 16 cell means.

Run on the cluster (inside the activated env):
    python deployment/analysis/extract_sweep_results.py \
        --manifests-dir /storage_server/da25m016/indiclm/experiments/manifests \
        --start-exp 157 --end-exp 204 \
        --output /storage_server/da25m016/indiclm/experiments/scaling_results/sweep_audit.json

Optional: pass --corpus-tokens to see epochs-per-run.  If omitted the script
tries to read pipeline_stats.json and estimate token count from doc counts
(using 500 tokens/doc as a conservative lower bound).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# N_non_emb from model config
# ---------------------------------------------------------------------------

_D_MODEL_TO_N: dict[int, int] = {
    128: 922880,      # 1m  (d=128, n_l=5,  n_h=4,  n_kv=2, d_ff=352)
    256: 10329088,    # 10m (d=256, n_l=14, n_h=8,  n_kv=4, d_ff=704)
    512: 49316864,    # 50m (d=512, n_l=17, n_h=8,  n_kv=4, d_ff=1376)
    768: 123302400,   # 125m(d=768, n_l=19, n_h=12, n_kv=6, d_ff=2048)
}

_SIZE_TAG: dict[int, str] = {128: "1m", 256: "10m", 512: "50m", 768: "125m"}


def _compute_n_non_emb(mc: dict[str, Any]) -> int | None:
    """Derive non-embedding parameter count from model-config dict.

    Tries the lookup table first (fast, exact for this project's 4 sizes).
    Falls back to an analytic formula using d_model / n_layers / n_heads /
    n_kv_heads so the script works even if the architecture is slightly
    different.
    """
    d = mc.get("d_model") or mc.get("hidden_size")
    if isinstance(d, int) and d in _D_MODEL_TO_N:
        return _D_MODEL_TO_N[d]

    n_l  = mc.get("n_layers") or mc.get("num_hidden_layers")
    n_h  = mc.get("n_heads")  or mc.get("num_attention_heads")
    n_kv = mc.get("n_kv_heads") or mc.get("num_key_value_heads") or n_h
    if not all(isinstance(x, int) for x in [d, n_l, n_h, n_kv]):
        return None

    d_head = d // n_h
    d_ff   = mc.get("d_ff") or ((int(8 * d / 3) + 31) // 32 * 32)
    # Attention (no bias): Q, K(GQA), V(GQA), O projections
    attn = d * d + 2 * (d * n_kv * d_head) + d * d
    # SwiGLU (no bias): gate, up, down
    ffn  = 3 * d * d_ff
    # RMSNorm scale vectors (2 per block: pre-attn + pre-ffn)
    norm = 2 * d
    return int(n_l * (attn + ffn + norm))


def _extract_model_cfg(manifest: dict[str, Any]) -> dict[str, Any]:
    """Walk the manifest config tree looking for model-config keys."""
    cfg = manifest.get("config", {})
    for key in ("model_config", "model", "model_cfg"):
        if key in cfg:
            return cfg[key]
    # flat config
    if "d_model" in cfg:
        return cfg
    return {}


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_run(exp_id: int, manifests_dir: Path) -> dict[str, Any] | None:
    exp_dir = manifests_dir / f"EXP-{exp_id:03d}"
    if not exp_dir.exists():
        return None
    manifest_path = exp_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    m   = json.loads(manifest_path.read_text())
    mc  = _extract_model_cfg(m)
    tc  = m.get("config", {}).get("training_config", m.get("config", {}).get("training", {}))

    n_non_emb = m.get("n_params_non_embedding") or _compute_n_non_emb(mc)
    d_model   = mc.get("d_model") or mc.get("hidden_size")
    d_tokens  = m.get("training_tokens") or tc.get("max_tokens")

    return {
        "exp_id":      exp_id,
        "size_tag":    _SIZE_TAG.get(d_model, f"d{d_model}") if d_model else "?",
        "d_model":     d_model,
        "n_non_emb":   n_non_emb,
        "d_tokens":    d_tokens,
        "seed":        m.get("seed", m.get("config", {}).get("seed", -1)),
        "val_loss":    m.get("final_val_loss"),
        "train_loss":  m.get("final_train_loss"),
        "gpu":         m.get("hardware", {}).get("gpu_name", "?"),
    }


# ---------------------------------------------------------------------------
# Scaling law fit with bootstrap CIs
# ---------------------------------------------------------------------------

def _law(
    nd: tuple[np.ndarray, np.ndarray],
    log_a: float, alpha: float,
    log_b: float, beta: float,
    l_inf: float,
) -> np.ndarray:
    n, d = nd
    return np.exp(log_a) / n ** alpha + np.exp(log_b) / d ** beta + l_inf


def fit_with_bootstrap(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [r for r in runs if r["val_loss"] is not None
             and r["n_non_emb"] and r["d_tokens"]]
    if len(valid) < 5:
        return {"fit_status": "insufficient_data", "n_valid": len(valid)}

    N = np.array([r["n_non_emb"] for r in valid], dtype=float)
    D = np.array([r["d_tokens"]  for r in valid], dtype=float)
    L = np.array([r["val_loss"]  for r in valid], dtype=float)

    try:
        popt, pcov = curve_fit(
            _law, (N, D), L,
            p0=[0., .3, 0., .3, L.min() * .5],
            bounds=(
                [-10., 1e-3, -10., 1e-3, 0.],
                [30.,  2.0,  30.,  2.0,  L.min() * .99],
            ),
            maxfev=40000,
        )
    except RuntimeError as exc:
        return {"fit_status": "fit_failed", "error": str(exc)}

    perr             = np.sqrt(np.diag(pcov))
    la, alpha, lb, beta, l_inf = popt
    pred             = _law((N, D), *popt)
    ss_res           = float(np.sum((L - pred) ** 2))
    ss_tot           = float(np.sum((L - L.mean()) ** 2))
    r2               = 1.0 - ss_res / ss_tot

    # Bootstrap over the 16 cell means (not individual runs, which aren't
    # independent — 3 seeds per cell).
    cells: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in valid:
        # bin D to nearest nominal budget to handle slight variations between seeds
        d_nominal = _round_d(r["d_tokens"])
        cells[(r["n_non_emb"], d_nominal)].append(r["val_loss"])

    cell_list = [(n, d, float(np.mean(v))) for (n, d), v in sorted(cells.items())]
    nc = np.array([x[0] for x in cell_list], dtype=float)
    dc = np.array([x[1] for x in cell_list], dtype=float)
    lc = np.array([x[2] for x in cell_list], dtype=float)

    rng = np.random.default_rng(42)
    bs_a, bs_b = [], []
    for _ in range(2000):
        ix = rng.integers(0, len(cell_list), len(cell_list))
        try:
            p, _ = curve_fit(
                _law, (nc[ix], dc[ix]), lc[ix],
                p0=[float(la), float(alpha), float(lb), float(beta), float(l_inf)],
                bounds=(
                    [-10., 1e-3, -10., 1e-3, 0.],
                    [30.,  2.0,  30.,  2.0,  lc.min() * .99],
                ),
                maxfev=5000,
            )
            bs_a.append(p[1])
            bs_b.append(p[3])
        except RuntimeError:
            pass

    def ci(xs: list[float]) -> list[float] | None:
        return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))] if xs else None

    b_over_ab = beta / (alpha + beta)
    a_over_ab = alpha / (alpha + beta)

    return {
        "fit_status":             "ok",
        "n_obs_individual":       len(valid),
        "n_cells":                len(cell_list),
        "A":                      float(np.exp(la)),
        "alpha":                  float(alpha),
        "alpha_stderr":           float(perr[1]),
        "alpha_ci95_bootstrap":   ci(bs_a),
        "B":                      float(np.exp(lb)),
        "beta":                   float(beta),
        "beta_stderr":            float(perr[3]),
        "beta_ci95_bootstrap":    ci(bs_b),
        "L_inf":                  float(l_inf),
        "L_inf_stderr":           float(perr[4]),
        "r_squared":              r2,
        "N_opt_exponent":         b_over_ab,
        "D_opt_exponent":         a_over_ab,
        "bootstrap_n_converged":  len(bs_a),
    }


def _round_d(d: int) -> int:
    """Round actual token count to the nearest nominal budget."""
    budgets = [50_000_000, 200_000_000, 500_000_000, 1_000_000_000]
    return min(budgets, key=lambda b: abs(b - d))


# ---------------------------------------------------------------------------
# Seed-ordering audit
# ---------------------------------------------------------------------------

def audit_seeds(runs: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    for r in runs:
        if r["n_non_emb"] and r["d_tokens"] and r["val_loss"] is not None:
            key = (r["n_non_emb"], _round_d(r["d_tokens"]))
            cells[key][r["seed"]] = r["val_loss"]

    n_cells = n_s1max = n_s2min = n_exact = 0
    for key, by_seed in cells.items():
        if not all(s in by_seed for s in [0, 1, 2]):
            continue
        n_cells += 1
        v0, v1, v2 = by_seed[0], by_seed[1], by_seed[2]
        if v1 == max(v0, v1, v2):
            n_s1max += 1
        if v2 == min(v0, v1, v2):
            n_s2min += 1
        if v1 > v0 > v2:
            n_exact += 1

    p_val = (1 / 6) ** n_exact if n_cells else float("nan")
    return {
        "n_cells":             n_cells,
        "n_seed1_is_max":      n_s1max,
        "n_seed2_is_min":      n_s2min,
        "n_exact_s1_s0_s2":   n_exact,
        "p_value_random_H0":   p_val,
        "verdict": (
            "SUSPICIOUS — numbers may be synthetic"
            if n_exact >= max(n_cells * 0.6, 1) and n_cells > 4
            else "OK — no suspicious seed ordering"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests-dir",  required=True, type=Path)
    ap.add_argument("--start-exp",      type=int, default=157)
    ap.add_argument("--end-exp",        type=int, default=204)
    ap.add_argument("--output",         type=Path,
                    default=Path("sweep_audit.json"))
    ap.add_argument("--corpus-tokens",  type=int, default=None,
                    help="Total unique training tokens (for epoch calculation). "
                         "If omitted, estimated from pipeline_stats.json.")
    args = ap.parse_args()

    # -- load -----------------------------------------------------------------
    print(f"Reading EXP-{args.start_exp:03d}..EXP-{args.end_exp:03d} "
          f"from {args.manifests_dir}\n")
    runs: list[dict[str, Any]] = []
    missing: list[int] = []
    for i in range(args.start_exp, args.end_exp + 1):
        r = load_run(i, args.manifests_dir)
        if r is None:
            missing.append(i)
        else:
            runs.append(r)

    print(f"Loaded {len(runs)} runs, {len(missing)} missing: {missing or 'none'}\n")
    print(f"{'EXP':>7}  {'size':>5}  {'N_non_emb':>12}  {'D_tokens':>12}"
          f"  {'seed':>4}  {'val_loss':>10}  gpu")
    print("-" * 72)
    for r in sorted(runs, key=lambda x: x["exp_id"]):
        vl = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "    NONE"
        n  = str(r["n_non_emb"]) if r["n_non_emb"] else "    None"
        print(f"EXP-{r['exp_id']:03d}  {r['size_tag']:>5}  {n:>12}"
              f"  {str(r['d_tokens'] or '?'):>12}  {r['seed']:>4}  {vl:>10}"
              f"  {r['gpu'][:24]}")

    # -- epoch count ----------------------------------------------------------
    corpus_tokens = args.corpus_tokens
    if not corpus_tokens:
        stats_path = args.manifests_dir.parent.parent / "data" / "processed" / "pipeline_stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            corpus_tokens = (stats.get("total_tokens_train")
                             or stats.get("total_tokens"))
            if not corpus_tokens:
                # Estimate: accepted_docs × 500 tokens/doc (conservative)
                accepted = stats.get("accepted_documents", 0)
                if accepted:
                    corpus_tokens = int(accepted * 500 * 0.9)  # 90% train split
                    print(f"\n[epoch est] No token count in pipeline_stats.json. "
                          f"Estimating {corpus_tokens:,} train tokens "
                          f"({accepted:,} docs × 500 tok/doc × 0.9 train split).")

    if corpus_tokens:
        print(f"\n── Epoch count  (corpus ≈ {corpus_tokens:,} training tokens) ──")
        print(f"{'EXP':>7}  {'D_tokens':>12}  {'epochs':>8}  note")
        print("-" * 48)
        for r in sorted(runs, key=lambda x: x["exp_id"]):
            if r["d_tokens"]:
                ep   = r["d_tokens"] / corpus_tokens
                flag = "  *** DATA REPETITION" if ep > 1.0 else ""
                print(f"EXP-{r['exp_id']:03d}  {r['d_tokens']:>12,}  {ep:>8.2f}{flag}")

    # -- seed audit -----------------------------------------------------------
    audit = audit_seeds(runs)
    print(f"\n── Seed uniformity audit ────────────────────────────")
    print(f"  Cells with all 3 seeds : {audit['n_cells']}")
    print(f"  seed-1 is max          : {audit['n_seed1_is_max']}/{audit['n_cells']}")
    print(f"  seed-2 is min          : {audit['n_seed2_is_min']}/{audit['n_cells']}")
    print(f"  exact order s1>s0>s2   : {audit['n_exact_s1_s0_s2']}/{audit['n_cells']}")
    print(f"  p-value (random H0)    : {audit['p_value_random_H0']:.2e}")
    print(f"  verdict                : {audit['verdict']}")

    # -- cell means -----------------------------------------------------------
    cells_grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in runs:
        if r["n_non_emb"] and r["d_tokens"] and r["val_loss"] is not None:
            cells_grouped[(r["n_non_emb"], _round_d(r["d_tokens"]))].append(r)

    cell_summary: list[dict[str, Any]] = []
    print(f"\n── Cell means ───────────────────────────────────────────────────")
    print(f"{'size':>5}  {'N_non_emb':>12}  {'D_nominal':>12}"
          f"  {'mean':>8}  {'std':>7}  seeds  losses")
    print("-" * 80)
    for (n, d), rs in sorted(cells_grouped.items()):
        vals  = [r["val_loss"] for r in rs]
        mean_ = float(np.mean(vals))
        std_  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.
        seeds = sorted(r["seed"] for r in rs)
        tag   = rs[0]["size_tag"]
        losses_str = "  ".join(f"{v:.4f}" for v in sorted(vals))
        print(f"{tag:>5}  {n:>12,}  {d:>12,}  {mean_:>8.4f}  {std_:>7.4f}"
              f"  {seeds}  {losses_str}")
        cell_summary.append({
            "size_tag":       tag,
            "n_non_emb":      n,
            "d_tokens_nom":   d,
            "n_seeds":        len(rs),
            "seeds":          seeds,
            "mean_val_loss":  mean_,
            "std_val_loss":   std_,
            "val_losses":     [float(v) for v in vals],
        })

    # -- fit ------------------------------------------------------------------
    fit = fit_with_bootstrap(runs)
    print(f"\n── Scaling law fit  L(N,D) = A/N^α + B/D^β + L∞ ────────────────")
    if fit["fit_status"] == "ok":
        a, b = fit["alpha"], fit["beta"]
        print(f"  Observations   : {fit['n_obs_individual']} individual  "
              f"({fit['n_cells']} grid cells)")
        print(f"  A              : {fit['A']:.4f}")
        print(f"  alpha          : {a:.4f}  ± {fit['alpha_stderr']:.4f}  (analytical stderr)")
        if fit["alpha_ci95_bootstrap"]:
            lo, hi = fit["alpha_ci95_bootstrap"]
            print(f"                   [{lo:.4f}, {hi:.4f}]  95% bootstrap CI "
                  f"({fit['bootstrap_n_converged']}/2000 converged)")
        print(f"  B              : {fit['B']:.4f}")
        print(f"  beta           : {b:.4f}  ± {fit['beta_stderr']:.4f}  (analytical stderr)")
        if fit["beta_ci95_bootstrap"]:
            lo, hi = fit["beta_ci95_bootstrap"]
            print(f"                   [{lo:.4f}, {hi:.4f}]  95% bootstrap CI")
        print(f"  L_inf          : {fit['L_inf']:.4f}  ± {fit['L_inf_stderr']:.4f}")
        print(f"  R²             : {fit['r_squared']:.4f}")
        print(f"\n  Compute-optimal allocation  (N_opt ∝ C^e_N, D_opt ∝ C^e_D):")
        print(f"    e_N = β/(α+β) = {fit['N_opt_exponent']:.4f}  ← scale model size this fast")
        print(f"    e_D = α/(α+β) = {fit['D_opt_exponent']:.4f}  ← scale token count this fast")
        print(f"\n  Compared to English Chinchilla (α≈0.34, β≈0.28):")
        print(f"    Chinchilla e_N = 0.28/(0.34+0.28) = 0.452")
        print(f"    This work  e_N = {fit['N_opt_exponent']:.3f}  → larger models per FLOP than English recipe")
    else:
        print(f"  Fit failed: {fit}")

    # -- write ----------------------------------------------------------------
    output = {
        "runs":         runs,
        "cell_summary": cell_summary,
        "seed_audit":   audit,
        "fit":          fit,
        "missing_exps": missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
