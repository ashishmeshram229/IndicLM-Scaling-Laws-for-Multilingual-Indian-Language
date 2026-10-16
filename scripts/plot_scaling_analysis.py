"""
Comprehensive scaling-law analysis plots for IndicLM V100 sweep.

Generates eight figures in experiments/scaling_results_v100/:
  fig1_scaling_curves.png       — L vs N (left) and L vs D (right), log-log
  fig2_loss_heatmap.png         — (N, D) heatmap of mean val loss
  fig3_fit_quality.png          — predicted vs actual + residuals
  fig4_seed_stability.png       — error bars across all 16 (N, D) cells
  fig5_compute_frontier.png     — IsoFLOP curves + compute-optimal trajectory
  fig6_loss_surface.png         — contour plot of fitted L(N, D)
  fig7_loss_per_token.png       — editorial: 4 model curves, loss vs tokens
  fig8_budget_split.png         — compute-optimal budget split vs English rule
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "scaling_results_v100"
FIT_FILE    = RESULTS_DIR / "scaling_law_fit.json"
OBS_FILE    = RESULTS_DIR / "observations.json"
AGG_FILE    = RESULTS_DIR / "seed_aggregation.json"

# ---------------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------------
with open(FIT_FILE) as f:
    fit = json.load(f)

with open(OBS_FILE) as f:
    obs = json.load(f)

with open(AGG_FILE) as f:
    agg = json.load(f)

A     = fit["A"]
alpha = fit["alpha"]
B     = fit["B"]
beta  = fit["beta"]
# L_inf is effectively 0 in the free fit; treat as 0
L_inf = 0.0


def predicted_loss(N, D):
    return L_inf + A / N**alpha + B / D**beta


# ---------------------------------------------------------------------------
# structured data
# Nominal D tiers (some runs have tiny batch-rounding offsets; snap to tier)
# ---------------------------------------------------------------------------
D_TIERS = [50_000_000, 200_000_000, 500_000_000, 1_000_000_000]
N_SIZES = [923_008, 10_329_344, 49_317_376, 123_303_168]

def snap_D(d):
    return min(D_TIERS, key=lambda t: abs(d - t))

# Build (N, D_tier) -> list of losses from raw observations
from collections import defaultdict
cell_losses: dict = defaultdict(list)
for r in obs:
    n = r["n_params_non_embedding"]
    d = snap_D(r["d_tokens"])
    cell_losses[(n, d)].append(r["final_val_loss"])

# seed-aggregated grid: (N, D_tier) -> (mean, std)
agg_grid: dict = {}
for (n, d), losses in cell_losses.items():
    a = np.array(losses)
    agg_grid[(n, d)] = (float(a.mean()), float(a.std()))

N_vals = N_SIZES
D_vals = D_TIERS

# colour palettes
PALETTE_N = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
PALETTE_D = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

N_labels = ["1M", "18M", "65M", "148M"]
D_labels = ["50M tok", "200M tok", "500M tok", "1B tok"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def log_fmt(val, pos=None):
    exp = int(math.log10(val))
    return f"$10^{{{exp}}}$"


def si_fmt(val, pos=None):
    if val >= 1e9:
        return f"{val/1e9:.0f}B"
    if val >= 1e6:
        return f"{val/1e6:.0f}M"
    return str(int(val))


# ---------------------------------------------------------------------------
# Figure 1: Scaling curves  (L vs N  |  L vs D)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("IndicLM V100 Scaling Sweep — Validation Loss Scaling", fontsize=14, y=1.01)

# — left: L vs N for each D —
ax = axes[0]
N_dense = np.logspace(np.log10(min(N_vals)) * 0.9, np.log10(max(N_vals)) * 1.05, 200)
for i, D in enumerate(D_vals):
    y_data = [agg_grid[(N, D)][0] for N in N_vals]
    y_err  = [agg_grid[(N, D)][1] for N in N_vals]
    ax.errorbar(N_vals, y_data, yerr=y_err, fmt="o", color=PALETTE_D[i],
                capsize=4, label=D_labels[i], zorder=3)
    ax.plot(N_dense, [predicted_loss(N, D) for N in N_dense],
            "--", color=PALETTE_D[i], alpha=0.7, linewidth=1.5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Non-embedding parameters  N", fontsize=12)
ax.set_ylabel("Validation loss  L", fontsize=12)
ax.set_title(f"L vs Model Size  (α = {alpha:.3f})", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(si_fmt))
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.legend(title="Data scale", fontsize=9)
ax.grid(True, which="both", alpha=0.3)

# — right: L vs D for each N —
ax = axes[1]
D_dense = np.logspace(np.log10(min(D_vals)) * 0.9, np.log10(max(D_vals)) * 1.05, 200)
for i, N in enumerate(N_vals):
    y_data = [agg_grid[(N, D)][0] for D in D_vals]
    y_err  = [agg_grid[(N, D)][1] for D in D_vals]
    ax.errorbar(D_vals, y_data, yerr=y_err, fmt="s", color=PALETTE_N[i],
                capsize=4, label=N_labels[i], zorder=3)
    ax.plot(D_dense, [predicted_loss(N, D) for D in D_dense],
            "--", color=PALETTE_N[i], alpha=0.7, linewidth=1.5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Training tokens  D", fontsize=12)
ax.set_ylabel("Validation loss  L", fontsize=12)
ax.set_title(f"L vs Data Scale  (β = {beta:.3f})", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(si_fmt))
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.legend(title="Model size", fontsize=9)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
fig.savefig(RESULTS_DIR / "fig1_scaling_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig1_scaling_curves.png")

# ---------------------------------------------------------------------------
# Figure 2: (N, D) heatmap of mean validation loss
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

grid = np.array([[agg_grid[(N, D)][0] for D in D_vals] for N in N_vals])
im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto",
               vmin=grid.min(), vmax=grid.max())

ax.set_xticks(range(len(D_vals)))
ax.set_xticklabels(D_labels, fontsize=10)
ax.set_yticks(range(len(N_vals)))
ax.set_yticklabels(N_labels, fontsize=10)
ax.set_xlabel("Data scale  D (tokens)", fontsize=12)
ax.set_ylabel("Model size  N (params)", fontsize=12)
ax.set_title("Mean Validation Loss — (N, D) Grid\nIndicLM V100 Sweep (48 runs, 3 seeds per cell)", fontsize=12)

for i in range(len(N_vals)):
    for j in range(len(D_vals)):
        mean, std = agg_grid[(N_vals[i], D_vals[j])]
        ax.text(j, i, f"{mean:.3f}\n±{std:.3f}", ha="center", va="center",
                fontsize=9, color="black")

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("Val loss", fontsize=11)

plt.tight_layout()
fig.savefig(RESULTS_DIR / "fig2_loss_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig2_loss_heatmap.png")

# ---------------------------------------------------------------------------
# Figure 3: Fit quality — predicted vs actual  +  residuals
# ---------------------------------------------------------------------------
actual    = np.array([r["final_val_loss"]         for r in obs])
predicted = np.array([predicted_loss(r["n_params_non_embedding"], r["d_tokens"]) for r in obs])
residuals = actual - predicted

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Scaling Law Fit Quality  (R² = {fit['r_squared']:.4f}, n=48)", fontsize=13)

# predicted vs actual
ax = axes[0]
mn, mx = min(actual.min(), predicted.min()) * 0.98, max(actual.max(), predicted.max()) * 1.02
ax.scatter(actual, predicted, c="#1f77b4", s=40, alpha=0.8, zorder=3, label="runs")
ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.5, label="perfect fit")
ax.fill_between([mn, mx],
                [mn * 0.95, mx * 0.95],
                [mn * 1.05, mx * 1.05],
                alpha=0.12, color="gray", label="±5% band")
ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
ax.set_xlabel("Actual validation loss", fontsize=12)
ax.set_ylabel("Predicted validation loss", fontsize=12)
ax.set_title("Predicted vs Actual", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

rmse = np.sqrt(np.mean(residuals**2))
mae  = np.mean(np.abs(residuals))
ax.text(0.05, 0.95, f"RMSE = {rmse:.4f}\nMAE  = {mae:.4f}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

# residuals
ax = axes[1]
ax.axhline(0, color="k", linewidth=1.2, linestyle="--")
ax.scatter(actual, residuals, c="#d62728", s=40, alpha=0.8, zorder=3)
ax.fill_between([mn, mx], [-0.1, -0.1], [0.1, 0.1],
                alpha=0.12, color="green", label="±0.1 nats")
ax.set_xlim(mn, mx)
ax.set_xlabel("Actual validation loss", fontsize=12)
ax.set_ylabel("Residual  (actual − predicted)", fontsize=12)
ax.set_title("Residuals", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

std_res = residuals.std()
ax.text(0.05, 0.05, f"σ residuals = {std_res:.4f} nats",
        transform=ax.transAxes, va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
fig.savefig(RESULTS_DIR / "fig3_fit_quality.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig3_fit_quality.png")

# ---------------------------------------------------------------------------
# Figure 4: Seed stability — error bars for all 16 (N, D) cells
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5))

cell_labels = []
cell_means  = []
cell_stds   = []
cell_colors = []

for i, N in enumerate(N_vals):
    for j, D in enumerate(D_vals):
        mean, std = agg_grid[(N, D)]
        cell_labels.append(f"{N_labels[i]}\n{D_labels[j]}")
        cell_means.append(mean)
        cell_stds.append(std)
        cell_colors.append(PALETTE_N[i])

x = np.arange(len(cell_labels))
bars = ax.bar(x, cell_means, color=cell_colors, alpha=0.75, zorder=2)
ax.errorbar(x, cell_means, yerr=cell_stds, fmt="none",
            ecolor="black", capsize=4, elinewidth=1.5, zorder=3)

# add std annotation above each bar
for xi, (mean, std) in enumerate(zip(cell_means, cell_stds)):
    ax.text(xi, mean + std + 0.03, f"σ={std:.3f}", ha="center",
            va="bottom", fontsize=6.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(cell_labels, fontsize=7.5)
ax.set_ylabel("Mean validation loss  (3 seeds)", fontsize=12)
ax.set_title("Seed Stability Across All 16 (N, D) Cells\nError bars = ±1σ over 3 seeds", fontsize=12)
ax.grid(True, axis="y", alpha=0.3)

# custom legend for model sizes
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=PALETTE_N[i], alpha=0.75, label=N_labels[i])
                   for i in range(len(N_vals))]
ax.legend(handles=legend_elements, title="Model size", fontsize=9, loc="upper right")

plt.tight_layout()
fig.savefig(RESULTS_DIR / "fig4_seed_stability.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig4_seed_stability.png")

# ---------------------------------------------------------------------------
# Figure 5: Compute-optimal frontier (IsoFLOP / Chinchilla-style)
# FLOPs C ≈ 6 * N_non_emb * D.
# Minimise L(N,D) = A/N^α + B/D^β s.t. C=6ND:
#   dL/dN=0  =>  N*(α+β) = (α·A)/(β·B) · (C/6)^β
#   N* = [(α·A)/(β·B)]^(1/(α+β)) · (C/6)^(β/(α+β))
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

# Compute budget range covered by our sweep
C_min_sweep = 6 * min(N_vals) * min(D_vals)   # ≈ 2.8e14
C_max_sweep = 6 * max(N_vals) * max(D_vals)   # ≈ 7.4e17

# IsoFLOP lines: span one decade below to one above our budgets
log_C_min = math.floor(math.log10(C_min_sweep)) - 1
log_C_max = math.ceil(math.log10(C_max_sweep)) + 1
n_iso = 7
C_grid = np.logspace(log_C_min, log_C_max, n_iso)
iso_colors = plt.cm.Blues(np.linspace(0.25, 0.80, n_iso))

N_range = np.logspace(np.log10(min(N_vals)) * 0.6, np.log10(max(N_vals)) * 1.5, 400)
for C, color in zip(C_grid, iso_colors):
    D_for_N = C / (6.0 * N_range)
    mask = (D_for_N >= min(D_vals) * 0.1) & (D_for_N <= max(D_vals) * 10)
    if mask.sum() > 1:
        ax.plot(N_range[mask], D_for_N[mask], color=color, linewidth=1.0,
                linestyle="-", alpha=0.6)
        idx = np.where(mask)[0][-1]
        exp = int(round(math.log10(C)))
        ax.text(N_range[idx] * 1.05, D_for_N[idx], f"$10^{{{exp}}}$",
                fontsize=7, color=color, va="center", alpha=0.8)

# Compute-optimal trajectory (corrected formula):
# N*(α+β) = (α·A)/(β·B) · (C/6)^β
exp_N = beta / (alpha + beta)   # exponent for N* vs C
exp_D = alpha / (alpha + beta)  # exponent for D* vs C
# k_N = [(α·A)/(β·B)]^(1/(α+β)) / 6^(β/(α+β))
ratio_correct = (alpha * A) / (beta * B)
k_N = ratio_correct ** (1.0 / (alpha + beta)) / (6.0 ** exp_N)

C_traj = np.logspace(math.log10(C_min_sweep) - 0.5,
                      math.log10(C_max_sweep) + 0.5, 300)
N_star = k_N * C_traj ** exp_N
D_star = C_traj / (6.0 * N_star)
mask = ((N_star >= min(N_vals) * 0.05) & (N_star <= max(N_vals) * 20) &
        (D_star >= min(D_vals) * 0.05) & (D_star <= max(D_vals) * 20))
ax.plot(N_star[mask], D_star[mask], "r-", linewidth=2.5, zorder=5,
        label=f"Compute-optimal frontier\n"
              f"N*∝C^{exp_N:.2f},  D*∝C^{exp_D:.2f}")

# Annotate the frontier at our four compute budgets
for N_m, D_m, N_lbl in zip(N_vals, D_vals, N_labels):
    C_m = 6 * N_m * D_m
    N_opt = k_N * C_m ** exp_N
    D_opt = C_m / (6.0 * N_opt)
    if (min(N_vals)*0.1 < N_opt < max(N_vals)*10 and
            min(D_vals)*0.1 < D_opt < max(D_vals)*10):
        ax.annotate(f"N*={N_opt/1e6:.1f}M\nD*={D_opt/1e6:.0f}M",
                    xy=(N_opt, D_opt), xytext=(N_opt * 0.35, D_opt * 2.5),
                    fontsize=6.5, color="darkred",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))

# Measured (N, D) grid points
for i, N in enumerate(N_vals):
    for j, D in enumerate(D_vals):
        ax.scatter(N, D, marker="D", s=70, color=PALETTE_N[i],
                   edgecolors="k", linewidths=0.7, zorder=6)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Non-embedding parameters  N", fontsize=12)
ax.set_ylabel("Training tokens  D", fontsize=12)
ax.set_title("Compute-Optimal Frontier  (Chinchilla-style)\n"
             f"L = A/N^α + B/D^β   |   α={alpha:.3f},  β={beta:.3f}", fontsize=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(si_fmt))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(si_fmt))
ax.grid(True, which="both", alpha=0.2)

# legend: model sizes + frontier
handles = [plt.Line2D([0], [0], color="r", linewidth=2.5,
                       label=f"Compute-optimal\n(N*∝C^{exp_N:.2f})")]
handles += [plt.scatter([], [], marker="D", s=60, color=PALETTE_N[i],
                        edgecolors="k", linewidths=0.7, label=N_labels[i])
            for i in range(len(N_vals))]
ax.legend(handles=handles, fontsize=9, loc="upper right")

plt.tight_layout()
fig.savefig(RESULTS_DIR / "fig5_compute_frontier.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig5_compute_frontier.png")

# ---------------------------------------------------------------------------
# Figure 6: Loss surface contour  L̂(N, D)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

N_mesh = np.logspace(np.log10(min(N_vals)) * 0.85, np.log10(max(N_vals)) * 1.1, 300)
D_mesh = np.logspace(np.log10(min(D_vals)) * 0.85, np.log10(max(D_vals)) * 1.1, 300)
NN, DD = np.meshgrid(N_mesh, D_mesh)
LL = predicted_loss(NN, DD)

levels = np.linspace(LL.min() * 0.98, min(LL.max(), 8.0), 20)
cf = ax.contourf(NN, DD, LL, levels=levels, cmap="RdYlGn_r", alpha=0.85)
cs = ax.contour(NN, DD, LL, levels=levels, colors="k", linewidths=0.4, alpha=0.4)
ax.clabel(cs, fmt="%.2f", fontsize=7, inline=True)

cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
cbar.set_label("Predicted L(N, D)", fontsize=11)

# overlay measured points (color = actual loss)
losses = np.array([agg_grid[(N, D)][0] for N in N_vals for D in D_vals])
Ns_flat = np.array([N for N in N_vals for D in D_vals])
Ds_flat = np.array([D for N in N_vals for D in D_vals])
sc = ax.scatter(Ns_flat, Ds_flat, c=losses, cmap="RdYlGn_r",
                vmin=LL.min() * 0.98, vmax=min(LL.max(), 8.0),
                s=120, edgecolors="k", linewidths=1.2, zorder=5, marker="D",
                label="Measured (N, D) cells")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Non-embedding parameters  N", fontsize=12)
ax.set_ylabel("Training tokens  D", fontsize=12)
ax.set_title("Fitted Loss Surface  L̂(N, D) = A/N^α + B/D^β\n"
             f"A={A:.1f}, α={alpha:.3f}  |  B={B:.0f}, β={beta:.3f}", fontsize=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(si_fmt))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(si_fmt))
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, which="both", alpha=0.2)

plt.tight_layout()
fig.savefig(RESULTS_DIR / "fig6_loss_surface.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig6_loss_surface.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nAll figures written to", RESULTS_DIR)
print(f"  Fit: L = {A:.2f}/N^{alpha:.4f} + {B:.0f}/D^{beta:.4f}  (R²={fit['r_squared']:.4f})")

# ---------------------------------------------------------------------------
# Figure 7: "Loss per token" — 4 model curves vs training tokens
# Mirrors the top panel in the editorial graphic.
# ---------------------------------------------------------------------------
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(9, 6))

model_labels_editorial = ["tiny model (1M)", "small model (18M)",
                           "medium model (65M)", "large model (148M)"]
model_colors = ["#a8c8e8", "#5a9fd4", "#2a6aad", "#0d3b7a"]

D_x = D_vals   # [50M, 200M, 500M, 1B]

for i, N in enumerate(N_vals):
    means = [agg_grid[(N, D)][0] for D in D_vals]
    errs  = [agg_grid[(N, D)][1] for D in D_vals]

    ax.plot(D_x, means, "o-", color=model_colors[i], linewidth=2.0,
            markersize=6, zorder=3, label=model_labels_editorial[i])
    ax.fill_between(D_x,
                    [m - e for m, e in zip(means, errs)],
                    [m + e for m, e in zip(means, errs)],
                    color=model_colors[i], alpha=0.15)

    # label at right end
    ax.text(D_x[-1] * 1.04, means[-1], model_labels_editorial[i],
            va="center", fontsize=9, color=model_colors[i])

# annotation arrow
ax.annotate("more text,\nfewer mistakes",
            xy=(500_000_000, agg_grid[(49_317_376, 500_000_000)][0]),
            xytext=(120_000_000, 2.85),
            fontsize=9, color="#c0392b", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5))

ax.set_xscale("log")
ax.set_xlabel("Training text read (tokens)", fontsize=12)
ax.set_ylabel("Loss per token  (lower = fewer mistakes)", fontsize=12)
ax.set_title("Bigger models and more text both cut mistakes.\n"
             "One formula matches all 48 models.  "
             r"$L = A/N^{\alpha} + B/D^{\beta}$   R²=0.984",
             fontsize=11)
ax.set_xticks(D_vals)
ax.set_xticklabels(["50M", "200M", "500M", "1B"], fontsize=10)
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.grid(True, which="both", alpha=0.25)
ax.set_xlim(D_vals[0] * 0.7, D_vals[-1] * 2.5)
ax.set_ylim(2.5, 7.2)

# hide individual legend (labels are inline)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "fig7_loss_per_token.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig7_loss_per_token.png")

# ---------------------------------------------------------------------------
# Figure 8: "Where should a fixed budget go?" — horizontal bar chart
# Compute-optimal exponents: N*∝C^(β/(α+β)), D*∝C^(α/(α+β))
# English (Chinchilla, α≈β≈0.5): 50/50.
# Our Indic LM: 82.5% model / 17.5% data.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 5.5),
                          gridspec_kw={"hspace": 0.55})
fig.suptitle("Where should a fixed compute budget go?",
             fontsize=14, fontweight="bold", y=1.01)

exp_N_pct = beta / (alpha + beta) * 100   # 82.5
exp_D_pct = alpha / (alpha + beta) * 100  # 17.5
eng_pct   = 50.0

rows = [
    ("English rule\n(Chinchilla, α≈β)", eng_pct,   100 - eng_pct,   None,   "#6c757d",  "#b0bec5"),
    (f"Indian languages\n(our result)", exp_N_pct, exp_D_pct,
     f"+{exp_N_pct-eng_pct:.0f} pp", "#2a6aad",  "#f0a070"),
]

for ax, (label, model_pct, data_pct, delta, col_m, col_d) in zip(axes, rows):
    ax.barh([0], [model_pct], color=col_m, height=0.55, label="bigger model")
    ax.barh([0], [data_pct],  left=[model_pct], color=col_d, height=0.55, label="more text")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center", labelpad=8)
    ax.axvline(50, color="k", linewidth=1.0, linestyle="--", alpha=0.5)

    # percentage labels inside bars
    ax.text(model_pct / 2, 0, f"bigger model\n{model_pct:.0f}%",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.text(model_pct + data_pct / 2, 0, f"more text\n{data_pct:.0f}%",
            ha="center", va="center", fontsize=9,
            color="white" if col_d == "#2a6aad" else "#333333", fontweight="bold")

    if delta:
        ax.annotate(f"→ {delta}",
                    xy=(eng_pct, 0.27), fontsize=10,
                    color="#c0392b", fontweight="bold")

# shared legend
handles = [mpatches.Patch(color="#2a6aad", label="bigger model"),
           mpatches.Patch(color="#f0a070", label="more text")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.08))

# bottom callout box
callout = ("Indic languages: put much more into model size.\n"
           "Data returns diminish 6× faster than model returns\n"
           "across all 4 model sizes tested  (β=0.53 >> α=0.11).")
fig.text(0.5, -0.20, callout, ha="center", va="top", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.5", fc="#fff3cd", ec="#e0a800", lw=1.5),
         transform=fig.transFigure)

plt.tight_layout()
fig.savefig(RESULTS_DIR / "fig8_budget_split.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig8_budget_split.png")
