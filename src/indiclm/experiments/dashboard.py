"""Static HTML research dashboard.

Per project principle "never rely solely on a hosted dashboard" (see
`experiments/manifest.py`), this reads only artifacts already on disk
(`experiments/manifests/`, `data/processed*/pipeline_stats.json`,
`data/tokenizer_*/benchmark_report.json`) and renders one self-contained
HTML file — no external JS/CSS CDN, no server, no hosted service. It is
deliberately hand-rolled SVG rather than a charting library: the charts
here are simple bars/lines and a dependency would be overkill (project
principle: no overengineering).

Every number in the page is read back from a file the pipeline actually
wrote; sections with no data present say so explicitly rather than being
silently omitted (per data/schema.py's "never silently discard" ethos,
applied here to results instead of documents).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTRY = [
    ("EXP-001", "Baseline tiny model (scaling sweep)"),
    ("EXP-002", "Baseline small model (scaling sweep)"),
    ("EXP-003", "Baseline medium model (scaling sweep)"),
    ("EXP-004", "Tokenizer comparison (BPE vs Unigram)"),
    ("EXP-005", "English-heavy mixture"),
    ("EXP-006", "Indic-balanced mixture"),
    ("EXP-007", "Temperature sampling mixture"),
    ("EXP-008", "Data filtering ablation"),
    ("EXP-009", "Deduplication ablation"),
    ("EXP-010", "Low-resource oversampling"),
    ("EXP-011", "Code-mixed training"),
    ("EXP-012", "Compute-optimal scaling sweep"),
]

_PALETTE = ["#4C6EF5", "#F76707", "#0CA678", "#E64980", "#7048E8", "#F59F00", "#1098AD", "#E03131"]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text())
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


# --------------------------------------------------------------------------
# Minimal inline SVG chart helpers (no charting library dependency).
# --------------------------------------------------------------------------


def _bar_chart(labels: list[str], values: list[float], width: int = 640, bar_h: int = 22) -> str:
    if not values:
        return "<p class='muted'>No data.</p>"
    max_v = max(values) or 1.0
    height = len(values) * (bar_h + 8) + 10
    label_w = 90
    plot_w = width - label_w - 60
    bars = []
    for i, (label, v) in enumerate(zip(labels, values)):
        y = i * (bar_h + 8) + 5
        w = max(2.0, (v / max_v) * plot_w)
        color = _PALETTE[i % len(_PALETTE)]
        bars.append(
            f'<text x="{label_w - 8}" y="{y + bar_h * 0.7}" text-anchor="end" '
            f'class="bar-label">{_esc(label)}</text>'
            f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{color}"/>'
            f'<text x="{label_w + w + 6}" y="{y + bar_h * 0.7}" class="bar-value">{_fmt(v)}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="bar chart">{"".join(bars)}</svg>'
    )


def _line_chart(series: list[tuple[str, list[float]]], width: int = 640, height: int = 200) -> str:
    """series: list of (name, y-values); x is just step index."""
    all_vals = [v for _, ys in series for v in ys if v is not None]
    if not all_vals:
        return "<p class='muted'>No step-level metrics recorded.</p>"
    pad = 24
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    y_min, y_max = min(all_vals), max(all_vals)
    y_range = (y_max - y_min) or 1.0
    paths = []
    legend = []
    for i, (name, ys) in enumerate(series):
        pts = [v for v in ys if v is not None]
        if not pts:
            continue
        n = len(pts)
        color = _PALETTE[i % len(_PALETTE)]
        coords = []
        for j, v in enumerate(pts):
            x = pad + (j / max(1, n - 1)) * plot_w
            y = pad + plot_h - ((v - y_min) / y_range) * plot_h
            coords.append(f"{x:.1f},{y:.1f}")
        paths.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend.append(f'<span class="legend-dot" style="background:{color}"></span>{_esc(name)}')
    axis_labels = (
        f'<text x="{pad}" y="{height - 4}" class="axis-label">{_fmt(y_min)}–{_fmt(y_max)}</text>'
    )
    svg = (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="line chart">'
        f"{''.join(paths)}{axis_labels}</svg>"
    )
    return f'<div class="legend">{" &nbsp; ".join(legend)}</div>{svg}'


# --------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------


def _dataset_section(processed_dir: Path) -> str:
    stats = _load_json(processed_dir / "pipeline_stats.json")
    if stats is None:
        return "<section><h2>Dataset</h2><p class='muted'>No pipeline_stats.json found — run `indiclm data prepare` first.</p></section>"

    lang_dist = stats.get("language_distribution", {})
    labels = sorted(lang_dist, key=lambda k: -lang_dist[k])
    values = [float(lang_dist[k]) for k in labels]

    stat_tiles = "".join(
        f'<div class="tile"><div class="tile-value">{_fmt(v, 4) if isinstance(v, float) else v}</div>'
        f'<div class="tile-label">{_esc(label)}</div></div>'
        for label, v in [
            ("Total documents", stats.get("total_documents")),
            ("Accepted", stats.get("accepted_documents")),
            ("Rejected", stats.get("rejected_documents")),
            ("Duplicate rate", stats.get("duplicate_rate")),
            ("Near-dup rate", stats.get("near_duplicate_rate", 0.0)),
            ("Mean quality score", stats.get("mean_quality_score")),
        ]
    )
    rejection_reasons = stats.get("rejection_reasons", {})
    reasons_html = (
        "".join(f"<li>{_esc(k)}: {v}</li>" for k, v in rejection_reasons.items())
        or "<li class='muted'>None</li>"
    )

    return f"""
    <section>
      <h2>Dataset</h2>
      <div class="tiles">{stat_tiles}</div>
      <h3>Language distribution (accepted documents)</h3>
      {_bar_chart(labels, values)}
      <h3>Rejection reasons</h3>
      <ul>{reasons_html}</ul>
    </section>
    """


def _tokenizer_section(data_dir: Path) -> str:
    variants = {
        "bpe": data_dir / "tokenizer_v1" / "benchmark_report.json",
        "unigram": data_dir / "tokenizer_unigram" / "benchmark_report.json",
    }
    tables = []
    for name, path in variants.items():
        report = _load_json(path)
        if report is None:
            continue
        rows = "".join(
            f"<tr><td>{_esc(lang)}</td><td>{_fmt(v.get('tokens_per_char'), 4)}</td>"
            f"<td>{_fmt(v.get('tokens_per_word'), 4)}</td><td>{_fmt(v.get('compression_ratio'), 4)}</td>"
            f"<td>{_fmt(v.get('unk_rate'), 4)}</td></tr>"
            for lang, v in sorted(report.items())
        )
        tables.append(
            f"<h3>{_esc(name)}</h3>"
            "<table><thead><tr><th>Language</th><th>Tokens/char</th><th>Tokens/word</th>"
            "<th>Compression ratio</th><th>UNK rate</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    if not tables:
        return "<section><h2>Tokenizer</h2><p class='muted'>No tokenizer benchmark reports found — run `indiclm tokenizer train` and `indiclm tokenizer benchmark`.</p></section>"
    return f"<section><h2>Tokenizer</h2>{''.join(tables)}</section>"


@dataclass
class _ExperimentView:
    exp_id: str
    description: str
    status: str
    final_val_loss: float | None
    overall_perplexity: float | None
    macro_avg_perplexity: float | None
    loss_curve: list[float]
    manifest: dict[str, Any] | None


def _collect_experiment(exp_id: str, description: str, root: Path) -> _ExperimentView:
    exp_dir = root / exp_id
    manifest = _load_json(exp_dir / "manifest.json")
    if manifest is None:
        return _ExperimentView(exp_id, description, "not_run", None, None, None, [], None)

    eval_report = _load_json(exp_dir / "evaluation.json")
    metrics = _load_jsonl(exp_dir / "metrics.jsonl")
    loss_curve = [m["loss"] for m in metrics if "loss" in m]

    final_val_loss = manifest.get("final_val_loss")
    overall_ppl = None
    macro_ppl = None
    if eval_report is not None:
        overall_ppl = eval_report.get("overall_perplexity")
        macro_ppl = eval_report.get("macro_avg_perplexity")
    else:
        em = manifest.get("evaluation_metrics", {})
        if isinstance(em, dict):
            overall_ppl = em.get("overall_perplexity")

    status = "run" if (final_val_loss is not None or manifest.get("evaluation_metrics")) else "partial"
    return _ExperimentView(
        exp_id, description, status, final_val_loss, overall_ppl, macro_ppl, loss_curve, manifest
    )


def _experiments_section(root: Path) -> str:
    views = [_collect_experiment(exp_id, desc, root) for exp_id, desc in REGISTRY]

    rows = []
    for v in views:
        status_class = {"run": "ok", "partial": "warn", "not_run": "muted"}[v.status]
        rows.append(
            "<tr>"
            f"<td><code>{v.exp_id}</code></td><td>{_esc(v.description)}</td>"
            f'<td><span class="status {status_class}">{v.status.replace("_", " ")}</span></td>'
            f"<td>{_fmt(v.final_val_loss)}</td><td>{_fmt(v.overall_perplexity, 1)}</td>"
            f"<td>{_fmt(v.macro_avg_perplexity, 1)}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr><th>ID</th><th>Description</th><th>Status</th>"
        "<th>Final val loss</th><th>Overall PPL</th><th>Macro-avg PPL</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    curves = [(v.exp_id, v.loss_curve) for v in views if v.loss_curve]
    curves_html = (
        _line_chart(curves) if curves else "<p class='muted'>No per-step metrics available.</p>"
    )

    n_run = sum(1 for v in views if v.status != "not_run")
    return f"""
    <section>
      <h2>Experiments ({n_run}/{len(views)} run)</h2>
      {table}
      <h3>Training loss curves</h3>
      {curves_html}
    </section>
    """


def _ablation_section(root: Path) -> str:
    groups = {
        "EXP-004 — Tokenizer ablation": ["EXP-004__bpe", "EXP-004__unigram"],
        "EXP-008 — Data filtering ablation": [
            "EXP-008__raw",
            "EXP-008__filtered",
            "EXP-008__filtered_dedup",
        ],
    }
    blocks = []
    for title, variant_ids in groups.items():
        labels, values = [], []
        for vid in variant_ids:
            manifest = _load_json(root / vid / "manifest.json")
            if manifest is None:
                continue
            loss = manifest.get("final_val_loss")
            if loss is None:
                continue
            labels.append(vid.split("__", 1)[-1])
            values.append(float(loss))
        if not values:
            continue
        blocks.append(f"<h3>{_esc(title)}</h3><p class='muted'>Final validation loss (lower is better)</p>{_bar_chart(labels, values)}")
    if not blocks:
        return ""
    return f"<section><h2>Ablations</h2>{''.join(blocks)}</section>"


def _downstream_section(root: Path) -> str:
    """Reads every `<exp_id>/downstream_evaluation.json` under `root`
    (written by `indiclm evaluate-downstream --out-path ...`) and compares
    zero-shot sentiment accuracy across whichever experiments have one —
    directly answering whether mixture/tokenizer/data-quality choices
    move downstream task quality, not just perplexity."""
    rows = []
    chance = None
    overall_accuracies = []
    for path in sorted(root.glob("*/downstream_evaluation.json")):
        report = _load_json(path)
        if report is None:
            continue
        exp_id = path.parent.name
        chance = report.get("chance_accuracy", chance)
        overall_accuracies.append(report.get("overall_accuracy"))
        rows.append(
            f"<tr><td><code>{_esc(exp_id)}</code></td>"
            f"<td>{_fmt(report.get('overall_accuracy'), 4)}</td>"
            f"<td>{_fmt(report.get('macro_avg_accuracy'), 4)}</td>"
            f"<td>{report.get('n_examples')}</td></tr>"
        )
    if not rows:
        return (
            "<section><h2>Downstream evaluation</h2>"
            "<p class='muted'>Run `indiclm evaluate-downstream --checkpoint ... "
            "--out-path experiments/manifests/&lt;exp_id&gt;/downstream_evaluation.json` "
            "to populate this section.</p></section>"
        )
    table = (
        "<table><thead><tr><th>Experiment</th><th>Overall accuracy</th>"
        "<th>Macro-avg accuracy</th><th>N examples</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    all_at_chance = chance is not None and all(a == chance for a in overall_accuracies)
    note = (
        f"<p class='muted'>Zero-shot sentiment classification (see "
        f"<code>indiclm.evaluation.downstream</code> for the scoring method); chance = {_fmt(chance, 4)}. "
        + (
            "Every run here scores at chance — at this project's scale "
            "(tens-to-hundreds of thousands of parameters, tens of thousands of training tokens), "
            "no configuration shows measurable zero-shot task signal; all collapse to predicting "
            "whichever label the tiny tokenizer/vocabulary makes marginally more probable, "
            "regardless of the input. This is an honest negative result about scale, not a "
            "claim that mixture/tokenizer/data-quality choices don't matter — see "
            "docs/reproducibility.md."
            if all_at_chance
            else "Differences between rows reflect real per-configuration variation."
        )
        + "</p>"
    )
    return f"<section><h2>Downstream evaluation</h2>{table}{note}</section>"


def _scaling_section(root: Path) -> str:
    fit = _load_json(root / "EXP-012" / "scaling_law_fit.json")
    if fit is None:
        return "<section><h2>Scaling</h2><p class='muted'>Run `indiclm experiment scaling-sweep` to populate this section.</p></section>"

    status = fit.get("fit_status", "unknown")
    plot_path = root / "EXP-012" / "loss_vs_params.png"
    plot_html = ""
    if plot_path.exists():
        import base64

        b64 = base64.b64encode(plot_path.read_bytes()).decode("ascii")
        plot_html = f'<img class="plot" src="data:image/png;base64,{b64}" alt="Loss vs parameters"/>'

    fit_html = ""
    if status == "ok":
        fit_html = f"""
        <table><tbody>
          <tr><th>alpha (N exponent)</th><td>{_fmt(fit.get('alpha'), 4)} &plusmn; {_fmt(fit.get('alpha_stderr'), 4)}</td></tr>
          <tr><th>beta (D exponent)</th><td>{_fmt(fit.get('beta'), 4)} &plusmn; {_fmt(fit.get('beta_stderr'), 4)}</td></tr>
          <tr><th>L_infinity</th><td>{_fmt(fit.get('L_infinity'), 6)} &plusmn; {_fmt(fit.get('L_infinity_stderr'), 4)}</td></tr>
          <tr><th>R&sup2;</th><td>{_fmt(fit.get('r_squared'), 4)}</td></tr>
          <tr><th>Observations</th><td>{fit.get('n_observations')}</td></tr>
        </tbody></table>
        <p class="muted">With only {fit.get('n_observations')} grid points for a 5-parameter model,
        treat these exponents as an illustration of the fitting methodology, not a trustworthy
        scaling law — see docs/scaling_laws.md and docs/reproducibility.md.</p>
        """
    else:
        fit_html = f"<p class='muted'>Fit status: {_esc(status)}. {_esc(fit.get('note', ''))}</p>"

    obs = fit.get("observations", [])
    obs_rows = "".join(
        f"<tr><td>{_esc(o.get('run_id'))}</td><td>{o.get('n_params_non_embedding')}</td>"
        f"<td>{o.get('d_tokens')}</td><td>{_fmt(o.get('final_val_loss'))}</td>"
        f"<td>{_fmt(o.get('mean_tokens_per_sec'), 0)}</td></tr>"
        for o in obs
    )
    obs_table = (
        "<table><thead><tr><th>Run</th><th>N (non-embed params)</th><th>D (tokens)</th>"
        "<th>Val loss</th><th>Tokens/sec</th></tr></thead>"
        f"<tbody>{obs_rows}</tbody></table>"
        if obs
        else ""
    )

    return f"""
    <section>
      <h2>Scaling</h2>
      {plot_html}
      {fit_html}
      {obs_table}
    </section>
    """


_CSS = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --border: #e5e7eb;
  --card: #f9fafb; --accent: #4C6EF5; --ok: #0CA678; --warn: #F59F00;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #14161a; --fg: #e8e9eb; --muted: #9aa1ac; --border: #2a2d33; --card: #1c1f24; }
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--fg); margin: 0; padding: 2rem 1.5rem 4rem;
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 920px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.15rem; margin-top: 2.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }
h3 { font-size: 0.95rem; color: var(--muted); margin-top: 1.5rem; text-transform: uppercase; letter-spacing: 0.03em; }
.subtitle { color: var(--muted); margin-top: 0; }
.callout {
  background: var(--card); border: 1px solid var(--border); border-radius: 8px;
  padding: 0.9rem 1.1rem; margin-top: 1rem; font-size: 0.9rem; color: var(--muted);
}
.tiles { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 1rem; }
.tile {
  background: var(--card); border: 1px solid var(--border); border-radius: 8px;
  padding: 0.75rem 1rem; min-width: 130px; flex: 1;
}
.tile-value { font-size: 1.3rem; font-weight: 600; }
.tile-label { font-size: 0.78rem; color: var(--muted); margin-top: 0.15rem; }
table { width: 100%; border-collapse: collapse; margin-top: 0.75rem; font-size: 0.87rem; }
th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; }
code { background: var(--card); padding: 0.1rem 0.35rem; border-radius: 4px; font-size: 0.85em; }
.muted { color: var(--muted); font-size: 0.88rem; }
.status { padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
.status.ok { background: color-mix(in srgb, var(--ok) 18%, transparent); color: var(--ok); }
.status.warn { background: color-mix(in srgb, var(--warn) 18%, transparent); color: var(--warn); }
.status.muted { background: var(--card); color: var(--muted); }
.bar-label { font-size: 11px; fill: var(--muted); }
.bar-value { font-size: 11px; fill: var(--fg); }
.axis-label { font-size: 10px; fill: var(--muted); }
.legend { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.35rem; }
.legend-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
.plot { max-width: 100%; border-radius: 6px; border: 1px solid var(--border); margin-top: 0.75rem; }
footer { margin-top: 3rem; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); padding-top: 1rem; }
"""


def generate_dashboard(
    experiments_root: Path = Path("experiments/manifests"),
    data_dir: Path = Path("data"),
    out_path: Path = Path("docs/dashboard.html"),
) -> Path:
    """Reads every artifact currently on disk and writes one static,
    dependency-free HTML dashboard. Safe to call with a partially-run
    project — sections with missing data say so instead of erroring."""
    import time

    dataset_html = _dataset_section(data_dir / "processed")
    tokenizer_html = _tokenizer_section(data_dir)
    experiments_html = _experiments_section(experiments_root)
    ablation_html = _ablation_section(experiments_root)
    downstream_html = _downstream_section(experiments_root)
    scaling_html = _scaling_section(experiments_root)

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>IndicLM Research Dashboard</title>
<style>{_CSS}</style>
</head>
<body>
<main>
  <h1>IndicLM Research Dashboard</h1>
  <p class="subtitle">Generated {generated_at} by <code>indiclm report dashboard</code> from files under
  <code>experiments/manifests/</code> and <code>data/</code>.</p>
  <div class="callout">
    Every number on this page is read directly from a file the pipeline wrote on a real run —
    nothing here is templated or estimated. This project trains on a small, hand-authored
    bootstrap corpus on CPU only; see <code>docs/reproducibility.md</code> for exactly what
    that does and doesn't tell you about model quality at real scale.
  </div>
  {dataset_html}
  {tokenizer_html}
  {experiments_html}
  {ablation_html}
  {downstream_html}
  {scaling_html}
  <footer>IndicLM &middot; static, dependency-free, regenerate anytime with
  <code>indiclm report dashboard</code>.</footer>
</main>
</body>
</html>
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
