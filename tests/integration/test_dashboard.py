"""Integration test for the static HTML dashboard generator.

Runs against the repo's actual `experiments/manifests/` and `data/`
artifacts (the same ones `docs/dashboard.html` is generated from) rather
than a synthetic fixture, since the whole point of this generator is to
faithfully reflect what's really on disk.
"""

from pathlib import Path

from indiclm.experiments.dashboard import generate_dashboard

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_generate_dashboard_writes_valid_html(tmp_path: Path) -> None:
    out_path = tmp_path / "dashboard.html"
    result = generate_dashboard(
        experiments_root=REPO_ROOT / "experiments" / "manifests",
        data_dir=REPO_ROOT / "data",
        out_path=out_path,
    )
    assert result == out_path
    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "<title>IndicLM Research Dashboard</title>" in html
    assert "<h2>Dataset</h2>" in html
    assert "<h2>Tokenizer</h2>" in html
    assert "Experiments (" in html


def test_generate_dashboard_handles_missing_artifacts_gracefully(tmp_path: Path) -> None:
    """An empty project (nothing run yet) must not crash the generator —
    each section should degrade to an explicit 'no data' message."""
    empty_root = tmp_path / "empty_manifests"
    empty_data = tmp_path / "empty_data"
    out_path = tmp_path / "dashboard.html"

    result = generate_dashboard(experiments_root=empty_root, data_dir=empty_data, out_path=out_path)

    html = result.read_text(encoding="utf-8")
    assert "run `indiclm data prepare`" in html
    assert "run `indiclm tokenizer train`" in html
    assert "Experiments (0/12 run)" in html
