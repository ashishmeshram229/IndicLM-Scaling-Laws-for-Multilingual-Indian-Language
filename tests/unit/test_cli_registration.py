"""Regression test for a real bug found while re-running the experiment
registry: `main.py` mounted `experiment_app`, `scaling_app`, and
`ablation_app` with three separate `app.add_typer(..., name="experiment")`
calls. Click's command registry is a flat dict keyed by subcommand name,
so those three calls didn't merge -- each silently overwrote the last,
leaving only `ablation_app`'s two commands reachable and making
`experiment run`, `experiment compare`, `experiment list-experiments`,
and `experiment scaling-sweep` invisible to `--help` and unusable, even
though the underlying functions worked fine when called directly. This
test asserts every expected top-level and `experiment` subcommand stays
registered, so a future refactor of `main.py` can't reintroduce the same
silent-shadowing bug without a test failure.
"""

from __future__ import annotations

from typer.testing import CliRunner

from indiclm.cli.main import app

runner = CliRunner()

EXPECTED_TOP_LEVEL = {"serve", "doctor", "data", "tokenizer", "train", "evaluate", "experiment", "report"}
EXPECTED_EXPERIMENT_SUBCOMMANDS = {
    "run",
    "compare",
    "list-experiments",
    "scaling-sweep",
    "tokenizer-ablation",
    "data-ablation",
}


def test_top_level_commands_registered() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in EXPECTED_TOP_LEVEL:
        assert name in result.output, f"top-level command {name!r} missing from --help"


def test_experiment_subcommands_all_registered() -> None:
    """The specific bug this guards against: three `add_typer(...,
    name="experiment")` calls in main.py silently clobbering each other
    down to only the last one's commands."""
    result = runner.invoke(app, ["experiment", "--help"])
    assert result.exit_code == 0
    for name in EXPECTED_EXPERIMENT_SUBCOMMANDS:
        assert name in result.output, f"`experiment {name}` missing -- CLI mount likely shadowed it again"
