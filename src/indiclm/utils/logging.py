"""Structured logging setup used across all IndicLM subsystems.

Design note: we standardize on `structlog` so that training, data-pipeline,
and evaluation logs are all machine-readable (JSON) in non-interactive
contexts (CI, cluster jobs) while remaining human-readable in a local
terminal. No component should call `print()`; everything goes through
`get_logger`.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(level: str | None = None, json_logs: bool | None = None) -> None:
    """Configure global logging. Safe to call multiple times (idempotent).

    Args:
        level: DEBUG/INFO/WARNING/ERROR/CRITICAL. Defaults to
            INDICLM_LOG_LEVEL env var, else INFO.
        json_logs: force JSON output. Defaults to True when stdout is not
            a TTY (e.g. CI, cluster logs), False for local interactive use.
    """
    resolved_level = (level or os.environ.get("INDICLM_LOG_LEVEL", "INFO")).upper()
    if json_logs is None:
        json_logs = not sys.stdout.isatty()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, resolved_level, logging.INFO),
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, resolved_level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structured logger bound to `name` (typically __name__)."""
    return structlog.get_logger(name)
