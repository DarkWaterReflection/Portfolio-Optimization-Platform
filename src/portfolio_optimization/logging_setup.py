"""Centralized logging configuration.

Call :func:`configure_logging` once at application/CLI entry. Library modules
should obtain a logger via :func:`get_logger` and never configure handlers
themselves.
"""

from __future__ import annotations

import logging

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install a single stream handler with a consistent format (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger("portfolio_optimization")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger."""
    return logging.getLogger(f"portfolio_optimization.{name}")
