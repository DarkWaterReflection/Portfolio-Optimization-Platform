"""Pluggable market-data sources behind a common ``DataSource`` interface."""

from __future__ import annotations

from portfolio_optimization.data.sources.base import DataSource, get_source, register_source

__all__ = ["DataSource", "get_source", "register_source"]
