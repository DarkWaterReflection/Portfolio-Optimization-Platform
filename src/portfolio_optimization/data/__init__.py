"""Data ingestion layer: sources, caching, cleaning, and return computation."""

from __future__ import annotations

from portfolio_optimization.data.cleaning import (
    clean_prices,
    daily_returns,
    monthly_returns,
)
from portfolio_optimization.data.ingestion import DataIngestion

__all__ = ["DataIngestion", "clean_prices", "daily_returns", "monthly_returns"]
