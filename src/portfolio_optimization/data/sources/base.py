"""``DataSource`` interface and a small name-based registry.

New providers (Alpha Vantage, Polygon, ...) implement :class:`DataSource` and
register themselves with :func:`register_source`, so the rest of the platform
selects a source purely by name from config.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataSource(Protocol):
    """A source of historical price data.

    Implementations return a wide DataFrame indexed by trading date with one
    column per ticker, containing the requested ``field`` (default adjusted
    close). Missing data is left as NaN for the cleaning layer to handle.
    """

    name: str

    def fetch_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
        field: str = "adj_close",
    ) -> pd.DataFrame:
        """Fetch a wide price panel for ``tickers`` over ``[start, end]``."""
        ...


_REGISTRY: dict[str, DataSource] = {}


def register_source(source: DataSource) -> None:
    """Register a source instance under its ``.name``."""
    _REGISTRY[source.name] = source


def get_source(name: str) -> DataSource:
    """Look up a registered source by name."""
    if name not in _REGISTRY:
        # Import side-effect: ensure built-ins are registered before we give up.
        from portfolio_optimization.data.sources import yfinance_source  # noqa: F401

    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown data source {name!r}. Registered: {available}")
    return _REGISTRY[name]
