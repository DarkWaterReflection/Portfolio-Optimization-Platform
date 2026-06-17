"""Ingestion orchestration: source + cache + cleaning behind one entry point."""

from __future__ import annotations

from datetime import date

import pandas as pd

from portfolio_optimization.config import Settings, load_settings
from portfolio_optimization.data.cache import PriceCache
from portfolio_optimization.data.cleaning import clean_prices, daily_returns, monthly_returns
from portfolio_optimization.data.sources.base import get_source
from portfolio_optimization.logging_setup import get_logger

logger = get_logger("data.ingestion")


class DataIngestion:
    """Fetch, cache, and clean price panels; derive returns.

    Honors ``settings.data.offline``: in offline mode a cache miss raises rather
    than hitting the network, which keeps CI and demos deterministic.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.cache = PriceCache(self.settings.cache_path)

    def get_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
        field: str = "adj_close",
        clean: bool = True,
    ) -> pd.DataFrame:
        """Return a (optionally cleaned) wide price panel, using the cache."""
        source_name = self.settings.data.source
        cached = self.cache.get(source_name, tickers, start, end, field)

        if cached is not None:
            prices = cached
        elif self.settings.data.offline:
            raise RuntimeError(
                "Offline mode: no cached data for this request. "
                "Run once online to populate the cache, or disable offline mode."
            )
        else:
            source = get_source(source_name)
            prices = source.fetch_prices(tickers, start, end, field)
            self.cache.put(source_name, tickers, start, end, field, prices)

        if clean:
            prices = clean_prices(
                prices,
                max_ffill_days=self.settings.data.max_ffill_days,
                max_missing_pct=self.settings.data.max_missing_pct,
            )
        return prices

    def get_returns(
        self,
        tickers: list[str],
        start: date,
        end: date,
        frequency: str = "daily",
        log: bool = False,
    ) -> pd.DataFrame:
        """Return daily or monthly returns for ``tickers``."""
        prices = self.get_prices(tickers, start, end, clean=True)
        if frequency == "daily":
            return daily_returns(prices, log=log)
        if frequency == "monthly":
            return monthly_returns(prices, log=log)
        raise ValueError(f"Unsupported frequency {frequency!r}; use 'daily' or 'monthly'.")
