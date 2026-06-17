"""Yahoo Finance data source (via the ``yfinance`` package).

No API key required. Adjusted-close prices already incorporate splits and
dividends, which is exactly what MPT return/risk estimation needs.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from portfolio_optimization.data.sources.base import register_source
from portfolio_optimization.logging_setup import get_logger

logger = get_logger("data.yfinance")


class YFinanceSource:
    """Fetch historical prices from Yahoo Finance."""

    name = "yfinance"

    def fetch_prices(
        self,
        tickers: list[str],
        start: date,
        end: date,
        field: str = "adj_close",
    ) -> pd.DataFrame:
        import yfinance as yf

        if not tickers:
            raise ValueError("tickers must be a non-empty list")

        logger.info("Fetching %d tickers from Yahoo Finance (%s..%s)", len(tickers), start, end)
        raw = yf.download(
            tickers=tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,        # 'Close' becomes split/dividend-adjusted
            progress=False,
            group_by="column",
            threads=True,
        )

        prices = self._extract_field(raw, field, tickers)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        prices.columns.name = "ticker"
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            logger.warning("No data returned for: %s", ", ".join(missing))
        return prices

    @staticmethod
    def _extract_field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
        """Normalize yfinance's (single vs multi-ticker) shapes to a wide panel.

        With ``auto_adjust=True`` the adjusted price lives in the 'Close' column.
        """
        col = "Close" if field in {"adj_close", "close"} else field.capitalize()

        if isinstance(raw.columns, pd.MultiIndex):
            # Columns are (field, ticker). Select the price field across tickers.
            if col not in raw.columns.get_level_values(0):
                raise KeyError(f"Field {col!r} not present in downloaded data.")
            wide = raw[col].copy()
        else:
            # Single ticker: flat columns. Build a one-column frame.
            if col not in raw.columns:
                raise KeyError(f"Field {col!r} not present in downloaded data.")
            wide = raw[[col]].copy()
            wide.columns = [tickers[0]]

        return wide


register_source(YFinanceSource())
