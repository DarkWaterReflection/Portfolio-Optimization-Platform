"""Price cleaning and return computation.

The cleaning policy is deliberately explicit because every downstream number
(covariance, Sharpe, drawdown) inherits its assumptions:

* Use adjusted close only (splits/dividends handled upstream).
* Forward-fill short gaps up to ``max_ffill_days``; drop tickers with more than
  ``max_missing_pct`` missing observations.
* Daily simple returns ``P_t / P_{t-1} - 1``; monthly via month-end resample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimization.logging_setup import get_logger

logger = get_logger("data.cleaning")


def clean_prices(
    prices: pd.DataFrame,
    max_ffill_days: int = 3,
    max_missing_pct: float = 0.05,
) -> pd.DataFrame:
    """Align, gap-fill, and prune a wide price panel.

    Args:
        prices: Wide DataFrame (DatetimeIndex × tickers) of adjusted closes.
        max_ffill_days: Maximum consecutive missing days to forward-fill.
        max_missing_pct: Drop a ticker if its missing fraction exceeds this.

    Returns:
        A cleaned price panel with a sorted DatetimeIndex and no remaining NaNs.

    Raises:
        ValueError: If no tickers survive the missing-data filter.
    """
    if prices.empty:
        raise ValueError("Received an empty price panel.")

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]

    # Drop tickers that are too sparse to trust.
    missing_frac = prices.isna().mean()
    too_sparse = missing_frac[missing_frac > max_missing_pct].index.tolist()
    if too_sparse:
        logger.warning("Dropping %d sparse tickers: %s", len(too_sparse), ", ".join(too_sparse))
        prices = prices.drop(columns=too_sparse)

    if prices.shape[1] == 0:
        raise ValueError("All tickers dropped by the missing-data filter; widen the window.")

    # Forward-fill only short gaps, then drop any rows still incomplete.
    prices = prices.ffill(limit=max_ffill_days)
    before = len(prices)
    prices = prices.dropna(how="any")
    dropped_rows = before - len(prices)
    if dropped_rows:
        logger.info("Dropped %d rows with residual gaps after limited ffill.", dropped_rows)

    if (prices <= 0).to_numpy().any():
        raise ValueError("Non-positive prices encountered after cleaning.")

    return prices


def daily_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """Daily returns from a cleaned price panel.

    Args:
        prices: Cleaned wide price panel.
        log: If True, return log returns ``ln(P_t / P_{t-1})``; otherwise simple.
    """
    rets = np.log(prices / prices.shift(1)) if log else prices.pct_change()
    return rets.dropna(how="all")


def monthly_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """Month-end returns from a cleaned price panel."""
    month_end = prices.resample("ME").last()
    rets = np.log(month_end / month_end.shift(1)) if log else month_end.pct_change()
    return rets.dropna(how="all")
