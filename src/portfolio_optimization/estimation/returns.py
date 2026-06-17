"""Expected (annualized) return estimation.

``mean_historical`` is the standard baseline: the sample mean of periodic returns
scaled to an annual figure. The estimator is intentionally simple and explicit;
more robust priors (CAPM-implied, exponentially weighted, Black–Litterman) are
future extensions that plug in behind the same signature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_returns(
    returns: pd.DataFrame,
    method: str = "mean_historical",
    periods_per_year: int = 252,
    log_input: bool = False,
    halflife: float | None = None,
) -> pd.Series:
    """Annualized expected returns, one per asset.

    Args:
        returns: Periodic (e.g. daily) returns, dates x tickers.
        method: ``mean_historical`` (arithmetic mean) or ``ewm`` (exponentially
            weighted mean, more responsive to recent regimes).
        periods_per_year: Annualization factor (252 trading days by default).
        log_input: If the input is log returns, annualize by multiplication and
            convert back to a simple annual return via ``exp(.)-1``.
        halflife: Half-life in periods for the ``ewm`` method.

    Returns:
        Series of annualized expected returns indexed by ticker.
    """
    if returns.empty:
        raise ValueError("returns is empty")

    if method == "mean_historical":
        per_period = returns.mean()
    elif method == "ewm":
        hl = halflife or len(returns) / 4.0
        per_period = returns.ewm(halflife=hl).mean().iloc[-1]
    else:
        raise ValueError(f"Unknown return method {method!r}")

    if log_input:
        # Annualize log returns additively, then express as a simple return.
        annual_log = per_period * periods_per_year
        return np.expm1(annual_log).rename("expected_return")

    return (per_period * periods_per_year).rename("expected_return")
