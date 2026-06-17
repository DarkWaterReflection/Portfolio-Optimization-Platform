"""The credibility test: the backtester must never feed future data to a strategy.

A spy strategy records the latest date in every window it is handed. For each
executed rebalance, that date must be strictly earlier than the rebalance date
itself — i.e. decisions use only information available beforehand.
"""

from __future__ import annotations

import pandas as pd

from portfolio_optimization.backtesting.engine import Backtester
from portfolio_optimization.optimization.base import equal_weights


class SpyStrategy:
    """Equal-weight strategy that logs the last date of each estimation window."""

    name = "Spy"

    def __init__(self) -> None:
        self.window_end_dates: list[pd.Timestamp] = []

    def weights(self, window_returns: pd.DataFrame) -> pd.Series:
        self.window_end_dates.append(window_returns.index.max())
        cols = window_returns.columns
        return pd.Series(equal_weights(len(cols)), index=cols)


def test_estimation_window_strictly_precedes_rebalance(synthetic_returns):
    spy = SpyStrategy()
    res = Backtester().run(
        synthetic_returns, spy, frequency="monthly", lookback=120, cost_bps=0.0
    )

    # Calls and executed rebalances are 1:1 and in order.
    assert len(spy.window_end_dates) == len(res.rebalance_dates)
    assert len(res.rebalance_dates) > 0

    for window_end, rebal_date in zip(
        spy.window_end_dates, res.rebalance_dates, strict=True
    ):
        assert window_end < rebal_date, (
            f"Lookahead! window ended {window_end} on/after rebalance {rebal_date}"
        )


def test_window_respects_lookback_length(synthetic_returns):
    """Each window holds at most `lookback` observations, all before the decision."""
    seen: list[int] = []

    class LenSpy:
        name = "LenSpy"

        def weights(self, window_returns: pd.DataFrame) -> pd.Series:
            seen.append(len(window_returns))
            cols = window_returns.columns
            return pd.Series(equal_weights(len(cols)), index=cols)

    Backtester().run(synthetic_returns, LenSpy(), lookback=120, min_obs=120)
    assert max(seen) <= 120
    assert min(seen) >= 120  # min_obs enforced
