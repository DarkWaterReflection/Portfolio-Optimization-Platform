"""Backtester behaviour: reconciliation, costs, turnover, weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization.backtesting.engine import Backtester
from portfolio_optimization.backtesting.strategy import EqualWeightStrategy

LOOKBACK = 120


class SingleAssetStrategy:
    """Always allocate 100% to one named asset (for exact reconciliation)."""

    name = "SingleAsset"

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def weights(self, window_returns: pd.DataFrame) -> pd.Series:
        w = pd.Series(0.0, index=window_returns.columns)
        w[self.ticker] = 1.0
        return w


def test_single_asset_reconciles_to_asset_returns(synthetic_returns):
    """A 100%-single-asset portfolio must reproduce that asset's returns exactly."""
    asset = synthetic_returns.columns[0]
    bt = Backtester()
    res = bt.run(
        synthetic_returns, SingleAssetStrategy(asset),
        frequency="monthly", lookback=LOOKBACK, cost_bps=0.0,
    )
    expected = synthetic_returns[asset].reindex(res.net_returns.index)
    np.testing.assert_allclose(res.net_returns.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_equity_curve_reconciles_with_net_returns(synthetic_returns):
    bt = Backtester()
    res = bt.run(synthetic_returns, EqualWeightStrategy(), lookback=LOOKBACK, cost_bps=5.0)
    rebuilt = (1.0 + res.net_returns).cumprod()
    np.testing.assert_allclose(res.equity_curve.to_numpy(), rebuilt.to_numpy(), rtol=1e-12)


def test_costs_reduce_terminal_wealth(synthetic_returns):
    bt = Backtester()
    free = bt.run(synthetic_returns, EqualWeightStrategy(), lookback=LOOKBACK, cost_bps=0.0)
    costly = bt.run(synthetic_returns, EqualWeightStrategy(), lookback=LOOKBACK, cost_bps=100.0)
    assert costly.equity_curve.iloc[-1] < free.equity_curve.iloc[-1]
    assert costly.total_cost > 0.0
    assert free.total_cost == pytest.approx(0.0)


def test_more_frequent_rebalancing_has_more_turnover(synthetic_returns):
    bt = Backtester()
    ew = EqualWeightStrategy()
    monthly = bt.run(synthetic_returns, ew, frequency="monthly", lookback=LOOKBACK)
    annual = bt.run(synthetic_returns, ew, frequency="annual", lookback=LOOKBACK)
    assert monthly.total_turnover >= annual.total_turnover
    assert monthly.n_rebalances > annual.n_rebalances


def test_target_weights_sum_to_one(synthetic_returns):
    bt = Backtester()
    res = bt.run(synthetic_returns, EqualWeightStrategy(), lookback=LOOKBACK)
    sums = res.weights_history.sum(axis=1)
    np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-9)


def test_empty_panel_raises():
    with pytest.raises(ValueError):
        Backtester().run(pd.DataFrame(), EqualWeightStrategy())


def test_insufficient_history_raises(synthetic_returns):
    with pytest.raises(ValueError):
        Backtester().run(
            synthetic_returns, EqualWeightStrategy(),
            lookback=5000, min_obs=5000,
        )


def test_report_generates_from_result(synthetic_returns):
    bt = Backtester()
    res = bt.run(synthetic_returns, EqualWeightStrategy(), lookback=LOOKBACK)
    report = res.to_report(periods_per_year=252)
    assert report.name == "EqualWeight"
    assert np.isfinite(report.sharpe_ratio)
