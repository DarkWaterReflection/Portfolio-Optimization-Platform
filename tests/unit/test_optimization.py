"""Optimizer behaviour tests (constraints, feasibility, tagging)."""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_optimization.optimization.base import portfolio_sharpe
from portfolio_optimization.optimization.frontier import efficient_frontier
from portfolio_optimization.optimization.max_sharpe import max_sharpe_portfolio
from portfolio_optimization.optimization.min_variance import min_variance_portfolio
from portfolio_optimization.optimization.random_portfolios import random_portfolios
from portfolio_optimization.optimization.result import Constraints

RF = 0.02


def test_weights_sum_to_one(mu_series, cov_frame):
    res = min_variance_portfolio(mu_series, cov_frame, RF)
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_long_only_weights_nonnegative(mu_series, cov_frame):
    res = max_sharpe_portfolio(mu_series, cov_frame, RF, Constraints(long_only=True))
    assert (res.weights >= -1e-8).all()


def test_weight_cap_respected(mu_series, cov_frame):
    cap = 0.30
    res = max_sharpe_portfolio(
        mu_series, cov_frame, RF, Constraints(long_only=True, w_max=cap)
    )
    assert (res.weights <= cap + 1e-6).all()


def test_min_variance_has_lowest_vol(mu_series, cov_frame):
    mv = min_variance_portfolio(mu_series, cov_frame, RF)
    cloud = random_portfolios(mu_series, cov_frame, RF, n_portfolios=5000, seed=1)
    # No random long-only portfolio should beat the min-variance volatility.
    assert mv.volatility <= cloud["volatility"].min() + 1e-6


def test_max_sharpe_beats_random_cloud(mu_series, cov_frame):
    ms = max_sharpe_portfolio(mu_series, cov_frame, RF)
    cloud = random_portfolios(mu_series, cov_frame, RF, n_portfolios=5000, seed=1)
    assert ms.sharpe >= cloud["sharpe"].max() - 1e-6


def test_infeasible_bounds_raise(mu_series, cov_frame):
    # w_max too small to ever sum to 1 across the universe -> bounds() raises.
    with pytest.raises(ValueError):
        min_variance_portfolio(mu_series, cov_frame, RF, Constraints(w_max=0.1))


def test_frontier_is_monotonic_and_tagged(mu_series, cov_frame):
    front = efficient_frontier(mu_series, cov_frame, RF, n_points=25)
    df = front.to_frame().sort_values("expected_return")
    # On the efficient set, volatility is non-decreasing as return rises.
    assert (df["volatility"].diff().dropna() >= -1e-6).all()
    # Tagged max-Sharpe must dominate every frontier point's Sharpe.
    assert front.max_sharpe.sharpe >= df["sharpe"].max() - 1e-6


def test_frontier_min_point_matches_min_variance(mu_series, cov_frame):
    front = efficient_frontier(mu_series, cov_frame, RF, n_points=25)
    lowest_vol_point = min(front.points, key=lambda p: p.volatility)
    assert lowest_vol_point.volatility >= front.min_variance.volatility - 1e-6


def test_custom_risk_free_changes_sharpe(mu_series, cov_frame):
    w = np.full(len(mu_series), 1.0 / len(mu_series))
    s_low = portfolio_sharpe(w, mu_series.to_numpy(), cov_frame.to_numpy(), 0.0)
    s_high = portfolio_sharpe(w, mu_series.to_numpy(), cov_frame.to_numpy(), 0.10)
    assert s_low > s_high
