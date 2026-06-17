"""Visualization smoke tests: figures build with the expected traces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from portfolio_optimization.optimization.frontier import efficient_frontier
from portfolio_optimization.optimization.random_portfolios import random_portfolios
from portfolio_optimization.visualization.allocation import allocation_bar
from portfolio_optimization.visualization.frontier_plot import efficient_frontier_figure
from portfolio_optimization.visualization.heatmaps import correlation_heatmap, covariance_heatmap
from portfolio_optimization.visualization.performance import (
    drawdown_figure,
    equity_curve_figure,
    rolling_sharpe_figure,
)


@pytest.fixture
def frontier(mu_series, cov_frame):
    return efficient_frontier(mu_series, cov_frame, 0.02, n_points=20)


@pytest.fixture
def return_curves() -> dict[str, pd.Series]:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=300)
    return {
        "MaxSharpe": pd.Series(rng.normal(0.0006, 0.01, 300), index=idx),
        "EqualWeight": pd.Series(rng.normal(0.0004, 0.012, 300), index=idx),
    }


def test_frontier_figure_has_all_layers(frontier, mu_series, cov_frame):
    cloud = random_portfolios(mu_series, cov_frame, 0.02, n_portfolios=500, seed=2)
    fig = efficient_frontier_figure(frontier, cloud, risk_free_rate=0.02)
    assert isinstance(fig, go.Figure)
    names = {t.name for t in fig.data}
    assert "Efficient frontier" in names
    assert "Random portfolios" in names
    assert "Capital Market Line" in names
    assert any("Max Sharpe" in (n or "") for n in names)


def test_frontier_figure_without_cloud(frontier):
    fig = efficient_frontier_figure(frontier, None, show_cml=False)
    names = {t.name for t in fig.data}
    assert "Random portfolios" not in names
    assert "Efficient frontier" in names


def test_correlation_heatmap(cov_frame):
    fig = correlation_heatmap(cov_frame)
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Heatmap)
    # Correlation diagonal is 1.
    z = np.array(fig.data[0].z)
    np.testing.assert_allclose(np.diag(z), 1.0, atol=1e-9)


def test_covariance_heatmap(cov_frame):
    fig = covariance_heatmap(cov_frame)
    assert isinstance(fig.data[0], go.Heatmap)


def test_allocation_bar_limits_top_n(mu_series, cov_frame):
    from portfolio_optimization.optimization.min_variance import min_variance_portfolio

    w = min_variance_portfolio(mu_series, cov_frame, 0.02).weights
    fig = allocation_bar(w, top_n=3, name="MinVariance")
    assert len(fig.data[0].y) == 3


def test_equity_curve_from_returns(return_curves):
    fig = equity_curve_figure(return_curves)
    assert len(fig.data) == 2
    # Curves should start near 1 (compounded from returns).
    assert fig.data[0].y[0] == pytest.approx(1 + return_curves[fig.data[0].name].iloc[0], abs=1e-6)


def test_drawdown_figure_nonpositive(return_curves):
    fig = drawdown_figure(return_curves)
    assert (np.array(fig.data[0].y) <= 1e-9).all()


def test_rolling_sharpe_figure(return_curves):
    fig = rolling_sharpe_figure(return_curves, window=60)
    assert len(fig.data) == 2
