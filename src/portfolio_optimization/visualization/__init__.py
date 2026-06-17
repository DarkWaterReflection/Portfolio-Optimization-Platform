"""Visualization layer: Plotly figure builders (pure; no rendering side effects).

Every function takes data and returns a ``plotly.graph_objects.Figure``, so the
same builders serve the Streamlit dashboard, notebooks, and the test suite.
"""

from __future__ import annotations

from portfolio_optimization.visualization.allocation import allocation_bar
from portfolio_optimization.visualization.frontier_plot import efficient_frontier_figure
from portfolio_optimization.visualization.heatmaps import correlation_heatmap, covariance_heatmap
from portfolio_optimization.visualization.performance import (
    drawdown_figure,
    equity_curve_figure,
    rolling_sharpe_figure,
)

__all__ = [
    "allocation_bar",
    "correlation_heatmap",
    "covariance_heatmap",
    "drawdown_figure",
    "efficient_frontier_figure",
    "equity_curve_figure",
    "rolling_sharpe_figure",
]
