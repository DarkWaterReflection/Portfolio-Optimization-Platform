"""Backtest performance visualizations: equity, drawdown, rolling Sharpe."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from portfolio_optimization.analytics.metrics import drawdown_series, rolling_sharpe
from portfolio_optimization.visualization.theme import apply_layout, color_for


def equity_curve_figure(curves: dict[str, pd.Series], log_y: bool = False) -> go.Figure:
    """Overlay growth-of-1 equity curves for several strategies.

    Args:
        curves: Mapping of strategy name -> equity curve (or net return series;
            return series are detected and compounded).
        log_y: Plot the y-axis on a log scale (useful over long horizons).
    """
    fig = go.Figure()
    for name, series in curves.items():
        curve = series if _looks_like_equity(series) else (1.0 + series).cumprod()
        fig.add_trace(
            go.Scatter(
                x=curve.index, y=curve.to_numpy(), mode="lines",
                name=name, line={"color": color_for(name), "width": 2},
            )
        )
    apply_layout(fig, "Portfolio Growth (Growth of 1)", "Date", "Cumulative value")
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def drawdown_figure(returns: dict[str, pd.Series]) -> go.Figure:
    """Drawdown curves (filled) for several strategies."""
    fig = go.Figure()
    for name, series in returns.items():
        dd = drawdown_series(series)
        fig.add_trace(
            go.Scatter(
                x=dd.index, y=dd.to_numpy(), mode="lines", name=name,
                line={"color": color_for(name), "width": 1.5}, fill="tozeroy",
            )
        )
    apply_layout(fig, "Drawdown", "Date", "Drawdown")
    fig.update_yaxes(tickformat=".0%")
    return fig


def rolling_sharpe_figure(
    returns: dict[str, pd.Series],
    window: int = 126,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> go.Figure:
    """Rolling annualized Sharpe over a trailing window for several strategies."""
    fig = go.Figure()
    for name, series in returns.items():
        rs = rolling_sharpe(series, window, risk_free_rate, periods_per_year)
        fig.add_trace(
            go.Scatter(
                x=rs.index, y=rs.to_numpy(), mode="lines", name=name,
                line={"color": color_for(name), "width": 1.5},
            )
        )
    fig.add_hline(y=0.0, line={"color": "#95a5a6", "width": 1, "dash": "dot"})
    apply_layout(fig, f"Rolling Sharpe ({window}-day)", "Date", "Annualized Sharpe")
    return fig


def _looks_like_equity(series: pd.Series) -> bool:
    """Heuristic: an equity curve is strictly positive and far from 0-centered."""
    s = series.dropna()
    return bool(len(s) > 0 and (s > 0).all() and s.iloc[0] > 0.5)
