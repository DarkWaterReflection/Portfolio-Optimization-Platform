"""Efficient frontier visualization with the random-portfolio cloud and CML."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from portfolio_optimization.optimization.result import EfficientFrontierResult
from portfolio_optimization.visualization.theme import COLORS, apply_layout


def efficient_frontier_figure(
    frontier: EfficientFrontierResult,
    random_cloud: pd.DataFrame | None = None,
    risk_free_rate: float = 0.02,
    show_cml: bool = True,
) -> go.Figure:
    """Build the classic risk-return diagram.

    Layers: the random-portfolio cloud (colored by Sharpe), the efficient
    frontier line, the minimum-variance and maximum-Sharpe markers, and the
    capital market line tangent at the max-Sharpe portfolio.

    Args:
        frontier: Result from :func:`efficient_frontier`.
        random_cloud: Optional DataFrame with ``volatility``/``expected_return``/
            ``sharpe`` columns from :func:`random_portfolios`.
        risk_free_rate: Annual risk-free rate (CML intercept).
        show_cml: Whether to draw the capital market line.
    """
    fig = go.Figure()

    if random_cloud is not None and not random_cloud.empty:
        fig.add_trace(
            go.Scattergl(
                x=random_cloud["volatility"],
                y=random_cloud["expected_return"],
                mode="markers",
                marker={
                    "size": 4,
                    "color": random_cloud["sharpe"],
                    "colorscale": "Viridis",
                    "opacity": 0.45,
                    "colorbar": {"title": "Sharpe"},
                },
                name="Random portfolios",
                hovertemplate="vol=%{x:.2%}<br>ret=%{y:.2%}<extra></extra>",
            )
        )

    df = frontier.to_frame().sort_values("volatility")
    fig.add_trace(
        go.Scatter(
            x=df["volatility"], y=df["expected_return"],
            mode="lines", line={"color": COLORS["frontier"], "width": 3},
            name="Efficient frontier",
        )
    )

    mv, ms = frontier.min_variance, frontier.max_sharpe
    fig.add_trace(
        go.Scatter(
            x=[mv.volatility], y=[mv.expected_return], mode="markers",
            marker={"symbol": "diamond", "size": 14, "color": COLORS["MinVariance"]},
            name=f"Min Variance (S={mv.sharpe:.2f})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ms.volatility], y=[ms.expected_return], mode="markers",
            marker={"symbol": "star", "size": 18, "color": COLORS["MaxSharpe"]},
            name=f"Max Sharpe (S={ms.sharpe:.2f})",
        )
    )

    if show_cml and ms.volatility > 0:
        x_max = float(df["volatility"].max()) * 1.05
        slope = (ms.expected_return - risk_free_rate) / ms.volatility
        xs = np.array([0.0, x_max])
        fig.add_trace(
            go.Scatter(
                x=xs, y=risk_free_rate + slope * xs, mode="lines",
                line={"color": COLORS["accent"], "width": 1.5, "dash": "dash"},
                name="Capital Market Line",
            )
        )

    apply_layout(fig, "Efficient Frontier", "Annualized Volatility", "Annualized Return")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig
