"""Correlation and covariance heatmaps."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from portfolio_optimization.estimation.risk import correlation_from_covariance
from portfolio_optimization.visualization.theme import apply_layout


def _heatmap(matrix: pd.DataFrame, title: str, zmid: float | None, colorscale: str) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=list(matrix.columns),
            y=list(matrix.index),
            zmid=zmid,
            colorscale=colorscale,
            hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
        )
    )
    apply_layout(fig, title)
    fig.update_yaxes(autorange="reversed")
    return fig


def correlation_heatmap(cov: pd.DataFrame) -> go.Figure:
    """Correlation matrix derived from a covariance matrix, on a diverging scale."""
    corr = correlation_from_covariance(cov)
    return _heatmap(corr, "Correlation Matrix", zmid=0.0, colorscale="RdBu")


def covariance_heatmap(cov: pd.DataFrame) -> go.Figure:
    """Annualized covariance matrix heatmap."""
    return _heatmap(cov, "Covariance Matrix (annualized)", zmid=None, colorscale="Viridis")
