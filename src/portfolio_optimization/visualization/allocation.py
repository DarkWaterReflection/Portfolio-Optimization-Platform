"""Portfolio allocation charts."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from portfolio_optimization.visualization.theme import apply_layout, color_for


def allocation_bar(weights: pd.Series, top_n: int = 15, name: str = "") -> go.Figure:
    """Horizontal bar chart of the largest weights.

    Args:
        weights: Portfolio weights indexed by ticker.
        top_n: Show only the ``top_n`` largest holdings (by absolute weight).
        name: Optional strategy name (drives the bar color and title).
    """
    ranked = weights.reindex(weights.abs().sort_values(ascending=False).index)
    ranked = ranked.head(top_n).iloc[::-1]  # largest at top after horizontal plot

    fig = go.Figure(
        go.Bar(
            x=ranked.to_numpy(),
            y=list(ranked.index),
            orientation="h",
            marker={"color": color_for(name)},
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    title = f"Allocation — {name}" if name else "Portfolio Allocation"
    apply_layout(fig, title, "Weight", "")
    fig.update_xaxes(tickformat=".0%")
    return fig
