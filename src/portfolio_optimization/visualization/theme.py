"""Shared visual styling for a consistent, professional look."""

from __future__ import annotations

import plotly.graph_objects as go

# A restrained, finance-desk palette.
COLORS = {
    "EqualWeight": "#7f8c8d",
    "MinVariance": "#2980b9",
    "MaxSharpe": "#27ae60",
    "benchmark": "#c0392b",
    "frontier": "#2c3e50",
    "cloud": "#bdc3c7",
    "accent": "#e67e22",
}

SEQUENTIAL = "RdBu"  # diverging scale for correlation heatmaps


def apply_layout(fig: go.Figure, title: str, xaxis: str = "", yaxis: str = "") -> go.Figure:
    """Apply the house style (fonts, margins, grid) to a figure in place."""
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 18}},
        template="plotly_white",
        font={"family": "Inter, Segoe UI, sans-serif", "size": 13},
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def color_for(name: str, fallback: str = "#34495e") -> str:
    """Resolve a strategy/series name to a palette color."""
    return COLORS.get(name, fallback)
