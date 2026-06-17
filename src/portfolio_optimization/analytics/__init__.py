"""Analytics layer: performance & risk metrics, attribution, and reporting.

All metrics operate on **periodic return series** (e.g. daily) and take a
``periods_per_year`` annualization factor plus an annual ``risk_free_rate``.
Conventions are documented per-function and validated against hand-computed
fixtures in the test suite.
"""

from __future__ import annotations

from portfolio_optimization.analytics.attribution import (
    return_attribution,
    risk_attribution,
)
from portfolio_optimization.analytics.metrics import (
    annualized_return,
    annualized_volatility,
    beta,
    cumulative_returns,
    drawdown_series,
    information_ratio,
    jensen_alpha,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)
from portfolio_optimization.analytics.report import (
    PerformanceReport,
    compare_strategies,
    performance_report,
)

__all__ = [
    "PerformanceReport",
    "annualized_return",
    "annualized_volatility",
    "beta",
    "compare_strategies",
    "cumulative_returns",
    "drawdown_series",
    "information_ratio",
    "jensen_alpha",
    "max_drawdown",
    "performance_report",
    "return_attribution",
    "risk_attribution",
    "sharpe_ratio",
    "sortino_ratio",
    "tracking_error",
]
