"""Performance reporting and multi-strategy comparison.

``performance_report`` bundles the full metric set for one return series into an
immutable :class:`PerformanceReport`. ``compare_strategies`` stacks several into a
ranked comparison table — the deliverable for the benchmarking section.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from portfolio_optimization.analytics import metrics as m


@dataclass(frozen=True)
class PerformanceReport:
    """A complete performance summary for a single strategy/portfolio."""

    name: str
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    beta: float | None = None
    alpha: float | None = None
    information_ratio: float | None = None
    tracking_error: float | None = None

    def as_row(self) -> dict:
        return asdict(self)


def performance_report(
    returns: pd.Series,
    name: str,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> PerformanceReport:
    """Compute every headline metric for one periodic return series.

    Benchmark-relative metrics (beta, alpha, information ratio, tracking error)
    are populated only when a ``benchmark`` series is supplied.
    """
    rel: dict[str, float | None] = {
        "beta": None,
        "alpha": None,
        "information_ratio": None,
        "tracking_error": None,
    }
    if benchmark is not None:
        rel = {
            "beta": m.beta(returns, benchmark),
            "alpha": m.jensen_alpha(returns, benchmark, risk_free_rate, periods_per_year),
            "information_ratio": m.information_ratio(returns, benchmark, periods_per_year),
            "tracking_error": m.tracking_error(returns, benchmark, periods_per_year),
        }

    return PerformanceReport(
        name=name,
        total_return=m.total_return(returns),
        annualized_return=m.annualized_return(returns, periods_per_year),
        annualized_volatility=m.annualized_volatility(returns, periods_per_year),
        sharpe_ratio=m.sharpe_ratio(returns, risk_free_rate, periods_per_year),
        sortino_ratio=m.sortino_ratio(returns, risk_free_rate, periods_per_year),
        max_drawdown=m.max_drawdown(returns),
        calmar_ratio=m.calmar_ratio(returns, periods_per_year),
        **rel,
    )


def compare_strategies(
    reports: list[PerformanceReport],
    rank_by: str = "sharpe_ratio",
) -> pd.DataFrame:
    """Stack reports into a comparison table with raw and risk-adjusted ranks.

    Args:
        reports: One :class:`PerformanceReport` per strategy.
        rank_by: Metric used for the primary (risk-adjusted) ranking.

    Returns:
        DataFrame indexed by strategy name, sorted by ``rank_by`` descending,
        with added ``rank_return`` and ``rank_risk_adjusted`` integer columns.
    """
    if not reports:
        raise ValueError("No reports to compare.")
    df = pd.DataFrame([r.as_row() for r in reports]).set_index("name")
    df["rank_return"] = df["annualized_return"].rank(ascending=False).astype(int)
    df["rank_risk_adjusted"] = df[rank_by].rank(ascending=False).astype(int)
    return df.sort_values(rank_by, ascending=False)
