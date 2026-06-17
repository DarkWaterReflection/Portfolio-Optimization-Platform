"""Boundary dataclass for backtest output."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from portfolio_optimization.analytics.report import PerformanceReport, performance_report


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a single strategy backtest.

    Attributes:
        name: Strategy label.
        net_returns: Daily portfolio returns **net** of transaction costs.
        gross_returns: Daily portfolio returns **before** costs.
        equity_curve: Compounded growth of 1 unit on ``net_returns``.
        weights_history: Target weights at each executed rebalance
            (rebalance date x asset).
        turnover: One-way turnover at each executed rebalance.
        costs: Per-day cost drag (zero except on rebalance days).
        rebalance_dates: Dates on which weights were actually reset.
    """

    name: str
    net_returns: pd.Series
    gross_returns: pd.Series
    equity_curve: pd.Series
    weights_history: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        """Total return lost to transaction costs over the backtest."""
        return float(self.costs.sum())

    @property
    def total_turnover(self) -> float:
        """Sum of one-way turnover across all rebalances."""
        return float(self.turnover.sum())

    @property
    def n_rebalances(self) -> int:
        return len(self.rebalance_dates)

    def to_report(
        self,
        benchmark: pd.Series | None = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> PerformanceReport:
        """Summarize net performance as a :class:`PerformanceReport`."""
        return performance_report(
            self.net_returns,
            name=self.name,
            benchmark=benchmark,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
