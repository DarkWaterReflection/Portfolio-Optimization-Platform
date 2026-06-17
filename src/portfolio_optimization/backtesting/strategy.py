"""Strategies: map a trailing returns window to target weights.

A :class:`Strategy` receives only the data the backtester deems available at the
decision point (a window of past returns) and returns a target weight vector.
Strategies must never look beyond the window they are handed — that discipline is
what makes the walk-forward simulation honest, and it is enforced by the
no-lookahead test.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from portfolio_optimization.estimation.returns import expected_returns
from portfolio_optimization.estimation.risk import covariance_matrix
from portfolio_optimization.optimization.base import equal_weights
from portfolio_optimization.optimization.max_sharpe import max_sharpe_portfolio
from portfolio_optimization.optimization.min_variance import min_variance_portfolio
from portfolio_optimization.optimization.result import Constraints


@runtime_checkable
class Strategy(Protocol):
    """Maps a window of past returns to target portfolio weights."""

    name: str

    def weights(self, window_returns: pd.DataFrame) -> pd.Series:
        """Target weights (ticker-indexed) given only ``window_returns``."""
        ...


class EqualWeightStrategy:
    """The 1/N portfolio — the benchmark optimizers must justify beating."""

    name = "EqualWeight"

    def weights(self, window_returns: pd.DataFrame) -> pd.Series:
        cols = window_returns.columns
        return pd.Series(equal_weights(len(cols)), index=cols, name="weight")


class MeanVarianceStrategy:
    """Re-estimate mu/Sigma each rebalance and solve a chosen MPT objective.

    Args:
        objective: ``"max_sharpe"`` or ``"min_variance"``.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Annualization factor for estimation.
        constraints: Weight constraints applied at every rebalance.
        risk_model: Covariance estimator (``"ledoit_wolf"`` or ``"sample"``).
    """

    def __init__(
        self,
        objective: str = "max_sharpe",
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        constraints: Constraints | None = None,
        risk_model: str = "ledoit_wolf",
    ) -> None:
        if objective not in {"max_sharpe", "min_variance"}:
            raise ValueError(f"Unknown objective {objective!r}")
        self.objective = objective
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.constraints = constraints or Constraints()
        self.risk_model = risk_model
        self.name = "MaxSharpe" if objective == "max_sharpe" else "MinVariance"

    def weights(self, window_returns: pd.DataFrame) -> pd.Series:
        mu = expected_returns(window_returns, periods_per_year=self.periods_per_year)
        cov = covariance_matrix(
            window_returns, method=self.risk_model, periods_per_year=self.periods_per_year
        )
        if self.objective == "max_sharpe":
            result = max_sharpe_portfolio(
                mu, cov, self.risk_free_rate, self.constraints
            )
        else:
            result = min_variance_portfolio(
                mu, cov, self.risk_free_rate, self.constraints
            )
        return result.weights
