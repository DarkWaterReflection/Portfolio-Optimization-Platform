"""Boundary dataclasses for the optimization layer.

These are the stable contracts that downstream modules (analytics, backtesting,
dashboard) depend on. Keep them small, immutable, and free of behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Constraints:
    """Portfolio weight constraints shared by every optimizer.

    Attributes:
        long_only: If True, weights are bounded below by ``max(w_min, 0)``.
        w_min: Lower bound on each individual weight.
        w_max: Upper bound on each individual weight.
        sum_to_one: Enforce the budget constraint ``sum(w) == 1``.
        target_return: Optional equality constraint on portfolio expected
            return (used when sweeping the efficient frontier).
    """

    long_only: bool = True
    w_min: float = 0.0
    w_max: float = 1.0
    sum_to_one: bool = True
    target_return: float | None = None

    def bounds(self, n_assets: int) -> list[tuple[float, float]]:
        """Per-asset ``(lower, upper)`` bounds for ``scipy.optimize``."""
        lower = max(self.w_min, 0.0) if self.long_only else self.w_min
        if lower > self.w_max:
            raise ValueError(f"Infeasible bounds: lower {lower} > upper {self.w_max}")
        if self.w_max * n_assets < 1.0 and self.sum_to_one:
            raise ValueError(
                f"Infeasible: w_max={self.w_max} across {n_assets} assets cannot sum to 1."
            )
        return [(lower, self.w_max)] * n_assets


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of a single portfolio optimization.

    ``weights`` is indexed by ticker. Scalar metrics are annualized and use the
    same ``periods_per_year`` / risk-free convention as the inputs that produced
    them.
    """

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    solver_status: str = "ok"
    method: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.weights, pd.Series):
            raise TypeError("weights must be a pandas Series indexed by ticker")

    @property
    def tickers(self) -> list[str]:
        return list(self.weights.index)

    def top_holdings(self, n: int = 10) -> pd.Series:
        """Largest ``n`` weights, descending."""
        return self.weights.sort_values(ascending=False).head(n)

    def as_dict(self) -> dict:
        return {
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "solver_status": self.solver_status,
            "method": self.method,
            "weights": self.weights.to_dict(),
        }


@dataclass(frozen=True)
class FrontierPoint:
    """A single point on the efficient frontier."""

    expected_return: float
    volatility: float
    sharpe: float
    weights: np.ndarray


@dataclass(frozen=True)
class EfficientFrontierResult:
    """The full efficient frontier plus the two canonical tagged portfolios."""

    points: list[FrontierPoint]
    min_variance: OptimizationResult
    max_sharpe: OptimizationResult
    tickers: list[str]

    def to_frame(self) -> pd.DataFrame:
        """Frontier as a tidy DataFrame (one row per point)."""
        return pd.DataFrame(
            {
                "expected_return": [p.expected_return for p in self.points],
                "volatility": [p.volatility for p in self.points],
                "sharpe": [p.sharpe for p in self.points],
            }
        )
