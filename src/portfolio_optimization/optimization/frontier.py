r"""Efficient frontier construction.

The frontier is traced by solving, for a grid of target returns
:math:`\mu^\*\in[\mu_{minvar}, \max_i\mu_i]`, the quadratic program

.. math:: \min_w w'\Sigma w \;\; \text{s.t.}\;\; \mathbf 1'w = 1,\; w'\mu=\mu^\*,\;
    w_{min}\le w_i\le w_{max}.

This is the *correct* way to build the frontier — random sampling is only a
visualization aid (see :mod:`random_portfolios`) and will not find the true
boundary in high dimensions. The minimum-variance and maximum-Sharpe portfolios
are computed separately and tagged onto the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_optimization.logging_setup import get_logger
from portfolio_optimization.optimization.base import (
    _as_arrays,
    budget_constraint,
    equal_weights,
    portfolio_return,
    portfolio_sharpe,
    portfolio_volatility,
    target_return_constraint,
)
from portfolio_optimization.optimization.max_sharpe import max_sharpe_portfolio
from portfolio_optimization.optimization.min_variance import min_variance_portfolio
from portfolio_optimization.optimization.result import (
    Constraints,
    EfficientFrontierResult,
    FrontierPoint,
)

logger = get_logger("optimization.frontier")


def _min_variance_for_target(
    mu_arr: np.ndarray,
    cov_arr: np.ndarray,
    target: float,
    cons: Constraints,
) -> np.ndarray | None:
    """Solve the target-return QP; return weights or ``None`` if infeasible."""
    n = len(mu_arr)
    scipy_cons = [target_return_constraint(mu_arr, target)]
    if cons.sum_to_one:
        scipy_cons.append(budget_constraint())

    res = minimize(
        lambda w: float(w @ cov_arr @ w),
        x0=equal_weights(n),
        jac=lambda w: 2.0 * cov_arr @ w,
        method="SLSQP",
        bounds=cons.bounds(n),
        constraints=scipy_cons,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not res.success:
        return None
    # Confirm the target was actually met (SLSQP can report success off-target).
    if abs(portfolio_return(res.x, mu_arr) - target) > 1e-6:
        return None
    return res.x


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.02,
    constraints: Constraints | None = None,
    n_points: int = 50,
) -> EfficientFrontierResult:
    """Construct the efficient frontier and tag the canonical portfolios.

    Args:
        mu: Annualized expected returns (ticker-indexed).
        cov: Annualized covariance (ticker-indexed, aligned with ``mu``).
        risk_free_rate: Annual risk-free rate for Sharpe.
        constraints: Weight constraints applied at every target.
        n_points: Number of target-return samples along the frontier.

    Returns:
        :class:`EfficientFrontierResult` with the frontier points plus the
        minimum-variance and maximum-Sharpe portfolios.
    """
    mu_arr, cov_arr, tickers = _as_arrays(mu, cov)
    cons = constraints or Constraints()

    min_var = min_variance_portfolio(mu, cov, risk_free_rate, cons)
    max_shp = max_sharpe_portfolio(mu, cov, risk_free_rate, cons)

    # Frontier spans from the min-variance return up to the max achievable return.
    lo = min_var.expected_return
    hi = float(np.max(mu_arr))
    if hi <= lo:
        hi = lo + 1e-6
    targets = np.linspace(lo, hi, n_points)

    points: list[FrontierPoint] = []
    for target in targets:
        w = _min_variance_for_target(mu_arr, cov_arr, float(target), cons)
        if w is None:
            continue
        points.append(
            FrontierPoint(
                expected_return=portfolio_return(w, mu_arr),
                volatility=portfolio_volatility(w, cov_arr),
                sharpe=portfolio_sharpe(w, mu_arr, cov_arr, risk_free_rate),
                weights=w,
            )
        )

    if not points:
        logger.warning("No feasible frontier points; check constraints vs. target range.")

    return EfficientFrontierResult(
        points=points,
        min_variance=min_var,
        max_sharpe=max_shp,
        tickers=tickers,
    )
