r"""Minimum-variance portfolio.

Solves :math:`\min_w w'\Sigma w` subject to the budget constraint and weight
bounds. When bounds are slack the solution matches the closed form
:math:`w = \frac{\Sigma^{-1}\mathbf 1}{\mathbf 1'\Sigma^{-1}\mathbf 1}`, which the
test suite uses as a correctness oracle.
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
    make_result,
)
from portfolio_optimization.optimization.result import Constraints, OptimizationResult

logger = get_logger("optimization.min_variance")


def min_variance_closed_form(cov: np.ndarray) -> np.ndarray:
    """Unconstrained (budget-only) minimum-variance weights."""
    n = cov.shape[0]
    ones = np.ones(n)
    inv_dot_one = np.linalg.solve(cov, ones)
    return inv_dot_one / (ones @ inv_dot_one)


def min_variance_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.0,
    constraints: Constraints | None = None,
) -> OptimizationResult:
    """Compute the minimum-variance portfolio under ``constraints``."""
    mu_arr, cov_arr, tickers = _as_arrays(mu, cov)
    cons = constraints or Constraints()
    n = len(tickers)

    def objective(w: np.ndarray) -> float:
        return float(w @ cov_arr @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * cov_arr @ w

    scipy_cons = [budget_constraint()] if cons.sum_to_one else []
    result = minimize(
        objective,
        x0=equal_weights(n),
        jac=gradient,
        method="SLSQP",
        bounds=cons.bounds(n),
        constraints=scipy_cons,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        logger.warning("Min-variance solver did not converge: %s", result.message)

    return make_result(
        weights=result.x,
        mu=mu_arr,
        cov=cov_arr,
        tickers=tickers,
        risk_free_rate=risk_free_rate,
        method="min_variance",
        solver_status="ok" if result.success else f"failed: {result.message}",
    )
