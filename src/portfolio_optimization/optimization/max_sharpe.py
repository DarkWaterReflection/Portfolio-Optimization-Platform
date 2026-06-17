r"""Maximum-Sharpe (tangency) portfolio.

The Sharpe ratio is non-convex in :math:`w` directly, so we minimize the negative
Sharpe with SLSQP from multiple starting points (equal-weight plus seeded random
simplex draws) and keep the best feasible optimum. For the unconstrained,
budget-only case there is a closed form

.. math:: w_{tan} \propto \Sigma^{-1}(\mu - r_f \mathbf 1)

which the test suite uses as a validation oracle.
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
    portfolio_sharpe,
)
from portfolio_optimization.optimization.result import Constraints, OptimizationResult

logger = get_logger("optimization.max_sharpe")


def max_sharpe_closed_form(
    mu: np.ndarray, cov: np.ndarray, risk_free_rate: float
) -> np.ndarray:
    """Unconstrained tangency weights (may be negative; renormalized to sum 1)."""
    excess = mu - risk_free_rate
    raw = np.linalg.solve(cov, excess)
    return raw / np.sum(raw)


def max_sharpe_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.02,
    constraints: Constraints | None = None,
    n_restarts: int = 8,
    seed: int = 42,
) -> OptimizationResult:
    """Compute the maximum-Sharpe portfolio under ``constraints``.

    Uses multi-start SLSQP to mitigate the non-convexity of the Sharpe objective.
    """
    mu_arr, cov_arr, tickers = _as_arrays(mu, cov)
    cons = constraints or Constraints()
    n = len(tickers)
    bounds = cons.bounds(n)
    scipy_cons = [budget_constraint()] if cons.sum_to_one else []

    def neg_sharpe(w: np.ndarray) -> float:
        return -portfolio_sharpe(w, mu_arr, cov_arr, risk_free_rate)

    rng = np.random.default_rng(seed)
    starts = [equal_weights(n)]
    for _ in range(max(0, n_restarts - 1)):
        starts.append(rng.dirichlet(np.ones(n)))

    best_w: np.ndarray | None = None
    best_obj = np.inf
    any_success = False
    for x0 in starts:
        res = minimize(
            neg_sharpe,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=scipy_cons,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if res.success:
            any_success = True
        if res.fun < best_obj:
            best_obj = res.fun
            best_w = res.x

    if best_w is None:  # pragma: no cover - solver pathology
        raise RuntimeError("Max-Sharpe optimization failed from all starting points.")

    if not any_success:
        logger.warning("Max-Sharpe: no start fully converged; returning best-effort optimum.")

    return make_result(
        weights=best_w,
        mu=mu_arr,
        cov=cov_arr,
        tickers=tickers,
        risk_free_rate=risk_free_rate,
        method="max_sharpe",
        solver_status="ok" if any_success else "best_effort",
        meta={"n_restarts": n_restarts},
    )
