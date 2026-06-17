r"""Core portfolio mathematics and shared optimizer utilities.

Pure, stateless functions for the MPT moments plus helpers used by every
optimizer. All inputs are expected **annualized** (mu, Sigma) so the returned
metrics are annualized too.

Moments:
    mu_p   = w' mu
    sig_p2 = w' Sigma w
    Sharpe = (mu_p - rf) / sqrt(sig_p2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimization.optimization.result import Constraints, OptimizationResult


def _as_arrays(mu: pd.Series, cov: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Validate and align mu/Sigma; return numpy views and the ticker order."""
    if list(mu.index) != list(cov.index) or list(cov.index) != list(cov.columns):
        # Reindex covariance to mu's order to guarantee alignment.
        cov = cov.reindex(index=mu.index, columns=mu.index)
    if cov.isna().to_numpy().any():
        raise ValueError("Covariance has NaNs after alignment; check inputs share tickers.")
    return mu.to_numpy(dtype=float), cov.to_numpy(dtype=float), list(mu.index)


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """Annualized portfolio expected return ``w' mu``."""
    return float(weights @ mu)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """Annualized portfolio volatility ``sqrt(w' Sigma w)``."""
    var = float(weights @ cov @ weights)
    return float(np.sqrt(max(var, 0.0)))


def portfolio_sharpe(
    weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free_rate: float
) -> float:
    """Annualized Sharpe ratio of a weight vector."""
    vol = portfolio_volatility(weights, cov)
    if vol == 0.0:
        return 0.0
    return (portfolio_return(weights, mu) - risk_free_rate) / vol


def make_result(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    tickers: list[str],
    risk_free_rate: float,
    method: str,
    solver_status: str = "ok",
    meta: dict | None = None,
) -> OptimizationResult:
    """Assemble an :class:`OptimizationResult` from a solved weight vector."""
    w = np.asarray(weights, dtype=float)
    return OptimizationResult(
        weights=pd.Series(w, index=tickers, name="weight"),
        expected_return=portfolio_return(w, mu),
        volatility=portfolio_volatility(w, cov),
        sharpe=portfolio_sharpe(w, mu, cov, risk_free_rate),
        solver_status=solver_status,
        method=method,
        meta=meta or {},
    )


def equal_weights(n: int) -> np.ndarray:
    """The 1/N portfolio — the benchmark every optimizer must justify beating."""
    return np.full(n, 1.0 / n)


def budget_constraint() -> dict:
    """SciPy equality constraint enforcing ``sum(w) == 1``."""
    return {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}


def target_return_constraint(mu: np.ndarray, target: float) -> dict:
    """SciPy equality constraint enforcing ``w' mu == target``."""
    return {"type": "eq", "fun": lambda w, mu=mu, t=target: float(w @ mu - t)}


def default_constraints(settings_long_only: bool, w_min: float, w_max: float) -> Constraints:
    """Convenience builder mirroring the config defaults."""
    return Constraints(long_only=settings_long_only, w_min=w_min, w_max=w_max)
