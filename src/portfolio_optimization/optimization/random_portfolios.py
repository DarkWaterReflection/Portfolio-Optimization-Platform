"""Random long-only portfolios for the efficient-frontier scatter cloud.

This is a **visualization / teaching aid**, not an optimizer. Dirichlet sampling
fills the feasible interior so the convex frontier (computed analytically in
:mod:`frontier`) is visibly the upper-left boundary of the cloud. Seeded for
reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimization.optimization.base import (
    _as_arrays,
)


def random_portfolios(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.02,
    n_portfolios: int = 20000,
    seed: int = 42,
    concentration: float = 1.0,
) -> pd.DataFrame:
    """Generate a cloud of random long-only portfolios.

    Args:
        mu: Annualized expected returns.
        cov: Annualized covariance.
        risk_free_rate: Annual risk-free rate for Sharpe.
        n_portfolios: Number of random weight vectors to draw.
        seed: RNG seed for reproducibility.
        concentration: Dirichlet alpha; <1 yields more concentrated (corner)
            portfolios, >1 yields more diversified ones.

    Returns:
        DataFrame with columns ``expected_return``, ``volatility``, ``sharpe``.
    """
    mu_arr, cov_arr, tickers = _as_arrays(mu, cov)
    n = len(tickers)
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.full(n, concentration), size=n_portfolios)

    rets = weights @ mu_arr
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_arr, weights))
    sharpe = np.divide(
        rets - risk_free_rate, vols, out=np.zeros_like(rets), where=vols > 0
    )
    return pd.DataFrame(
        {"expected_return": rets, "volatility": vols, "sharpe": sharpe}
    )
