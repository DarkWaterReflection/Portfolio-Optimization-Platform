"""Shared pytest fixtures: deterministic synthetic market data (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

N_ASSETS = 5
N_DAYS = 1000
SEED = 7


@pytest.fixture(scope="session")
def tickers() -> list[str]:
    return [f"A{i}" for i in range(N_ASSETS)]


@pytest.fixture(scope="session")
def true_cov() -> np.ndarray:
    """A known positive-definite daily covariance matrix."""
    rng = np.random.default_rng(SEED)
    # Random factor structure -> guaranteed PD when we add a diagonal.
    loadings = rng.normal(size=(N_ASSETS, 2)) * 0.01
    cov = loadings @ loadings.T + np.diag(rng.uniform(0.5e-4, 1.5e-4, N_ASSETS))
    return cov


@pytest.fixture(scope="session")
def true_mu() -> np.ndarray:
    """Known daily mean returns (distinct so frontier has a real spread)."""
    return np.array([0.0002, 0.0004, 0.0006, 0.0003, 0.0005])


@pytest.fixture(scope="session")
def synthetic_returns(tickers, true_mu, true_cov) -> pd.DataFrame:
    """Multivariate-normal daily returns from the known mu/cov."""
    rng = np.random.default_rng(SEED + 1)
    sample = rng.multivariate_normal(true_mu, true_cov, size=N_DAYS)
    idx = pd.bdate_range("2018-01-01", periods=N_DAYS)
    return pd.DataFrame(sample, index=idx, columns=tickers)


@pytest.fixture(scope="session")
def synthetic_prices(synthetic_returns) -> pd.DataFrame:
    """Price panel implied by the synthetic returns (start at 100)."""
    return 100.0 * (1.0 + synthetic_returns).cumprod()


@pytest.fixture
def mu_series(tickers, true_mu) -> pd.Series:
    """Annualized expected returns as a labeled Series (252-day convention)."""
    return pd.Series(true_mu * 252, index=tickers, name="expected_return")


@pytest.fixture
def cov_frame(tickers, true_cov) -> pd.DataFrame:
    """Annualized covariance as a labeled DataFrame."""
    return pd.DataFrame(true_cov * 252, index=tickers, columns=tickers)
