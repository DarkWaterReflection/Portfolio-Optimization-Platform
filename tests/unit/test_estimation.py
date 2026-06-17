"""Return and risk estimation tests."""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_optimization.estimation.returns import expected_returns
from portfolio_optimization.estimation.risk import (
    _ledoit_wolf_identity,
    correlation_from_covariance,
    covariance_matrix,
)


def test_expected_returns_annualize(synthetic_returns):
    """Annualized mu is exactly the per-period sample mean times the factor.

    (We test the annualization identity deterministically; recovering the *true*
    drift from 1000 noisy samples is statistically infeasible -- the standard
    error of an annualized daily mean here is ~0.08.)
    """
    mu = expected_returns(synthetic_returns, periods_per_year=252)
    expected = synthetic_returns.mean() * 252
    np.testing.assert_allclose(mu.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_expected_returns_recovers_ranking(synthetic_returns, true_mu):
    """Even with noise, the cross-sectional ordering of drifts is preserved here."""
    mu = expected_returns(synthetic_returns, periods_per_year=252)
    # Highest-drift asset (index 2) should rank top; lowest (index 0) near bottom.
    assert mu.to_numpy().argmax() == int(np.argmax(true_mu))


def test_expected_returns_rejects_empty():
    import pandas as pd

    with pytest.raises(ValueError):
        expected_returns(pd.DataFrame())


def test_covariance_is_symmetric_psd(synthetic_returns):
    cov = covariance_matrix(synthetic_returns, method="ledoit_wolf")
    arr = cov.to_numpy()
    np.testing.assert_allclose(arr, arr.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(arr)
    assert (eigvals > 0).all()


def test_covariance_annualization(synthetic_returns):
    daily = covariance_matrix(synthetic_returns, method="sample", periods_per_year=1)
    annual = covariance_matrix(synthetic_returns, method="sample", periods_per_year=252)
    np.testing.assert_allclose(annual.to_numpy(), daily.to_numpy() * 252, rtol=1e-10)


def test_ledoit_wolf_intensity_in_unit_interval(synthetic_returns):
    x = synthetic_returns.to_numpy()
    # Reconstruct intensity from the shrunk result vs. sample to confirm [0, 1] blend.
    shrunk = _ledoit_wolf_identity(x)
    assert np.isfinite(shrunk).all()
    # Shrunk matrix must remain symmetric.
    np.testing.assert_allclose(shrunk, shrunk.T, atol=1e-12)


def test_ledoit_wolf_diag_between_sample_and_target(synthetic_returns):
    """Shrunk variances lie between the sample variances and the common target."""
    x = synthetic_returns.to_numpy()
    xc = x - x.mean(0)
    s = xc.T @ xc / len(x)
    shrunk = _ledoit_wolf_identity(x)
    mu_target = np.trace(s) / s.shape[0]
    lo = np.minimum(np.diag(s), mu_target)
    hi = np.maximum(np.diag(s), mu_target)
    assert (np.diag(shrunk) >= lo - 1e-12).all()
    assert (np.diag(shrunk) <= hi + 1e-12).all()


def test_correlation_has_unit_diagonal(cov_frame):
    corr = correlation_from_covariance(cov_frame)
    np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0, atol=1e-12)
    assert (corr.to_numpy() <= 1.0 + 1e-9).all()
