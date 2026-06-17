"""Hardening tests: edge cases and degenerate inputs are handled, not silently
corrupted. These guard the robustness fixes added in the docs/hardening phase."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization.analytics.attribution import risk_attribution
from portfolio_optimization.analytics.metrics import (
    annualized_return,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from portfolio_optimization.estimation.risk import (
    correlation_from_covariance,
    covariance_matrix,
)

# --- Estimation guards --------------------------------------------------------

def test_covariance_rejects_nonfinite(synthetic_returns):
    """inf survives dropna() and would poison np.cov / the LW sums — reject it."""
    poisoned = synthetic_returns.copy()
    poisoned.iloc[10, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        covariance_matrix(poisoned)


def test_covariance_too_few_observations(synthetic_returns):
    with pytest.raises(ValueError, match="at least 2"):
        covariance_matrix(synthetic_returns.iloc[:1])


def test_covariance_unknown_method(synthetic_returns):
    with pytest.raises(ValueError, match="Unknown risk method"):
        covariance_matrix(synthetic_returns, method="bogus")


def test_correlation_handles_zero_variance(cov_frame):
    """A zero-variance asset must not produce nan/inf in the correlation matrix."""
    cov = cov_frame.copy()
    cov.iloc[0, :] = 0.0
    cov.iloc[:, 0] = 0.0
    corr = correlation_from_covariance(cov)
    assert np.isfinite(corr.to_numpy()).all()
    # Self-correlation stays 1; the degenerate row collapses to 0 off-diagonal.
    assert corr.iloc[0, 0] == pytest.approx(1.0)
    assert corr.iloc[0, 1] == pytest.approx(0.0)


def test_correlation_diagonal_is_one(cov_frame):
    corr = correlation_from_covariance(cov_frame)
    np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0, atol=1e-9)


# --- Metric degeneracy --------------------------------------------------------

def test_sharpe_on_short_series_is_nan():
    assert np.isnan(sharpe_ratio(pd.Series([0.01])))


def test_sharpe_on_constant_series_is_nan():
    # Zero dispersion -> undefined Sharpe, must not divide-by-zero.
    assert np.isnan(sharpe_ratio(pd.Series([0.001] * 50)))


def test_sortino_with_no_downside_is_nan():
    # All returns above the MAR -> zero downside deviation -> nan, not inf.
    assert np.isnan(sortino_ratio(pd.Series([0.01] * 50)))


def test_annualized_return_total_wipeout():
    # A -100% period wipes the curve; report -100%, not nan/complex.
    r = pd.Series([0.01, -1.0, 0.02])
    assert annualized_return(r) == -1.0


def test_max_drawdown_monotone_up_is_zero():
    r = pd.Series([0.01] * 20)
    assert max_drawdown(r) == pytest.approx(0.0)


def test_information_ratio_identical_benchmark_is_nan():
    s = pd.Series(np.linspace(0.0, 0.02, 30))
    assert np.isnan(information_ratio(s, s.copy()))


# --- Attribution invariant ----------------------------------------------------

def test_risk_attribution_sums_to_portfolio_vol(mu_series, cov_frame):
    """Euler identity: component contributions sum exactly to portfolio vol."""
    w = pd.Series(1.0 / len(mu_series), index=mu_series.index)
    attr = risk_attribution(w, cov_frame)
    port_vol = float(np.sqrt(w.to_numpy() @ cov_frame.to_numpy() @ w.to_numpy()))
    assert attr["ctr"].sum() == pytest.approx(port_vol, rel=1e-10)
    assert attr["pct_contribution"].sum() == pytest.approx(1.0, rel=1e-10)


def test_risk_attribution_zero_vol_is_safe(mu_series):
    """Zero-volatility portfolio returns zeros, not a divide-by-zero."""
    n = len(mu_series)
    zero_cov = pd.DataFrame(np.zeros((n, n)), index=mu_series.index, columns=mu_series.index)
    w = pd.Series(1.0 / n, index=mu_series.index)
    attr = risk_attribution(w, zero_cov)
    assert (attr["ctr"] == 0.0).all()
