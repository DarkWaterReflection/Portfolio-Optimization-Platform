"""Validation: numerical optimizers must match closed-form solutions.

These are the credibility tests. With slack bounds (no active inequality
constraints) the SLSQP solutions must reproduce the analytic minimum-variance and
tangency portfolios to tight tolerance.
"""

from __future__ import annotations

import numpy as np

from portfolio_optimization.optimization.max_sharpe import (
    max_sharpe_closed_form,
    max_sharpe_portfolio,
)
from portfolio_optimization.optimization.min_variance import (
    min_variance_closed_form,
    min_variance_portfolio,
)
from portfolio_optimization.optimization.result import Constraints

RF = 0.02
# Unbounded (allow shorting, wide caps) so the closed form is the true optimum.
UNBOUNDED = Constraints(long_only=False, w_min=-10.0, w_max=10.0)


def test_min_variance_matches_closed_form(mu_series, cov_frame):
    numerical = min_variance_portfolio(mu_series, cov_frame, RF, UNBOUNDED)
    analytic = min_variance_closed_form(cov_frame.to_numpy())
    np.testing.assert_allclose(numerical.weights.to_numpy(), analytic, atol=1e-6)


def test_max_sharpe_matches_closed_form(mu_series, cov_frame):
    numerical = max_sharpe_portfolio(mu_series, cov_frame, RF, UNBOUNDED)
    analytic = max_sharpe_closed_form(mu_series.to_numpy(), cov_frame.to_numpy(), RF)
    # Compare Sharpe (scale/normalization-robust) and the weight vector directly.
    np.testing.assert_allclose(numerical.weights.to_numpy(), analytic, atol=1e-5)


def test_closed_form_min_variance_sums_to_one(cov_frame):
    w = min_variance_closed_form(cov_frame.to_numpy())
    assert w.sum() == np.float64(w.sum())
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)


def test_tangency_weights_sum_to_one(mu_series, cov_frame):
    w = max_sharpe_closed_form(mu_series.to_numpy(), cov_frame.to_numpy(), RF)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)
