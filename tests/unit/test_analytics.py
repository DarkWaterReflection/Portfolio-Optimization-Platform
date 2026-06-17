"""Analytics tests: metrics vs hand-computed values, Euler attribution, reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization.analytics import metrics as m
from portfolio_optimization.analytics.attribution import (
    diversification_ratio,
    return_attribution,
    risk_attribution,
)
from portfolio_optimization.analytics.report import compare_strategies, performance_report


@pytest.fixture
def simple_returns() -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=2)
    return pd.Series([0.10, -0.20], index=idx)


def test_cumulative_and_total_return(simple_returns):
    cum = m.cumulative_returns(simple_returns)
    np.testing.assert_allclose(cum.to_numpy(), [1.10, 0.88])
    assert m.total_return(simple_returns) == pytest.approx(0.88 - 1.0)


def test_max_drawdown_known(simple_returns):
    # Peak 1.10 -> trough 0.88 => drawdown 0.88/1.10 - 1 = -0.20.
    assert m.max_drawdown(simple_returns) == pytest.approx(-0.20)


def test_annualized_vol_identity():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    expected = r.std(ddof=1) * np.sqrt(252)
    assert m.annualized_volatility(r, 252) == pytest.approx(expected)


def test_sharpe_zero_rf_matches_formula():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.001, 0.01, 300))
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert m.sharpe_ratio(r, risk_free_rate=0.0, periods_per_year=252) == pytest.approx(expected)


def test_sortino_undefined_when_no_downside():
    r = pd.Series([0.01, 0.02, 0.03, 0.015])  # all above MAR=0
    assert np.isnan(m.sortino_ratio(r, risk_free_rate=0.0))


def test_beta_of_series_with_itself_is_one():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0, 0.01, 200))
    assert m.beta(r, r) == pytest.approx(1.0)


def test_alpha_of_benchmark_against_itself_is_zero():
    rng = np.random.default_rng(6)
    b = pd.Series(rng.normal(0.0004, 0.01, 400))
    assert m.jensen_alpha(b, b, risk_free_rate=0.02) == pytest.approx(0.0, abs=1e-12)


def test_information_ratio_zero_active_is_nan():
    rng = np.random.default_rng(7)
    b = pd.Series(rng.normal(0, 0.01, 200))
    assert np.isnan(m.information_ratio(b, b))


def test_metrics_align_on_overlapping_dates():
    idx = pd.bdate_range("2021-01-01", periods=10)
    r = pd.Series(np.linspace(0.01, 0.02, 10), index=idx)
    b = pd.Series(np.linspace(0.005, 0.015, 10), index=idx[2:].union(idx[:2]))
    # Should not raise despite reordered/partial index.
    assert np.isfinite(m.beta(r, b))


# --- Attribution ---------------------------------------------------------------

def test_risk_attribution_sums_to_portfolio_vol(cov_frame, tickers):
    w = pd.Series(np.full(len(tickers), 1.0 / len(tickers)), index=tickers)
    table = risk_attribution(w, cov_frame)
    port_vol = float(np.sqrt(w.to_numpy() @ cov_frame.to_numpy() @ w.to_numpy()))
    # Euler: component contributions sum exactly to portfolio volatility.
    assert table["ctr"].sum() == pytest.approx(port_vol, rel=1e-10)
    assert table["pct_contribution"].sum() == pytest.approx(1.0, rel=1e-10)


def test_return_attribution_sums_to_portfolio_return(mu_series, tickers):
    w = pd.Series(np.full(len(tickers), 1.0 / len(tickers)), index=tickers)
    table = return_attribution(w, mu_series)
    assert table["contribution"].sum() == pytest.approx(float(w @ mu_series), rel=1e-12)
    assert table["pct_contribution"].sum() == pytest.approx(1.0, rel=1e-12)


def test_diversification_ratio_at_least_one(cov_frame, tickers):
    w = pd.Series(np.full(len(tickers), 1.0 / len(tickers)), index=tickers)
    assert diversification_ratio(w, cov_frame) >= 1.0 - 1e-9


def test_risk_attribution_rejects_missing_tickers(cov_frame):
    w = pd.Series([1.0], index=["NOT_IN_UNIVERSE"])
    with pytest.raises(ValueError):
        risk_attribution(w, cov_frame)


# --- Reporting -----------------------------------------------------------------

def test_performance_report_populates_relative_metrics():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2020-01-01", periods=500)
    bench = pd.Series(rng.normal(0.0003, 0.01, 500), index=idx)
    port = bench * 1.2 + 0.0001  # levered-ish tilt on the benchmark
    rep = performance_report(port, "TestPort", benchmark=bench)
    assert rep.beta is not None and rep.information_ratio is not None
    assert rep.name == "TestPort"


def test_compare_strategies_ranks(monkeypatch):
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2020-01-01", periods=400)
    good = pd.Series(rng.normal(0.0008, 0.008, 400), index=idx)
    bad = pd.Series(rng.normal(0.0001, 0.02, 400), index=idx)
    reps = [performance_report(good, "Good"), performance_report(bad, "Bad")]
    table = compare_strategies(reps, rank_by="sharpe_ratio")
    # Highest Sharpe ranks first (top row) and gets rank 1.
    assert table.index[0] == "Good"
    assert table.loc["Good", "rank_risk_adjusted"] == 1
