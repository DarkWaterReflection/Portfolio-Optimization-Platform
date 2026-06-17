"""Rebalance schedule and cost/turnover tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization.rebalancing.costs import transaction_cost, turnover
from portfolio_optimization.rebalancing.schedule import rebalance_dates


@pytest.fixture
def three_years() -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-01", "2023-12-31")


def test_monthly_count(three_years):
    dates = rebalance_dates(three_years, "monthly")
    assert len(dates) == 36  # 3 years x 12 months


def test_quarterly_count(three_years):
    assert len(rebalance_dates(three_years, "quarterly")) == 12


def test_semiannual_count(three_years):
    assert len(rebalance_dates(three_years, "semi-annual")) == 6


def test_annual_count(three_years):
    assert len(rebalance_dates(three_years, "annual")) == 3


def test_none_is_single_buy_and_hold(three_years):
    dates = rebalance_dates(three_years, "none")
    assert dates == [three_years[0]]


def test_first_rebalance_is_first_trading_day(three_years):
    dates = rebalance_dates(three_years, "monthly")
    assert dates[0] == three_years[0]


def test_dates_are_trading_days(three_years):
    dates = rebalance_dates(three_years, "quarterly")
    assert all(d in three_years for d in dates)


def test_unknown_frequency_raises(three_years):
    with pytest.raises(ValueError):
        rebalance_dates(three_years, "fortnightly")


def test_empty_index_returns_empty():
    assert rebalance_dates(pd.DatetimeIndex([]), "monthly") == []


def test_turnover_formula():
    old = np.array([0.5, 0.5])
    new = np.array([0.6, 0.4])
    assert turnover(old, new) == pytest.approx(0.1)  # 0.5 * (0.1 + 0.1)


def test_turnover_from_cash_is_half():
    assert turnover(np.zeros(4), np.array([0.25, 0.25, 0.25, 0.25])) == pytest.approx(0.5)


def test_transaction_cost_scaling():
    assert transaction_cost(0.1, cost_bps=10.0) == pytest.approx(0.1 * 10 / 1e4)
    assert transaction_cost(1.0, cost_bps=0.0) == 0.0
