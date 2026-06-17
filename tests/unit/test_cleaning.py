"""Data-quality and cleaning tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_optimization.data.cleaning import clean_prices, daily_returns, monthly_returns


def test_clean_prices_sorts_and_dedups(synthetic_prices):
    shuffled = synthetic_prices.sample(frac=1.0, random_state=1)
    dup = pd.concat([shuffled, shuffled.iloc[[0]]])
    cleaned = clean_prices(dup)
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.has_duplicates


def test_clean_prices_drops_sparse_ticker(synthetic_prices):
    prices = synthetic_prices.copy()
    prices.iloc[:, 0] = np.nan  # entirely missing -> must be dropped
    cleaned = clean_prices(prices, max_missing_pct=0.05)
    assert synthetic_prices.columns[0] not in cleaned.columns
    assert cleaned.shape[1] == synthetic_prices.shape[1] - 1


def test_clean_prices_fills_short_gaps(synthetic_prices):
    prices = synthetic_prices.copy()
    prices.iloc[10, 1] = np.nan  # single-day gap, within ffill limit
    cleaned = clean_prices(prices, max_ffill_days=3)
    assert not cleaned.isna().to_numpy().any()


def test_clean_prices_rejects_empty():
    with pytest.raises(ValueError):
        clean_prices(pd.DataFrame())


def test_daily_returns_match_manual(synthetic_prices):
    rets = daily_returns(synthetic_prices)
    manual = synthetic_prices.iloc[1, 0] / synthetic_prices.iloc[0, 0] - 1
    assert rets.iloc[0, 0] == pytest.approx(manual)


def test_log_returns_sum_to_total(synthetic_prices):
    col = synthetic_prices.columns[0]
    log_rets = daily_returns(synthetic_prices, log=True)[col]
    total = np.log(synthetic_prices[col].iloc[-1] / synthetic_prices[col].iloc[0])
    assert log_rets.sum() == pytest.approx(total, rel=1e-9)


def test_monthly_returns_have_fewer_rows(synthetic_prices):
    monthly = monthly_returns(synthetic_prices)
    daily = daily_returns(synthetic_prices)
    assert len(monthly) < len(daily)
