"""Rebalance-date generation aligned to an actual trading calendar.

Given the DatetimeIndex of available trading days, a frequency selects the
**first trading day of each period** as a rebalance date. Anchoring to real
trading days (rather than calendar month-ends, which may be weekends/holidays)
keeps the backtester's decision dates executable.
"""

from __future__ import annotations

import pandas as pd

REBALANCE_FREQUENCIES = (
    "monthly",
    "quarterly",
    "semi-annual",
    "annual",
    "none",
)


def _period_keys(index: pd.DatetimeIndex, frequency: str) -> list:
    """Map each date to a period label used to detect period boundaries."""
    freq = frequency.lower()
    if freq == "monthly":
        return list(index.to_period("M"))
    if freq == "quarterly":
        return list(index.to_period("Q"))
    if freq == "annual":
        return list(index.to_period("Y"))
    if freq == "semi-annual":
        return [(d.year, 1 if d.month <= 6 else 2) for d in index]
    raise ValueError(
        f"Unknown frequency {frequency!r}; choose from {REBALANCE_FREQUENCIES}."
    )


def rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> list[pd.Timestamp]:
    """Return the rebalance dates for ``frequency`` over a trading calendar.

    Args:
        index: Sorted DatetimeIndex of available trading days.
        frequency: One of :data:`REBALANCE_FREQUENCIES`. ``"none"`` yields a
            single date (the first day) — i.e. allocate once and buy-and-hold.

    Returns:
        The first trading day within each period, in chronological order.
    """
    if len(index) == 0:
        return []
    index = pd.DatetimeIndex(index).sort_values()

    if frequency.lower() == "none":
        return [index[0]]

    keys = _period_keys(index, frequency)
    seen: set = set()
    dates: list[pd.Timestamp] = []
    for date, key in zip(index, keys, strict=True):
        if key not in seen:
            seen.add(key)
            dates.append(date)
    return dates
