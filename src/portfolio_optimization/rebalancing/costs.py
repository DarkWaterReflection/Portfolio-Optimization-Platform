r"""Turnover and transaction-cost modeling.

One-way turnover between an existing (drifted) allocation and a new target is

.. math:: \text{Turnover} = \tfrac{1}{2}\sum_i |w_i^{new} - w_i^{old}|.

When both weight vectors sum to 1 (a self-financing rebalance) this equals the
fraction of the portfolio traded on one side. The transaction cost is a linear
function of turnover at a configurable per-unit rate in basis points.
"""

from __future__ import annotations

import numpy as np


def turnover(w_old: np.ndarray, w_new: np.ndarray) -> float:
    """One-way turnover ``0.5 * sum|w_new - w_old|``.

    For the initial allocation pass ``w_old`` as zeros; the result is then
    ``0.5`` for a fully invested long-only portfolio (documented convention).
    """
    w_old = np.asarray(w_old, dtype=float)
    w_new = np.asarray(w_new, dtype=float)
    return float(0.5 * np.abs(w_new - w_old).sum())


def transaction_cost(turnover_value: float, cost_bps: float) -> float:
    """Linear transaction cost: ``turnover * cost_bps / 10000``."""
    return float(turnover_value * cost_bps / 1e4)
