r"""Risk and return attribution (portfolio decomposition).

**Risk attribution** uses Euler's theorem for the (homogeneous-degree-1) volatility
function. The marginal contribution to risk of asset :math:`i` is

.. math:: MCR_i = \frac{(\Sigma w)_i}{\sqrt{w'\Sigma w}},

the component contribution is :math:`CTR_i = w_i \cdot MCR_i`, and by Euler the
contributions sum exactly to the portfolio volatility:
:math:`\sum_i CTR_i = \sigma_p`.

**Return attribution** is the trivial linear decomposition
:math:`w_i \mu_i`, summing to :math:`w'\mu`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _align_weights(weights: pd.Series, matrix: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
    """Reorder ``weights`` to match ``matrix`` index; return array + ticker index."""
    w = weights.reindex(matrix.index)
    if w.isna().any():
        missing = w.index[w.isna()].tolist()
        raise ValueError(f"weights missing entries for: {missing}")
    return w.to_numpy(dtype=float), matrix.index


def risk_attribution(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Decompose portfolio volatility into per-asset contributions.

    Args:
        weights: Portfolio weights (ticker-indexed).
        cov: Annualized covariance matrix (ticker-indexed).

    Returns:
        DataFrame indexed by ticker with columns:
        ``weight``, ``mcr`` (marginal contribution to risk),
        ``ctr`` (component contribution, sums to portfolio vol),
        ``pct_contribution`` (CTR as a fraction of total risk).
    """
    w, idx = _align_weights(weights, cov)
    sigma = cov.to_numpy(dtype=float)
    port_var = float(w @ sigma @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))

    if port_vol == 0:
        zeros = np.zeros_like(w)
        return pd.DataFrame(
            {"weight": w, "mcr": zeros, "ctr": zeros, "pct_contribution": zeros},
            index=idx,
        )

    marginal = (sigma @ w) / port_vol          # MCR_i
    contribution = w * marginal                # CTR_i ; sum == port_vol
    pct = contribution / port_vol
    return pd.DataFrame(
        {"weight": w, "mcr": marginal, "ctr": contribution, "pct_contribution": pct},
        index=idx,
    )


def return_attribution(weights: pd.Series, expected_returns: pd.Series) -> pd.DataFrame:
    """Decompose portfolio expected return into per-asset contributions.

    Returns:
        DataFrame indexed by ticker with columns ``weight``,
        ``contribution`` (``w_i * mu_i``), and ``pct_contribution``.
    """
    w = weights.reindex(expected_returns.index)
    if w.isna().any():
        raise ValueError("weights and expected_returns must share tickers")
    contribution = w * expected_returns
    total = float(contribution.sum())
    pct = contribution / total if total != 0 else contribution * 0.0
    return pd.DataFrame(
        {"weight": w, "contribution": contribution, "pct_contribution": pct}
    )


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    r"""Weighted-average asset vol divided by portfolio vol.

    .. math:: DR = \frac{\sum_i w_i \sigma_i}{\sqrt{w'\Sigma w}} \ge 1.

    A value of 1 means no diversification benefit; higher is better.
    """
    w, _ = _align_weights(weights, cov)
    sigma = cov.to_numpy(dtype=float)
    asset_vol = np.sqrt(np.diag(sigma))
    weighted_avg_vol = float(w @ asset_vol)
    port_vol = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    if port_vol == 0:
        return float("nan")
    return weighted_avg_vol / port_vol
