r"""Covariance (risk) estimation.

Sample covariance on many names with limited history is badly conditioned and,
once inverted by an optimizer, produces error-maximizing weights. The default is
therefore **Ledoit-Wolf linear shrinkage** toward a scaled-identity target
(Ledoit & Wolf, 2004, *A Well-Conditioned Estimator for Large-Dimensional
Covariance Matrices*), which has a closed-form optimal shrinkage intensity.

All covariances are returned **annualized** (multiplied by ``periods_per_year``)
and PSD-repaired so downstream Cholesky / inversion is safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimization.logging_setup import get_logger

logger = get_logger("estimation.risk")


def covariance_matrix(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Annualized covariance matrix.

    Args:
        returns: Periodic returns, dates x tickers (NaNs dropped row-wise).
        method: ``sample`` or ``ledoit_wolf``.
        periods_per_year: Annualization factor.

    Returns:
        Annualized covariance as a labeled, symmetric, PSD DataFrame.
    """
    clean = returns.dropna(how="any")
    if clean.shape[0] < 2:
        raise ValueError("Need at least 2 observations to estimate covariance.")

    tickers = list(clean.columns)
    x = clean.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        # inf survives dropna() and would silently poison np.cov / the LW sums.
        raise ValueError("Returns contain non-finite values (inf); clean inputs first.")

    if method == "sample":
        cov = np.cov(x, rowvar=False, ddof=1)
    elif method == "ledoit_wolf":
        cov = _ledoit_wolf_identity(x)
    else:
        raise ValueError(f"Unknown risk method {method!r}")

    cov = _nearest_psd(np.atleast_2d(cov))
    cov_annual = cov * periods_per_year
    return pd.DataFrame(cov_annual, index=tickers, columns=tickers)


def _ledoit_wolf_identity(x: np.ndarray) -> np.ndarray:
    r"""Ledoit-Wolf shrinkage toward ``mu * I``.

    Returns the (per-period) shrunk covariance. The optimal intensity is

    .. math:: \delta^\* = b^2 / d^2,\quad
        \Sigma^\* = \delta^\* \mu I + (1-\delta^\*) S

    where :math:`\mu = \mathrm{tr}(S)/N`, :math:`d^2 = \lVert S-\mu I\rVert_F^2`,
    and :math:`b^2 = \min(\bar b^2, d^2)` estimates the error of the sample
    covariance ``S``.
    """
    t, n = x.shape
    xc = x - x.mean(axis=0, keepdims=True)
    s = (xc.T @ xc) / t  # MLE sample covariance (1/T), matching the LW derivation

    mu = np.trace(s) / n
    d2 = np.sum((s - mu * np.eye(n)) ** 2)

    # b_bar^2 = (1/T^2) * sum_t || x_t x_t^T - S ||_F^2, computed in closed form:
    #   sum_t ||x_t x_t^T||^2 = sum_t ||x_t||^4 ;  cross/term collapses to T||S||^2.
    sq_row_norms = np.einsum("ij,ij->i", xc, xc)  # ||x_t||^2 per observation
    sum_quartic = np.sum(sq_row_norms**2)
    fro_s2 = np.sum(s**2)
    b_bar2 = (sum_quartic - t * fro_s2) / (t**2)

    b2 = min(b_bar2, d2)
    shrink = 0.0 if d2 == 0 else b2 / d2
    logger.info("Ledoit-Wolf shrinkage intensity: %.4f", shrink)

    return shrink * mu * np.eye(n) + (1.0 - shrink) * s


def _nearest_psd(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone via eigenvalue clipping."""
    sym = (cov + cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    n_negative = int((eigvals < 0).sum())
    if n_negative:
        logger.warning("Clipping %d negative eigenvalue(s) for PSD repair.", n_negative)
    eigvals_clipped = np.clip(eigvals, eps, None)
    repaired = (eigvecs * eigvals_clipped) @ eigvecs.T
    return (repaired + repaired.T) / 2.0


def correlation_from_covariance(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix."""
    std = np.sqrt(np.diag(cov.to_numpy()))
    denom = np.outer(std, std)
    # A zero-variance asset has an undefined correlation; avoid 0/0 -> nan warnings
    # by treating its off-diagonal correlations as 0 and its self-correlation as 1.
    safe_denom = np.where(denom == 0.0, 1.0, denom)
    corr = cov.to_numpy() / safe_denom
    np.fill_diagonal(corr, np.where(std == 0.0, 1.0, np.diag(corr)))
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)
