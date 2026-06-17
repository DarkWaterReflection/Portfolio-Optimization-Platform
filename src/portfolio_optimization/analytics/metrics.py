r"""Performance and risk metrics on periodic return series.

Conventions (documented because they silently corrupt results otherwise):

* **Sharpe / Sortino** use *arithmetic* mean excess return per period, annualized
  by :math:`\sqrt{P}` (matches the ``empyrical`` convention), where the annual
  risk-free rate is de-annualized geometrically to a per-period figure.
* **Annualized return** is *geometric* (compounded), the honest CAGR.
* **Volatility** uses the sample standard deviation (``ddof=1``).
* **Drawdown** is computed on the compounded growth curve.
* **Beta / alpha / information ratio** require a benchmark return series aligned
  to the same dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _per_period_rf(risk_free_rate: float, periods_per_year: int) -> float:
    """Geometrically de-annualize an annual risk-free rate to one period."""
    return (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Inner-join two return series on their index and drop NaNs."""
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    return joined.iloc[:, 0], joined.iloc[:, 1]


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compounded growth of 1 unit: ``cumprod(1 + r)``."""
    return (1.0 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    """Total compounded return over the whole series."""
    return float((1.0 + returns).prod() - 1.0)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Geometric (compounded) annualized return — the CAGR."""
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return float("nan")
    growth = float((1.0 + r).prod())
    if growth <= 0:  # total wipeout; CAGR undefined, report -100%
        return -1.0
    return growth ** (periods_per_year / n) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of periodic returns."""
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252
) -> float:
    """Annualized Sharpe ratio (arithmetic mean excess / volatility)."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_p = _per_period_rf(risk_free_rate, periods_per_year)
    excess = r - rf_p
    sd = excess.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def downside_deviation(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Annualized downside deviation below the per-period MAR (= rf).

    Uses the full sample size in the denominator (the standard semideviation
    convention), not just the count of downside observations.
    """
    r = returns.dropna()
    rf_p = _per_period_rf(risk_free_rate, periods_per_year)
    shortfall = np.minimum(r - rf_p, 0.0)
    dd = np.sqrt(np.mean(shortfall**2))
    return float(dd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252
) -> float:
    """Annualized Sortino ratio (excess return / downside deviation)."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_p = _per_period_rf(risk_free_rate, periods_per_year)
    dd = downside_deviation(returns, risk_free_rate, periods_per_year)
    if dd == 0:
        return float("nan")
    return float((r - rf_p).mean() * periods_per_year / dd)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown at each point: ``curve / running_max - 1`` (values in [-1, 0])."""
    curve = cumulative_returns(returns)
    running_max = curve.cummax()
    return curve / running_max - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (a negative number)."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else float("nan")


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return divided by the absolute maximum drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return annualized_return(returns, periods_per_year) / abs(mdd)


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """CAPM beta: ``cov(r, b) / var(b)`` on aligned dates."""
    r, b = _align(returns, benchmark)
    if len(r) < 2:
        return float("nan")
    var_b = b.var(ddof=1)
    if var_b == 0:
        return float("nan")
    return float(r.cov(b) / var_b)


def jensen_alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Annualized Jensen's alpha: ``R_p - [rf + beta (R_b - rf)]`` (CAPM).

    Returns are annualized geometrically before applying the CAPM relation.
    """
    r, b = _align(returns, benchmark)
    bta = beta(r, b)
    ann_p = annualized_return(r, periods_per_year)
    ann_b = annualized_return(b, periods_per_year)
    return float(ann_p - (risk_free_rate + bta * (ann_b - risk_free_rate)))


def tracking_error(
    returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252
) -> float:
    """Annualized standard deviation of active (portfolio minus benchmark) returns."""
    r, b = _align(returns, benchmark)
    active = r - b
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(
    returns: pd.Series, benchmark: pd.Series, periods_per_year: int = 252
) -> float:
    """Annualized information ratio: active return / tracking error."""
    r, b = _align(returns, benchmark)
    active = r - b
    sd = active.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(active.mean() / sd * np.sqrt(periods_per_year))


def rolling_sharpe(
    returns: pd.Series,
    window: int = 126,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rolling annualized Sharpe ratio over a trailing ``window``."""
    rf_p = _per_period_rf(risk_free_rate, periods_per_year)
    excess = returns.dropna() - rf_p
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std(ddof=1)
    return (mean / std * np.sqrt(periods_per_year)).rename("rolling_sharpe")
