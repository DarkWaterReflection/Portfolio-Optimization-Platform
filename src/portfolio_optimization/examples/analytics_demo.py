"""End-to-end analytics demo on real data.

Builds Equal-Weight, Minimum-Variance, and Maximum-Sharpe portfolios, applies the
(static, in-sample) weights to realized returns, compares them against the index
benchmark, and prints performance + risk-attribution tables.

NOTE: applying optimized weights back to the same window is *in-sample* and will
flatter the optimizers. It validates the analytics plumbing, not out-of-sample
skill — that is the job of the Phase 5 walk-forward backtester.

Run::

    python -m portfolio_optimization.examples.analytics_demo --universe demo_us
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from portfolio_optimization.analytics.attribution import risk_attribution
from portfolio_optimization.analytics.report import compare_strategies, performance_report
from portfolio_optimization.config import load_settings, load_universe
from portfolio_optimization.data.ingestion import DataIngestion
from portfolio_optimization.estimation.returns import expected_returns
from portfolio_optimization.estimation.risk import covariance_matrix
from portfolio_optimization.logging_setup import configure_logging
from portfolio_optimization.optimization.base import equal_weights
from portfolio_optimization.optimization.max_sharpe import max_sharpe_portfolio
from portfolio_optimization.optimization.min_variance import min_variance_portfolio
from portfolio_optimization.optimization.result import Constraints

_BENCHMARKS = {"demo_us": "^GSPC", "demo_uk": "^FTSE"}


def _portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Realized periodic return of a static-weight portfolio."""
    aligned = weights.reindex(returns.columns).fillna(0.0)
    return returns.mul(aligned, axis=1).sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analytics demo")
    parser.add_argument("--universe", default="demo_us")
    parser.add_argument("--years", type=int, default=3)
    args = parser.parse_args()

    configure_logging()
    settings = load_settings()
    ppy = settings.market.periods_per_year
    rf = settings.market.risk_free_rate

    tickers = load_universe(args.universe)
    end = date.today()
    start = end - timedelta(days=365 * args.years + 10)

    ing = DataIngestion(settings)
    rets = ing.get_returns(tickers, start, end, frequency="daily")
    mu = expected_returns(rets, periods_per_year=ppy)
    cov = covariance_matrix(rets, method=settings.estimation.risk_model, periods_per_year=ppy)

    cons = Constraints(long_only=True, w_max=0.4)
    ew = pd.Series(equal_weights(len(mu)), index=mu.index)
    mv = min_variance_portfolio(mu, cov, rf, cons).weights
    ms = max_sharpe_portfolio(mu, cov, rf, cons).weights

    strategies = {"EqualWeight": ew, "MinVariance": mv, "MaxSharpe": ms}

    # Benchmark return series (aligned to the same dates).
    bench_ticker = _BENCHMARKS.get(args.universe, "^GSPC")
    bench_rets = ing.get_returns([bench_ticker], start, end, frequency="daily").iloc[:, 0]

    reports = [
        performance_report(
            _portfolio_returns(rets, w), name, benchmark=bench_rets,
            risk_free_rate=rf, periods_per_year=ppy,
        )
        for name, w in strategies.items()
    ]
    reports.append(
        performance_report(bench_rets, bench_ticker, risk_free_rate=rf, periods_per_year=ppy)
    )

    table = compare_strategies(reports, rank_by="sharpe_ratio")
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(f"\n=== Strategy comparison: {args.universe} ({args.years}y, in-sample) ===\n")
    cols = ["annualized_return", "annualized_volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "beta", "alpha", "information_ratio"]
    print(table[cols + ["rank_risk_adjusted"]].to_string())

    print("\n=== Risk attribution: MaxSharpe (top 6 by % of total risk) ===\n")
    attr = risk_attribution(ms, cov).sort_values("pct_contribution", ascending=False)
    print(attr.head(6).to_string(float_format=lambda x: f"{x:,.4f}"))


if __name__ == "__main__":
    main()
