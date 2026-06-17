"""Out-of-sample walk-forward backtest on real data.

Unlike ``analytics_demo`` (in-sample), this re-estimates inputs and re-optimizes
at every rebalance using only prior data, then compares Equal-Weight,
Minimum-Variance, and Maximum-Sharpe strategies against the index benchmark --
all run through the *same* point-in-time engine over the *same* dates.

Run::

    python -m portfolio_optimization.examples.backtest_demo --universe demo_us
    python -m portfolio_optimization.examples.backtest_demo --frequency quarterly
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from portfolio_optimization.analytics.report import compare_strategies, performance_report
from portfolio_optimization.backtesting.engine import Backtester
from portfolio_optimization.backtesting.strategy import EqualWeightStrategy, MeanVarianceStrategy
from portfolio_optimization.config import load_settings, load_universe
from portfolio_optimization.data.ingestion import DataIngestion
from portfolio_optimization.logging_setup import configure_logging
from portfolio_optimization.optimization.result import Constraints

_BENCHMARKS = {"demo_us": "^GSPC", "demo_uk": "^FTSE"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest demo")
    parser.add_argument("--universe", default="demo_us")
    parser.add_argument("--years", type=int, default=6)
    parser.add_argument("--frequency", default="quarterly")
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--cost-bps", type=float, default=10.0)
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
    bench_ticker = _BENCHMARKS.get(args.universe, "^GSPC")
    bench = ing.get_returns([bench_ticker], start, end, frequency="daily").iloc[:, 0]

    cons = Constraints(long_only=True, w_max=0.40)
    strategies = [
        EqualWeightStrategy(),
        MeanVarianceStrategy("min_variance", rf, ppy, cons, settings.estimation.risk_model),
        MeanVarianceStrategy("max_sharpe", rf, ppy, cons, settings.estimation.risk_model),
    ]

    bt = Backtester(settings)
    results = [
        bt.run(rets, s, frequency=args.frequency, lookback=args.lookback, cost_bps=args.cost_bps)
        for s in strategies
    ]

    # Align the benchmark to the common backtest window for fair comparison.
    common_start = max(r.net_returns.index.min() for r in results)
    bench_bt = bench.loc[bench.index >= common_start]

    reports = [r.to_report(benchmark=bench_bt, risk_free_rate=rf, periods_per_year=ppy)
               for r in results]
    reports.append(
        performance_report(bench_bt, bench_ticker, risk_free_rate=rf, periods_per_year=ppy)
    )

    table = compare_strategies(reports, rank_by="sharpe_ratio")
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(f"\n=== Walk-forward backtest: {args.universe} | {args.frequency} rebalancing "
          f"| {args.cost_bps:.0f}bps | {args.years}y ===\n")
    cols = ["annualized_return", "annualized_volatility", "sharpe_ratio",
            "max_drawdown", "beta", "alpha", "information_ratio"]
    print(table[cols + ["rank_risk_adjusted"]].to_string())

    print("\n--- Turnover & cost (out-of-sample) ---")
    for r in results:
        print(f"  {r.name:12s} rebalances={r.n_rebalances:3d}  "
              f"total_turnover={r.total_turnover:6.2f}  "
              f"cost_drag={r.total_cost * 100:5.2f}%")


if __name__ == "__main__":
    main()
