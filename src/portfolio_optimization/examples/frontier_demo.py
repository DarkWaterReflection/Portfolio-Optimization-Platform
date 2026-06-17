"""End-to-end demo: fetch real data, estimate inputs, build the efficient frontier.

Run with::

    python -m portfolio_optimization.examples.frontier_demo
    python -m portfolio_optimization.examples.frontier_demo --universe demo_uk

Requires network access on first run (results are cached to ``data_cache/`` for
subsequent offline runs). Prints the minimum-variance and maximum-Sharpe
portfolios and a short frontier table.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from portfolio_optimization.config import load_settings, load_universe
from portfolio_optimization.data.ingestion import DataIngestion
from portfolio_optimization.estimation.returns import expected_returns
from portfolio_optimization.estimation.risk import covariance_matrix
from portfolio_optimization.logging_setup import configure_logging
from portfolio_optimization.optimization.frontier import efficient_frontier
from portfolio_optimization.optimization.result import Constraints


def _format_weights(label: str, result) -> str:  # type: ignore[no-untyped-def]
    lines = [f"\n{label}:  return={result.expected_return:6.2%}  "
             f"vol={result.volatility:6.2%}  Sharpe={result.sharpe:5.2f}"]
    for ticker, w in result.top_holdings(8).items():
        if abs(w) > 1e-4:
            lines.append(f"    {ticker:8s} {w:7.2%}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Efficient frontier demo")
    parser.add_argument("--universe", default="demo_us", help="universe name in universes.yaml")
    parser.add_argument("--years", type=int, default=3, help="lookback window in years")
    args = parser.parse_args()

    configure_logging()
    settings = load_settings()
    tickers = load_universe(args.universe)

    end = date.today()
    start = end - timedelta(days=365 * args.years + 10)

    ingestion = DataIngestion(settings)
    returns = ingestion.get_returns(tickers, start, end, frequency="daily")

    mu = expected_returns(
        returns,
        method=settings.estimation.return_model,
        periods_per_year=settings.market.periods_per_year,
    )
    cov = covariance_matrix(
        returns,
        method=settings.estimation.risk_model,
        periods_per_year=settings.market.periods_per_year,
    )

    cons = Constraints(
        long_only=settings.optimization.long_only,
        w_min=settings.optimization.w_min,
        w_max=settings.optimization.w_max,
    )
    frontier = efficient_frontier(
        mu, cov, risk_free_rate=settings.market.risk_free_rate,
        constraints=cons, n_points=settings.optimization.frontier_points,
    )

    print(f"\n=== Efficient Frontier: {args.universe} ({len(mu)} assets, {args.years}y daily) ===")
    print(_format_weights("Minimum Variance", frontier.min_variance))
    print(_format_weights("Maximum Sharpe ", frontier.max_sharpe))
    print(f"\nFrontier points solved: {len(frontier.points)}")
    print(frontier.to_frame().describe().loc[["min", "max"]].to_string())


if __name__ == "__main__":
    main()
