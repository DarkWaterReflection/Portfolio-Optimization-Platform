"""Portfolio Optimization Platform — Streamlit dashboard.

A thin UI over the platform engines: the dashboard collects parameters and
renders results, but holds **no** quantitative logic of its own — every number
comes from the ``portfolio_optimization`` package. Heavy steps (data fetch,
optimization, backtest) are cached on their inputs so re-renders are instant.

Launch::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from portfolio_optimization.analytics.attribution import risk_attribution
from portfolio_optimization.analytics.report import compare_strategies, performance_report
from portfolio_optimization.backtesting.engine import Backtester
from portfolio_optimization.backtesting.strategy import EqualWeightStrategy, MeanVarianceStrategy
from portfolio_optimization.config import load_settings, load_universe
from portfolio_optimization.data.ingestion import DataIngestion
from portfolio_optimization.estimation.returns import expected_returns
from portfolio_optimization.estimation.risk import covariance_matrix
from portfolio_optimization.optimization.frontier import efficient_frontier
from portfolio_optimization.optimization.random_portfolios import random_portfolios
from portfolio_optimization.optimization.result import Constraints
from portfolio_optimization.visualization import (
    allocation_bar,
    correlation_heatmap,
    covariance_heatmap,
    drawdown_figure,
    efficient_frontier_figure,
    equity_curve_figure,
    rolling_sharpe_figure,
)

_BENCHMARKS = {"demo_us": "^GSPC", "demo_uk": "^FTSE"}
SETTINGS = load_settings()


# --- Cached data / compute layer ----------------------------------------------

@st.cache_data(show_spinner="Fetching market data…")
def fetch_returns(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    return DataIngestion(SETTINGS).get_returns(list(tickers), start, end, frequency="daily")


@st.cache_data(show_spinner="Fetching benchmark…")
def fetch_benchmark(ticker: str, start: date, end: date) -> pd.Series:
    return DataIngestion(SETTINGS).get_returns([ticker], start, end, frequency="daily").iloc[:, 0]


@st.cache_data(show_spinner=False)
def estimate_inputs(returns: pd.DataFrame, risk_model: str, ppy: int):
    mu = expected_returns(returns, periods_per_year=ppy)
    cov = covariance_matrix(returns, method=risk_model, periods_per_year=ppy)
    return mu, cov


@st.cache_data(show_spinner="Building efficient frontier…")
def build_frontier(returns, risk_model, ppy, rf, w_max, long_only, n_points):
    mu, cov = estimate_inputs(returns, risk_model, ppy)
    cons = Constraints(long_only=long_only, w_max=w_max)
    front = efficient_frontier(mu, cov, rf, cons, n_points)
    cloud = random_portfolios(mu, cov, rf, n_portfolios=4000, seed=SETTINGS.random.seed)
    return front, cloud, mu, cov


@st.cache_data(show_spinner="Running walk-forward backtests…")
def run_backtests(returns, risk_model, ppy, rf, w_max, long_only, frequency, lookback, cost_bps):
    cons = Constraints(long_only=long_only, w_max=w_max)
    strategies = [
        EqualWeightStrategy(),
        MeanVarianceStrategy("min_variance", rf, ppy, cons, risk_model),
        MeanVarianceStrategy("max_sharpe", rf, ppy, cons, risk_model),
    ]
    bt = Backtester(SETTINGS)
    return [bt.run(returns, s, frequency=frequency, lookback=lookback, cost_bps=cost_bps)
            for s in strategies]


# --- UI ------------------------------------------------------------------------

def sidebar() -> dict:
    st.sidebar.header("Configuration")
    universe = st.sidebar.selectbox("Universe", ["demo_us", "demo_uk"], index=0)
    all_tickers = load_universe(universe)
    tickers = st.sidebar.multiselect("Tickers", all_tickers, default=all_tickers)

    years = st.sidebar.slider("Lookback (years)", 2, 10, 5)
    rf = st.sidebar.number_input("Risk-free rate", 0.0, 0.10, SETTINGS.market.risk_free_rate, 0.005)

    st.sidebar.subheader("Optimization")
    long_only = st.sidebar.checkbox("Long only", value=True)
    w_max = st.sidebar.slider("Max weight per asset", 0.1, 1.0, 0.40, 0.05)
    risk_model = st.sidebar.selectbox("Risk model", ["ledoit_wolf", "sample"], index=0)

    st.sidebar.subheader("Backtest")
    frequency = st.sidebar.selectbox(
        "Rebalance frequency", ["monthly", "quarterly", "semi-annual", "annual"], index=1
    )
    lookback = st.sidebar.slider("Estimation window (days)", 60, 504, 252, 21)
    cost_bps = st.sidebar.slider("Transaction cost (bps)", 0.0, 50.0, 10.0, 1.0)

    return {
        "universe": universe, "tickers": tuple(tickers), "years": years, "rf": rf,
        "long_only": long_only, "w_max": w_max, "risk_model": risk_model,
        "frequency": frequency, "lookback": lookback, "cost_bps": cost_bps,
    }


def main() -> None:
    st.set_page_config(page_title="Portfolio Optimization Platform", layout="wide")
    st.title("📈 Portfolio Optimization Platform")
    st.caption("Modern Portfolio Theory · Efficient Frontier · Walk-forward Backtesting")

    cfg = sidebar()
    if len(cfg["tickers"]) < 2:
        st.warning("Select at least two tickers to continue.")
        st.stop()

    ppy = SETTINGS.market.periods_per_year
    end = date.today()
    start = end - timedelta(days=365 * cfg["years"] + 10)

    returns = fetch_returns(cfg["tickers"], start, end)
    bench_ticker = _BENCHMARKS.get(cfg["universe"], "^GSPC")
    benchmark = fetch_benchmark(bench_ticker, start, end)

    tabs = st.tabs(
        ["Universe & Data", "Risk / Return", "Optimization", "Backtest", "Rebalancing"]
    )

    with tabs[0]:
        st.subheader("Price-implied cumulative returns")
        st.line_chart((1 + returns).cumprod())
        c1, c2 = st.columns(2)
        c1.metric("Assets", returns.shape[1])
        c2.metric("Trading days", returns.shape[0])
        st.dataframe(returns.describe().T, use_container_width=True)

    front, cloud, mu, cov = build_frontier(
        returns, cfg["risk_model"], ppy, cfg["rf"], cfg["w_max"], cfg["long_only"],
        SETTINGS.optimization.frontier_points,
    )

    with tabs[1]:
        st.subheader("Annualized expected return & volatility")
        inputs = pd.DataFrame({"expected_return": mu, "volatility": pd.Series(
            {t: cov.loc[t, t] ** 0.5 for t in mu.index})})
        st.dataframe(inputs.style.format("{:.2%}"), use_container_width=True)
        h1, h2 = st.columns(2)
        h1.plotly_chart(correlation_heatmap(cov), use_container_width=True)
        h2.plotly_chart(covariance_heatmap(cov), use_container_width=True)

    with tabs[2]:
        st.plotly_chart(
            efficient_frontier_figure(front, cloud, cfg["rf"]), use_container_width=True
        )
        a1, a2 = st.columns(2)
        a1.plotly_chart(
            allocation_bar(front.max_sharpe.weights, name="MaxSharpe"), use_container_width=True
        )
        a2.plotly_chart(
            allocation_bar(front.min_variance.weights, name="MinVariance"),
            use_container_width=True,
        )
        st.markdown("**Risk attribution — Max Sharpe**")
        attr = risk_attribution(front.max_sharpe.weights, cov).sort_values(
            "pct_contribution", ascending=False
        )
        st.dataframe(attr.style.format("{:.4f}"), use_container_width=True)
        st.download_button(
            "⬇ Download Max-Sharpe weights (CSV)",
            front.max_sharpe.weights.to_csv(), file_name="max_sharpe_weights.csv",
        )

    results = run_backtests(
        returns, cfg["risk_model"], ppy, cfg["rf"], cfg["w_max"], cfg["long_only"],
        cfg["frequency"], cfg["lookback"], cfg["cost_bps"],
    )
    common_start = max(r.net_returns.index.min() for r in results)
    bench_bt = benchmark.loc[benchmark.index >= common_start]

    reports = [r.to_report(benchmark=bench_bt, risk_free_rate=cfg["rf"], periods_per_year=ppy)
               for r in results]
    reports.append(performance_report(bench_bt, bench_ticker, cfg["rf"], periods_per_year=ppy))
    table = compare_strategies(reports, rank_by="sharpe_ratio")

    with tabs[3]:
        curves = {r.name: r.equity_curve for r in results}
        curves[bench_ticker] = (1 + bench_bt).cumprod()
        st.plotly_chart(equity_curve_figure(curves), use_container_width=True)

        rets = {r.name: r.net_returns for r in results}
        rets[bench_ticker] = bench_bt
        d1, d2 = st.columns(2)
        d1.plotly_chart(drawdown_figure(rets), use_container_width=True)
        d2.plotly_chart(
            rolling_sharpe_figure(rets, risk_free_rate=cfg["rf"], periods_per_year=ppy),
            use_container_width=True,
        )
        st.subheader("Performance comparison (out-of-sample, net of costs)")
        st.dataframe(table.style.format("{:.3f}", na_rep="—"), use_container_width=True)
        st.download_button(
            "⬇ Download comparison (CSV)", table.to_csv(), file_name="strategy_comparison.csv"
        )

    with tabs[4]:
        st.subheader("Turnover & transaction-cost impact")
        rows = [
            {
                "strategy": r.name,
                "rebalances": r.n_rebalances,
                "total_turnover": r.total_turnover,
                "cost_drag_%": r.total_cost * 100,
                "ann_return_%": rep.annualized_return * 100,
                "sharpe": rep.sharpe_ratio,
            }
            for r, rep in zip(results, reports[:-1], strict=True)
        ]
        st.dataframe(pd.DataFrame(rows).set_index("strategy"), use_container_width=True)
        st.info(
            f"Current frequency: **{cfg['frequency']}** at **{cfg['cost_bps']:.0f} bps**. "
            "More frequent rebalancing tracks targets more tightly but raises turnover and cost — "
            "compare by changing the frequency control in the sidebar."
        )


# Streamlit executes the script top-level, so invoke the app directly.
main()
