# Portfolio Optimization Platform (`pyfolio-opt`)

Institutional-grade portfolio construction, optimization, and backtesting on real
market data (S&P 500 & FTSE 100), built around Modern Portfolio Theory.

> **Status:** Feature-complete end to end — data ingestion, return/risk
> estimation, the three core optimizers (Minimum Variance, Maximum Sharpe,
> Efficient Frontier), ex-post analytics, a walk-forward backtester with
> rebalancing and transaction costs, Plotly visualizations, and a Streamlit
> dashboard. The math is documented in `docs/theory.md` and the layering in
> `docs/architecture.md`.

## Quickstart

```bash
python -m pip install -e ".[dev,viz,dashboard]"   # package + dev/plot/UI extras
pytest -q                                         # full suite, offline & deterministic
python -m portfolio_optimization.examples.frontier_demo    # fetch data + plot a frontier
streamlit run dashboard/app.py                    # interactive dashboard
```

## What's here

- **Data layer** — pluggable `DataSource` (yfinance impl), content-keyed Parquet
  cache, return computation (daily/monthly/annualized), an offline mode for
  reproducible runs.
- **Estimation** — expected returns (mean-historical / EWM) and a covariance model
  with sample + **Ledoit–Wolf shrinkage** and PSD repair.
- **Optimization** — Minimum-Variance, Maximum-Sharpe (multi-start SLSQP),
  Efficient Frontier (target-return QP sweep), and a random-portfolio cloud for
  visualization.
- **Analytics** — Sharpe / Sortino / Calmar / Information Ratio, drawdown,
  beta / Jensen's alpha, and **Euler risk attribution** (contributions sum exactly
  to portfolio volatility), with multi-strategy comparison reports.
- **Backtesting** — walk-forward engine with strict **point-in-time / no-lookahead**
  discipline, weight drift, periodic rebalancing (monthly → annual), and linear
  transaction costs in basis points.
- **Visualization & dashboard** — pure Plotly figure builders (frontier, heatmaps,
  allocation, equity/drawdown/rolling-Sharpe) and a thin Streamlit UI that holds
  no quantitative logic of its own.
- **Validation** — numerical solvers checked against closed-form solutions to 1e-6;
  the no-lookahead invariant asserted by a spy-strategy test.

## Architecture

```
Dashboard (Streamlit) → Engines (optimization / analytics / backtest / viz)
                      → Estimation (expected_returns, covariance_matrix)
                      → Data (DataSource, cache, cleaning)
```

Strict layering, dependency inversion at every seam, typed frozen dataclasses at
module boundaries. See `docs/architecture.md` and `docs/theory.md`.

## Key design stances

- Frontier is solved as a **quadratic program**, not found by random sampling.
- Covariance is **shrunk by default** (Ledoit–Wolf) — sample covariance on many
  names with limited history is an error-maximizer.
- Annualization conventions are explicit and tested (`periods_per_year`).
