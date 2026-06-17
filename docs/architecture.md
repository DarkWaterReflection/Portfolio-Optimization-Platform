# Architecture

A layered, dependency-inverted design. Each layer depends only on the contracts
(Protocols and frozen dataclasses) of the layer beneath it — never on a concrete
implementation, and never upward. This keeps the quantitative core testable
offline and deterministic, and keeps the UI a thin shell with no math of its own.

```
┌─────────────────────────────────────────────────────────────┐
│  Dashboard (Streamlit)            dashboard/app.py           │  ← UI only
│  • collects params, renders figures, caches on inputs        │
└───────────────┬─────────────────────────────────────────────┘
                │ calls
┌───────────────▼─────────────────────────────────────────────┐
│  Engines                                                     │
│  • optimization/   min-var, max-sharpe, frontier (QP)        │
│  • analytics/      metrics, attribution, reports             │
│  • backtesting/    walk-forward engine, strategies, results  │
│  • rebalancing/    schedule, turnover & cost models          │
│  • visualization/  pure Plotly figure builders               │
└───────────────┬─────────────────────────────────────────────┘
                │ depends on
┌───────────────▼─────────────────────────────────────────────┐
│  Estimation       estimation/                                │
│  • expected_returns()   • covariance_matrix() (Ledoit–Wolf)  │
└───────────────┬─────────────────────────────────────────────┘
                │ depends on
┌───────────────▼─────────────────────────────────────────────┐
│  Data             data/                                      │
│  • DataSource (Protocol) → YFinanceSource                    │
│  • cache (content-keyed Parquet)  • cleaning  • ingestion    │
└─────────────────────────────────────────────────────────────┘

Cross-cutting:  config/ (typed, frozen)   logging_setup/
```

## Dependency rules

1. **Downward only.** `optimization` may import `estimation`; `estimation` must
   not import `optimization`. The dashboard imports engines; nothing imports the
   dashboard.
2. **Inversion at every seam.** Data sources implement the `DataSource` Protocol
   (`data/sources/base.py`); the ingestion layer depends on the Protocol and
   resolves a concrete source through a registry, so adding Alpha Vantage or
   Polygon is a new file plus a `register_source` call — no edits upstream.
3. **Typed, immutable boundaries.** Modules exchange `@dataclass(frozen=True)`
   contracts (`Constraints`, `OptimizationResult`, `EfficientFrontierResult`,
   `FrontierPoint`, `BacktestResult`, `PerformanceReport`, `Settings`). Frozen =
   no caller can mutate a result and surprise another.
4. **Pure functions in the hot path.** The MPT moments, metrics, attribution, and
   every Plotly builder are pure: same inputs → same outputs, no I/O, no globals.
   This is what makes the validation tests deterministic.

## Module map

| Package | Responsibility | Key entry points |
|---|---|---|
| `config` | Load/validate YAML into frozen settings | `load_settings`, `load_universe` |
| `data.sources` | Pluggable market-data providers | `DataSource` Protocol, `YFinanceSource` |
| `data` | Fetch → cache → clean → returns | `DataIngestion.get_prices/get_returns` |
| `estimation` | Moments of the return distribution | `expected_returns`, `covariance_matrix` |
| `optimization` | Portfolio construction (QP/SLSQP) | `min_variance_portfolio`, `max_sharpe_portfolio`, `efficient_frontier` |
| `analytics` | Ex-post metrics, attribution, reports | `performance_report`, `risk_attribution`, `compare_strategies` |
| `rebalancing` | When to trade, what it costs | `rebalance_dates`, `turnover`, `transaction_cost` |
| `backtesting` | Walk-forward simulation | `Backtester.run`, `Strategy`, `BacktestResult` |
| `visualization` | Plotly figure builders | `efficient_frontier_figure`, `equity_curve_figure`, … |
| `dashboard` | Streamlit UI | `dashboard/app.py` |

## Data flow (a frontier render)

```
universes.yaml ─► DataIngestion.get_returns ─► clean_prices/daily_returns
        │                    │ (cache: SHA-256 content key → Parquet)
        ▼                    ▼
   tickers           returns DataFrame
                          │
        ┌─────────────────┼──────────────────┐
        ▼                                     ▼
 expected_returns(μ)                covariance_matrix(Σ)  ← Ledoit–Wolf + PSD
        └─────────────────┬──────────────────┘
                          ▼
             efficient_frontier(μ, Σ, rf, Constraints)
                          │  (+ random_portfolios cloud, viz only)
                          ▼
            efficient_frontier_figure(...)  ─►  Streamlit tab
```

## Design stances (and why)

- **Frontier is a QP, not a sample cloud.** Random sampling cannot reach the true
  boundary in high dimension (see `docs/theory.md` §1.3). The cloud is decoration.
- **Covariance is shrunk by default.** Sample covariance on many names with
  limited history is an error-maximizer once inverted; Ledoit–Wolf is the default
  risk model.
- **No-lookahead is enforced, not assumed.** The backtester only ever sees data
  strictly before the rebalance date, and a spy-strategy test asserts it.
- **Offline & deterministic tests.** The full suite runs without network
  (fixtures + cache + a fixed RNG seed), so CI is reproducible. Live data fetches
  are isolated behind the `network` pytest marker and the example scripts.
- **Annualization is explicit and tested.** `periods_per_year` flows from config;
  the annualization identities are unit-tested rather than trusted.

## Configuration & reproducibility

- `config/settings.yaml` — data source, cache dir, offline flag, market constants
  (`periods_per_year`, `risk_free_rate`), estimation/optimization defaults, RNG
  seed. Resolution order: explicit path → `PFOPT_CONFIG` env var → bundled YAML.
- `config/universes.yaml` — named asset universes (`demo_us`, `demo_uk`) and
  benchmarks (`^GSPC`, `^FTSE`).
- Everything random (Dirichlet clouds, multi-start optimizer seeds) is seeded, so
  results are reproducible run to run.

## Testing layout

| Path | What it guards |
|---|---|
| `tests/unit/` | Per-module behavior (estimation, optimization, analytics, backtesting, rebalancing, cleaning, visualization smoke tests) |
| `tests/validation/` | Numerical solvers vs. closed-form oracles; the no-lookahead invariant |
| `tests/conftest.py` | Shared deterministic fixtures (`mu_series`, `cov_frame`, …) |

Run: `pytest -q` (offline). Lint/type: `ruff check src tests`, `mypy src`.
