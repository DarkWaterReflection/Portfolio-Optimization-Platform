# Theory & Derivations

This document derives the mathematics the platform implements, states the
estimation and annualization **conventions** explicitly (they silently corrupt
results when left implicit), and records the design stances that follow from the
theory. Every formula here maps to a tested function; the validation suite checks
the numerical solvers against the closed forms below to `1e-6`.

Notation: $N$ assets, weight vector $w\in\mathbb R^N$, expected (annualized)
return vector $\mu$, annualized covariance $\Sigma$ (symmetric PSD), risk-free
rate $r_f$, the all-ones vector $\mathbf 1$.

---

## 1. Modern Portfolio Theory

A portfolio's first two annualized moments are

$$\mu_p = w'\mu, \qquad \sigma_p^2 = w'\Sigma w, \qquad
\text{Sharpe} = \frac{w'\mu - r_f}{\sqrt{w'\Sigma w}}.$$

Implemented in `optimization/base.py` (`portfolio_return`,
`portfolio_volatility`, `portfolio_sharpe`). $\sigma_p^2$ is clamped at 0 before
the square root so that floating-point negatives near the PSD boundary never
produce `nan`.

### 1.1 Minimum-variance portfolio

$$\min_w \; w'\Sigma w \quad \text{s.t.}\quad \mathbf 1'w = 1.$$

With only the budget constraint, a Lagrangian
$\mathcal L = w'\Sigma w - \lambda(\mathbf 1'w - 1)$ gives
$\nabla_w\mathcal L = 2\Sigma w - \lambda\mathbf 1 = 0
\Rightarrow w = \tfrac{\lambda}{2}\Sigma^{-1}\mathbf 1$, and normalizing to the
budget:

$$\boxed{\;w_{\text{mv}} = \frac{\Sigma^{-1}\mathbf 1}{\mathbf 1'\Sigma^{-1}\mathbf 1}\;}$$

This is `min_variance_closed_form`. The constrained problem (weight bounds,
caps) is solved as a QP by SLSQP in `min_variance_portfolio`; when the bounds are
slack the two agree to solver tolerance — asserted in
`tests/validation/test_closed_form.py`.

### 1.2 Maximum-Sharpe (tangency) portfolio

The tangency portfolio maximizes the Sharpe ratio. The ratio is **not convex** in
$w$, but the budget-only optimum has a closed form. Maximizing
$(w'\mu - r_f)/\sqrt{w'\Sigma w}$ is invariant to scaling $w$; fixing the excess
return and minimizing variance yields

$$\boxed{\;w_{\text{tan}} \propto \Sigma^{-1}(\mu - r_f\mathbf 1),\qquad
\text{then renormalize } \mathbf 1'w = 1.\;}$$

This is `max_sharpe_closed_form`, used purely as a validation oracle. Under real
(long-only, capped) constraints there is no closed form, so
`max_sharpe_portfolio` minimizes $-\text{Sharpe}$ with **multi-start SLSQP**
(equal-weight start plus seeded Dirichlet draws) and keeps the best feasible
optimum. Multi-start is the practical defense against the non-convexity: a single
start can stall on a saddle or a boundary local optimum.

### 1.3 The efficient frontier as a QP — *not* random sampling

The frontier is the set of minimum-variance portfolios for each attainable target
return $\mu^\*$. We trace it by solving, over a grid
$\mu^\*\in[\mu_{\text{mv}},\, \max_i \mu_i]$,

$$\min_w \; w'\Sigma w \quad \text{s.t.}\quad
\mathbf 1'w = 1,\;\; w'\mu = \mu^\*,\;\; w_{\min}\le w_i \le w_{\max}.$$

Implemented in `optimization/frontier.py` (`_min_variance_for_target` swept by
`efficient_frontier`). After each solve we re-check $|w'\mu-\mu^\*|<10^{-6}$ and
drop the point if SLSQP reported success off-target — a guard against the solver
declaring victory on an infeasible target.

**Why not random portfolios?** Sampling weights from a Dirichlet simplex and
plotting their $(\sigma,\mu)$ cloud *never reaches the boundary* in high
dimension — the convex hull of random points sits strictly inside the true
frontier, and the gap grows with $N$. Random portfolios are therefore a
**visualization aid only** (`optimization/random_portfolios.py`), drawn as a
backdrop behind the analytically-solved frontier, never used to choose weights.

---

## 2. Estimation

### 2.1 Expected returns

`estimation/returns.py` supports the historical mean (`mean_historical`) and an
exponentially-weighted mean (`ewm`, half-life parameterized). Expected returns
are notoriously hard to estimate — small sample errors in $\mu$ dominate
optimized weights — which is *why* the platform leans on the Sharpe/min-variance
structure and on shrinkage of $\Sigma$ rather than chasing return forecasts.

### 2.2 Covariance: Ledoit–Wolf shrinkage (the default)

The sample covariance $S$ is unbiased but badly **conditioned** when $T$
(observations) is not $\gg N$. An optimizer inverts $\Sigma$, so it amplifies the
smallest, noisiest eigenvalues — the optimizer becomes an *error maximizer*,
loading on directions that merely look low-variance in-sample.

Ledoit & Wolf (2004) shrink $S$ toward a scaled identity $\mu I$
($\mu=\operatorname{tr}(S)/N$) by the convex combination

$$\Sigma^\* = \delta^\*\,\mu I + (1-\delta^\*)\,S,$$

with the analytically optimal intensity minimizing
$\mathbb E\lVert\Sigma^\*-\Sigma\rVert_F^2$:

$$\delta^\* = \frac{b^2}{d^2},\qquad
d^2 = \lVert S-\mu I\rVert_F^2,\qquad
b^2 = \min\!\big(\bar b^2,\, d^2\big),$$

$$\bar b^2 = \frac{1}{T^2}\sum_{t=1}^{T}\big\lVert x_t x_t' - S\big\rVert_F^2
= \frac{1}{T^2}\Big(\sum_t \lVert x_t\rVert^4 - T\lVert S\rVert_F^2\Big),$$

where $x_t$ are the demeaned return rows and $S = \tfrac1T\sum_t x_t x_t'$ (the
MLE $1/T$ scaling that matches the derivation). The closed form for $\bar b^2$ is
what makes the estimator $O(TN^2)$ and vectorizable —
`_ledoit_wolf_identity` computes $\sum_t\lVert x_t\rVert^4$ with a single
`einsum`. $\delta^\*$ is clamped to $[0,1]$ implicitly via $b^2\le d^2$.

### 2.3 PSD repair

Shrinkage, NaN-handling, or numerical drift can leave $\Sigma$ marginally
non-PSD. `_nearest_psd` projects onto the PSD cone by eigen-decomposition and
clips eigenvalues to $\varepsilon=10^{-12}$, then re-symmetrizes. This guarantees
Cholesky/inversion downstream never fails.

### 2.4 Annualization conventions

| Quantity | Per-period → annual |
|---|---|
| Mean return $\mu$ | $\times P$ |
| Volatility $\sigma$ | $\times\sqrt P$ |
| Covariance $\Sigma$ | $\times P$ |
| Risk-free $r_f$ (annual → per period) | $(1+r_f)^{1/P}-1$ |

$P = $ `periods_per_year` (252 for daily). These are applied once, at the
estimation boundary, and every downstream metric assumes annualized inputs.

---

## 3. Performance & risk metrics

Conventions in `analytics/metrics.py` (chosen to match the widely-used
`empyrical` library so numbers are comparable):

- **Sharpe** — arithmetic mean *excess* return per period $\div$ sample stdev
  (`ddof=1`), annualized by $\sqrt P$. The annual $r_f$ is de-annualized
  geometrically first.
- **Sortino** — excess return $\div$ **downside deviation**
  $\sqrt{\tfrac1T\sum_t \min(r_t - r_f, 0)^2}$. The full sample size $T$ is the
  denominator (standard semideviation convention), not the count of down days.
- **Annualized return** — *geometric* CAGR
  $\big(\prod(1+r_t)\big)^{P/T} - 1$; a total wipeout reports $-100\%$ rather
  than `nan`.
- **Max drawdown** — min of $C_t/\max_{s\le t}C_s - 1$ on the compounded curve
  $C_t=\prod_{s\le t}(1+r_s)$.
- **Calmar** — CAGR $\div\,|\text{MDD}|$.
- **Beta / Jensen's alpha** — CAPM on date-aligned benchmark returns:
  $\beta=\operatorname{cov}(r,b)/\operatorname{var}(b)$,
  $\alpha = R_p - [r_f + \beta(R_b - r_f)]$ (returns annualized geometrically
  before the CAPM relation).
- **Information ratio** — mean active return $\div$ tracking error (annualized
  stdev of $r-b$).

Degenerate inputs (fewer than 2 observations, zero dispersion) return `nan`
rather than raising, so a report table never crashes on a thin series.

---

## 4. Euler risk attribution

Volatility $\sigma_p(w)=\sqrt{w'\Sigma w}$ is **homogeneous of degree 1** in $w$,
so Euler's theorem gives an *exact* additive decomposition:

$$\sigma_p = \sum_i w_i\,\frac{\partial \sigma_p}{\partial w_i}
= \sum_i \underbrace{w_i\,\frac{(\Sigma w)_i}{\sigma_p}}_{\text{CTR}_i}.$$

- **Marginal contribution to risk** $\text{MCR}_i = (\Sigma w)_i/\sigma_p$.
- **Component contribution** $\text{CTR}_i = w_i\,\text{MCR}_i$, and crucially
  $\sum_i \text{CTR}_i = \sigma_p$ **exactly** (an identity the test suite
  verifies, not an approximation).
- **Percent contribution** $\text{CTR}_i/\sigma_p$.

This is the meaningful sense of "how much risk does each holding contribute" —
unlike $w_i$ alone, it accounts for correlations. Implemented in
`analytics/attribution.py`. Return attribution is the trivial linear split
$w_i\mu_i$. The **diversification ratio**
$\big(\sum_i w_i\sigma_i\big)/\sigma_p \ge 1$ measures the benefit captured.

---

## 5. Backtesting & no-lookahead discipline

`backtesting/engine.py` runs a walk-forward simulation. At each rebalance date
$d$:

1. **Point-in-time estimation.** The window is the last `lookback` rows with index
   **strictly** $< d$. The strategy sees *nothing* dated $\ge d$. This
   no-lookahead invariant is asserted directly by a spy strategy in
   `tests/validation/test_no_lookahead.py`, which records `window.index.max()` and
   checks it is strictly earlier than $d$.
2. **Execution.** Target weights set at the open of $d$; the portfolio earns
   $d$'s return with the new weights.
3. **Drift.** Between rebalances, weights evolve with realized returns,
   $w \leftarrow w\,(1+r^{\text{asset}})/(1+r^{\text{gross}})$, so they always sum
   to 1 without spurious trading.
4. **Costs.** One-way **turnover** $\tfrac12\sum_i|w^{\text{new}}_i-w^{\text{old}}_i|$
   is charged a linear cost `cost_bps` (basis points) as a drag on that day's net
   return (`rebalancing/costs.py`).

By construction `equity_curve.pct_change() == net_returns`. A stronger
reconciliation test — a single-asset strategy reproduces that asset's return
stream to $10^{-12}$ — guards the accounting.

### The lookahead gap, demonstrated

On the demo universe, in-sample Max-Sharpe shows a Sharpe of ~2.3, but the honest
**out-of-sample** walk-forward ranks Equal-Weight (1/N) Sharpe ≈ 1.15 *above*
Max-Sharpe ≈ 1.08 and Min-Variance ≈ 0.81 — the DeMiguel–Garlappi–Uppal (2009)
"1/N is hard to beat" result, reproduced here precisely because the engine
forbids lookahead. The in-sample/out-of-sample Sharpe gap *is* the lookahead
bias, made visible.

---

## References

- Markowitz, H. (1952). *Portfolio Selection.* Journal of Finance.
- Ledoit, O., & Wolf, M. (2004). *A Well-Conditioned Estimator for
  Large-Dimensional Covariance Matrices.* Journal of Multivariate Analysis.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). *Optimal Versus Naive
  Diversification.* Review of Financial Studies.
- Sharpe, W. (1966); Sortino & Price (1994); Jensen (1968) — the eponymous ratios.
