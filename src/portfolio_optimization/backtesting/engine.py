r"""Walk-forward backtesting engine with strict point-in-time discipline.

Mechanics, per rebalance date :math:`d`:

1. The estimation window is the last ``lookback`` returns with index **strictly
   before** :math:`d`. The strategy sees nothing dated :math:`\ge d` — this is the
   no-lookahead invariant, asserted directly by the test suite.
2. Target weights are set at the open of :math:`d`; the portfolio earns day
   :math:`d`'s return with the new weights.
3. Between rebalances, weights **drift** with realized asset returns.
4. On a rebalance day, one-way turnover and a linear transaction cost are charged
   as a drag on that day's net return.

Net daily returns compound into the equity curve, so
``equity_curve.pct_change() == net_returns`` by construction; a stronger
reconciliation (a single-asset strategy reproduces that asset's returns) is
tested separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_optimization.backtesting.result import BacktestResult
from portfolio_optimization.backtesting.strategy import Strategy
from portfolio_optimization.config import Settings, load_settings
from portfolio_optimization.logging_setup import get_logger
from portfolio_optimization.rebalancing.costs import transaction_cost, turnover
from portfolio_optimization.rebalancing.schedule import rebalance_dates

logger = get_logger("backtesting.engine")


class Backtester:
    """Run a rebalanced strategy over a panel of realized returns."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

    def run(
        self,
        returns: pd.DataFrame,
        strategy: Strategy,
        frequency: str = "monthly",
        lookback: int = 252,
        cost_bps: float = 10.0,
        min_obs: int | None = None,
    ) -> BacktestResult:
        """Backtest ``strategy`` over ``returns`` with periodic rebalancing.

        Args:
            returns: Daily asset returns (dates x tickers), cleaned & aligned.
            strategy: A :class:`Strategy` mapping a returns window to weights.
            frequency: Rebalance frequency (see ``REBALANCE_FREQUENCIES``).
            lookback: Trailing window length (trading days) for estimation.
            cost_bps: One-way transaction cost in basis points of turnover.
            min_obs: Minimum observations required before the first rebalance
                executes (defaults to ``lookback``).

        Returns:
            A :class:`BacktestResult` with net/gross returns, equity curve,
            weights history, turnover, and costs.
        """
        if returns.empty:
            raise ValueError("returns panel is empty")
        returns = returns.sort_index()
        min_obs = min_obs if min_obs is not None else lookback
        assets = returns.columns

        rebal_set = set(rebalance_dates(returns.index, frequency))

        weights: np.ndarray | None = None
        gross: dict[pd.Timestamp, float] = {}
        cost_series: dict[pd.Timestamp, float] = {}
        turnover_hist: dict[pd.Timestamp, float] = {}
        weights_hist: dict[pd.Timestamp, np.ndarray] = {}
        executed_dates: list[pd.Timestamp] = []

        for d in returns.index:
            day_cost = 0.0

            if d in rebal_set:
                # Point-in-time: only data strictly before d is visible.
                window = returns.loc[returns.index < d].tail(lookback)
                if len(window) >= min_obs:
                    target = strategy.weights(window).reindex(assets).fillna(0.0)
                    target_arr = target.to_numpy(dtype=float)
                    prev = weights if weights is not None else np.zeros(len(assets))
                    to = turnover(prev, target_arr)
                    day_cost = transaction_cost(to, cost_bps)

                    weights = target_arr
                    turnover_hist[d] = to
                    weights_hist[d] = target_arr
                    executed_dates.append(d)

            if weights is None:
                # Not yet invested (still warming up): no position, skip the day.
                continue

            day_ret = returns.loc[d].to_numpy(dtype=float)
            gross_ret = float(weights @ day_ret)
            gross[d] = gross_ret
            cost_series[d] = day_cost

            # Drift weights with realized gross performance for the next day.
            denom = 1.0 + gross_ret
            if denom > 0:
                weights = weights * (1.0 + day_ret) / denom

        if not gross:
            raise ValueError(
                "No investable period: lookback/min_obs exceed available history."
            )

        gross_returns = pd.Series(gross, name=f"{strategy.name}_gross").sort_index()
        costs = pd.Series(cost_series, name="cost").reindex(gross_returns.index).fillna(0.0)
        net_returns = (gross_returns - costs).rename(strategy.name)
        equity_curve = (1.0 + net_returns).cumprod().rename(f"{strategy.name}_equity")

        weights_history = pd.DataFrame(
            {d: w for d, w in weights_hist.items()}, index=assets
        ).T.sort_index()
        turnover_ser = pd.Series(turnover_hist, name="turnover").sort_index()

        logger.info(
            "%s: %d days, %d rebalances, total turnover=%.2f, cost drag=%.3f%%",
            strategy.name, len(net_returns), len(executed_dates),
            turnover_ser.sum(), costs.sum() * 100,
        )

        return BacktestResult(
            name=strategy.name,
            net_returns=net_returns,
            gross_returns=gross_returns,
            equity_curve=equity_curve,
            weights_history=weights_history,
            turnover=turnover_ser,
            costs=costs,
            rebalance_dates=executed_dates,
        )
