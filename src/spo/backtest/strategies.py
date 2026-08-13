"""
Strategy registry.
Each function returns a callable `lookback_df -> weights_series` that fits
the backtest function. Strategies are looked up by name so we can iterate over them uniformly.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from spo.models import calibrate_gbm, calibrate_heston_multi, simulate_gbm_paths, simulate_heston_paths
from spo.optim import black_litterman_portfolio, min_cvar_portfolio, min_var_portfolio, robust_max_sharpe_portfolio, scenarios_from_sim

StrategyFn = Callable[[pd.DataFrame], pd.Series]


def equal_weight(**_) -> StrategyFn:
    def _fn(lookback: pd.DataFrame) -> pd.Series:
        n = lookback.shape[1]
        return pd.Series(1.0 / n, index=lookback.columns)
    return _fn


def min_var(covariance: str="ledoit_wolf", **_) -> StrategyFn:
    def _fn(lookback: pd.DataFrame) -> pd.Series:
        return min_var_portfolio(lookback, covariance=covariance)
    return _fn


def black_litterman(
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    market_weights: pd.Series | None = None,
    **_,
) -> StrategyFn:
    def _fn(lookback: pd.DataFrame) -> pd.Series:
        mw = market_weights
        if mw is None:
            # Inverse-vol weights computed from the lookback window.
            # No look-ahead bias; overweights low-vol (typically large-cap)
            # names, a reasonable market-cap proxy without external data.
            vol = lookback.std()
            mw = (1.0 / vol) / (1.0 / vol).sum()
        return black_litterman_portfolio(
            lookback, market_weights=mw,
            tau=tau, risk_aversion=risk_aversion,
        )
    return _fn


def robust_mv(kappa: float = 3.0, risk_aversion: float = 2.5, max_weight: float = 0.10, **_) -> StrategyFn:
    def _fn(lookback: pd.DataFrame) -> pd.Series:
        return robust_max_sharpe_portfolio(lookback, kappa=kappa,
                                           risk_aversion=risk_aversion,
                                           max_weight=max_weight)
    return _fn


def gbm_cvar(alpha: float=0.95, n_paths: int=3000, n_steps: int=21, seed: int=0, **_) -> StrategyFn:
    def _fn(lookback: pd.DataFrame) -> pd.Series:
        params = calibrate_gbm(lookback)
        sim = simulate_gbm_paths(params, n_paths=n_paths, n_steps=n_steps, seed=seed)
        scen = scenarios_from_sim(sim, return_type="simple")
        return min_cvar_portfolio(scen, alpha=alpha, tickers=params.tickers)
    return _fn


def heston_cvar(alpha: float=0.95, n_paths: int=3000, n_steps: int = 21, seed: int=0, **_) -> StrategyFn:
    def _fn(lookback: pd.DataFrame) -> pd.Series:
        params = calibrate_heston_multi(lookback)
        sim = simulate_heston_paths(params, n_paths=n_paths, n_steps=n_steps, seed=seed)
        scen = scenarios_from_sim(sim, return_type="simple")
        return min_cvar_portfolio(scen, alpha=alpha, tickers=params.tickers)
    return _fn


REGISTRY: dict[str, Callable[..., StrategyFn]] = {
    "equal_weight": equal_weight,
    "min_variance": min_var,
    "black_litterman": black_litterman,
    "robust_mv": robust_mv,
    "gbm_cvar": gbm_cvar,
    "heston_cvar": heston_cvar
}


def get_strategy(name: str, **params) -> StrategyFn:
    """
    Look up a strategy by name and return a function.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy {name}. Available: {list(REGISTRY)}")
    return REGISTRY[name](**params)