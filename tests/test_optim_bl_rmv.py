"""
Tests for Black-Litterman and robust MV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spo.optim import (
    black_litterman_portfolio,
    black_litterman_posterior,
    implied_equilibrium_returns,
    min_var_portfolio,
    robust_max_sharpe_portfolio,
)


@pytest.fixture
def returns_and_market():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2018-01-01", periods=1000)
    n = 5
    L = rng.standard_normal((n, n)) * 0.3
    cov = L @ L.T + np.eye(n) * 0.0001
    r = rng.multivariate_normal(np.full(n, 0.0005), cov, size=len(idx))
    returns = pd.DataFrame(r, index=idx, columns=[f"A{i}" for i in range(n)])
    mkt = pd.Series(1.0 / n, index=returns.columns)
    return returns, mkt


@pytest.fixture
def daily_returns():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2018-01-01", periods=1500)
    n = 20
    L = rng.standard_normal((n, n)) * 0.3
    cov = L @ L.T + np.eye(n) * 0.0001
    r = rng.multivariate_normal(np.full(n, 0.0005), cov, size=len(idx))
    return pd.DataFrame(r, index=idx, columns=[f"A{i}" for i in range(n)])


# ── Black-Litterman ───────────────────────────────────────────────────────────

def test_bl_no_views_returns_prior(returns_and_market):
    returns, mkt = returns_and_market
    cov = returns.cov().values
    mu_bl, cov_bl = black_litterman_posterior(mkt, cov, tau=0.05)
    pi = implied_equilibrium_returns(mkt, cov)
    np.testing.assert_allclose(mu_bl.values, pi.values, atol=1e-10)


def test_bl_portfolio_weights_valid(returns_and_market):
    returns, mkt = returns_and_market
    w = black_litterman_portfolio(returns, market_weights=mkt)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-8).all()


def test_bl_views_move_posterior(returns_and_market):
    returns, mkt = returns_and_market
    cov = returns.cov().values
    P = pd.DataFrame([[1, -1, 0, 0, 0]], columns=returns.columns)
    Q = pd.Series([0.0002])
    mu_bl, _ = black_litterman_posterior(mkt, cov, views=P, view_returns=Q)
    pi = implied_equilibrium_returns(mkt, cov)
    prior_spread = float(pi["A0"] - pi["A1"])
    posterior_spread = float(mu_bl["A0"] - mu_bl["A1"])
    view_spread = float(Q.iloc[0])
    assert abs(posterior_spread - view_spread) < abs(prior_spread - view_spread)


# ── Robust MV ────────────────────────────────────────────────────────────────

def test_robust_mv_weights_valid(returns_and_market):
    returns, _ = returns_and_market
    w = robust_max_sharpe_portfolio(returns, kappa=1.0)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-8).all()


def test_robust_mv_kappa_zero_matches_markowitz(returns_and_market):
    """κ=0 should collapse to standard mean-variance (in annualised units)."""
    returns, _ = returns_and_market
    periods = 252
    w_robust = robust_max_sharpe_portfolio(returns, kappa=0.0, risk_aversion=1.0,
                                           periods_per_year=periods)
    import cvxpy as cp
    from spo.optim.mean_variance import estimate_covariance
    cov_ann = estimate_covariance(returns, method="ledoit_wolf") * periods
    mu_ann = returns.mean().values * periods
    n = len(mu_ann)
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Maximize(mu_ann @ w - 0.5 * cp.quad_form(w, cp.psd_wrap(cov_ann))),
        [cp.sum(w) == 1, w >= 0],
    )
    prob.solve()
    np.testing.assert_allclose(w_robust.values, w.value, atol=1e-4)


def test_robust_mv_higher_kappa_is_more_defensive(returns_and_market):
    """As κ grows, portfolio tilts toward min-var (lower expected return)."""
    returns, _ = returns_and_market
    w_low = robust_max_sharpe_portfolio(returns, kappa=0.1)
    w_high = robust_max_sharpe_portfolio(returns, kappa=10.0)
    mu = returns.mean()
    assert mu @ w_high < mu @ w_low + 1e-4


def test_robust_mv_vol_bounded_by_min_var(daily_returns):
    """Robust MV with reasonable kappa should not have wildly higher vol than min-var."""
    w_mv = min_var_portfolio(daily_returns).values
    w_r = robust_max_sharpe_portfolio(daily_returns, kappa=1.0).values
    daily_cov = daily_returns.cov().values
    var_mv = w_mv @ daily_cov @ w_mv
    var_r = w_r @ daily_cov @ w_r
    assert var_r < 4.0 * var_mv, (
        f"Robust MV variance {var_r:.6f} >> Min-Var {var_mv:.6f}"
    )


def test_robust_mv_weights_sum_to_one(daily_returns):
    w = robust_max_sharpe_portfolio(daily_returns, kappa=1.0)
    assert np.isclose(w.sum(), 1.0)


def test_robust_mv_respects_max_weight(daily_returns):
    cap = 0.10
    w = robust_max_sharpe_portfolio(daily_returns, kappa=1.0, max_weight=cap)
    assert (w <= cap + 1e-6).all()


def test_robust_mv_no_cap_allows_concentration(daily_returns):
    """
    Without a weight cap, weights may concentrate beyond the capped level.
    """
    w_capped = robust_max_sharpe_portfolio(daily_returns, kappa=0.0, max_weight=0.10)
    w_free = robust_max_sharpe_portfolio(daily_returns, kappa=0.0, max_weight=None)
    assert w_free.max() >= w_capped.max() - 1e-6
