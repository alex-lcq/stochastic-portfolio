"""
Robust mean-variance: worst-case optimization over an ellipsoidal uncertainty set for the mean vector.

Reference: Ceria & Stubbs (2006), "Incorporating estimation errors into
portfolio selection: Robust portfolio construction."
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from spo.optim.mean_variance import estimate_covariance


def robust_max_sharpe_portfolio(
        returns: pd.DataFrame,
        kappa: float = 1.0,
        cov_method: str = "ledoit_wolf",
        long_only: bool = True,
        risk_aversion: float = 2.5,
        periods_per_year: int = 252,
        max_weight: float | None = None,
) -> pd.Series:
    """
    Robust Markowitz with worst-case mean penalty.
    Solves (in annualized units):
        max  μ'w - κ √(w'Σ_μ w) - (λ/2) w'Σ w
        s.t. w' 1 = 1, w >= 0 (if long_only), w <= max_weight (if set)

    Inputs are scaled to annual units internally so that risk_aversion=2.5
    (the standard equity-market value) gives sensible results regardless of
    the return frequency. κ controls the uncertainty-ellipsoid radius
    (κ=0 collapses to standard Markowitz, larger κ is more defensive).

    Calibration note: with daily data and a ~2-year lookback (n_obs≈504),
    the uncertainty covariance is Σ_ann/2, so the break-even κ between
    "concentrated" and "diversified" is roughly 1.2–2. Use κ≥3 to see a
    meaningful shift toward min-variance; κ≈10 gives near-min-var behaviour.
    """
    returns = returns.dropna(axis=1)
    n_obs, n_assets = returns.shape
    tickers = list(returns.columns)

    mu_hat = returns.mean().values * periods_per_year
    cov = estimate_covariance(returns, method=cov_method) * periods_per_year
    # Correct estimation-error covariance for the annualised mean:
    # Var(μ_ann) = T² × Var(μ_daily) = T × Σ_ann / n_obs
    # (plain cov/n_obs understates the penalty by √T when using daily data)
    cov_mu = cov * periods_per_year / n_obs

    sqrt_cov_mu = np.linalg.cholesky(cov_mu + 1e-10 * np.eye(n_assets))

    w = cp.Variable(n_assets)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    if max_weight is not None:
        constraints.append(w <= max_weight)

    prob = cp.Problem(
        cp.Maximize(
            mu_hat @ w
            - kappa * cp.norm(sqrt_cov_mu.T @ w, 2)
            - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))
        ),
        constraints,
    )
    prob.solve()
    if w.value is None:
        raise RuntimeError(f"Robust MV solve failed: {prob.status}")
    return pd.Series(w.value, index=tickers)
