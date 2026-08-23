"""
Alpha decomposition: attributing the daily outperformance of one strategy over
another (e.g. Heston-CVaR vs Min-Variance) to a vol-timing factor, a
tail-shape factor, and Fama-French control factors.

Methodology: alpha_t = beta_0 + beta_VT*VT_t + beta_TS*TS_t + beta_FF.FF_t + eps_t,
following the Treynor & Mazuy (1966) market-timing regression framework applied
to the CVaR-vs-MV alpha residual.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor

FF_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

DEFAULT_GROUPS = {
    "VT": ["VT"],
    "TS": ["TS"],
    "FF": FF_COLS,
}

DEFAULT_PERIODS = {
    "Full sample": (None, None),
    "2020 (COVID)": ("2020-01-01", "2020-12-31"),
    "2022 (rate hikes)": ("2022-01-01", "2022-12-31"),
}


def compute_alpha(
    net_returns: pd.DataFrame,
    strategy_a: str = "Heston-CVaR",
    strategy_b: str = "Min-Variance",
) -> pd.Series:
    """
    Daily outperformance series alpha_t = r_a,t - r_b,t between two backtested
    strategies' net-of-cost return columns.
    """
    alpha = (net_returns[strategy_a] - net_returns[strategy_b]).dropna()
    return alpha.rename("alpha")


def vol_timing_factor(vix: pd.Series) -> pd.Series:
    """
    VT_t = daily change in the VIX level. A genuine daily-frequency proxy for
    innovations in expected forward volatility, independent of the backtest's
    monthly rebalance cadence.
    """
    return vix.diff().rename("VT")


def tail_shape_factor(returns_panel: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    TS_t = excess kurtosis pooled across all assets over the trailing `window`
    days. Deliberately cross-sectional-over-time (pooling ~N_assets x window
    observations per day) rather than each asset's own rolling kurtosis, which
    would be far noisier at daily frequency.

    Uses an explicit loop over dates (not .rolling().apply(), which operates
    per-column) since pooling across columns needs the full window slice at
    once. Fine at this scale (~2,500 dates x a few thousand pooled values).
    """
    values = returns_panel.values
    n = len(returns_panel)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        pooled = values[t - window + 1: t + 1, :].ravel()
        pooled = pooled[~np.isnan(pooled)]
        out[t] = stats.kurtosis(pooled, fisher=True, bias=False)
    return pd.Series(out, index=returns_panel.index, name="TS")


def build_factor_frame(
    alpha: pd.Series,
    vt: pd.Series,
    ts: pd.Series,
    ff: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Align alpha, VT, TS, and FF factors on a common date index and drop rows
    with missing values (covers TS's rolling warm-up window and any
    Fama-French publication-lag gaps). RF is dropped -- it's a rate, not a
    return-driving regressor.
    """
    ff_factors = ff.drop(columns=["RF"], errors="ignore")
    combined = pd.concat({"alpha": alpha, "VT": vt, "TS": ts}, axis=1)
    combined = combined.join(ff_factors, how="inner").dropna(how="any")
    return combined["alpha"], combined.drop(columns=["alpha"])


def run_decomposition(
    alpha: pd.Series,
    factors: pd.DataFrame,
    hac_lags: int | None = None,
) -> RegressionResultsWrapper:
    """
    OLS regression of alpha on vol-timing, tail-shape, and Fama-French
    factors, with Newey-West (1987) HAC standard errors.

    Classical OLS SEs assume i.i.d. residuals, but TS is a rolling-window
    construction (mechanically autocorrelated) and daily alpha exhibits
    volatility clustering -- both violate that assumption and would
    understate significance under plain OLS.
    """
    n = len(alpha)
    if hac_lags is None:
        hac_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    X = sm.add_constant(factors)
    return sm.OLS(alpha, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def regression_table(model: RegressionResultsWrapper) -> pd.DataFrame:
    """
    Tidy coef / t-stat / p-value table. r_squared and n_obs are stashed on
    .attrs for display alongside the table.
    """
    table = pd.DataFrame({
        "coef": model.params,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
    })
    table.attrs["r_squared"] = model.rsquared
    table.attrs["n_obs"] = int(model.nobs)
    return table


def r2_decomposition(
    alpha: pd.Series,
    factors: pd.DataFrame,
    groups: dict[str, list[str]] | None = None,
) -> pd.Series:
    """
    Sequential (Type-I) R^2: nested OLS fits adding one factor group at a
    time, in the given order (default VT -> TS -> FF), so the increments sum
    exactly to total R^2 -- unlike naive incremental R^2, which can double
    count when regressors correlate.

    This is order-dependent when regressors correlate (VT and TS both spike
    in crises). See variance_inflation_factors for when to distrust the
    ordering.
    """
    groups = groups or DEFAULT_GROUPS

    def _r2(cols: list[str]) -> float:
        if not cols:
            return 0.0
        X = sm.add_constant(factors[cols])
        return float(sm.OLS(alpha, X).fit().rsquared)

    cumulative_cols: list[str] = []
    prev_r2 = 0.0
    increments = {}
    for name, cols in groups.items():
        cumulative_cols = cumulative_cols + cols
        r2 = _r2(cumulative_cols)
        increments[name] = r2 - prev_r2
        prev_r2 = r2

    increments["Residual"] = 1.0 - prev_r2
    return pd.Series(increments)


def cumulative_contribution(
    model: RegressionResultsWrapper,
    factors: pd.DataFrame,
    alpha: pd.Series,
    period: tuple[str | None, str | None] | None = None,
) -> pd.DataFrame:
    """
    Attribute cumulative alpha over `period` to each factor's beta *
    factor_t, plus an Intercept bucket (pure unexplained alpha) and a
    Residual bucket.

    Note: sum(residuals) is exactly 0 over the *fitting* sample for any OLS
    fit with an intercept (the normal equations force it) -- so Residual will
    be ~0 when period=None (the full fitting sample). It is generically
    nonzero over a sub-period, since a sub-slice's own residuals need not sum
    to zero even though the full sample's do. That's not a bug: it's what
    makes the sub-period table informative.
    """
    idx = factors.index
    if period is not None:
        start, end = period
        idx = factors.loc[start:end].index

    beta = model.params.drop("const")
    contributions = (factors.loc[idx, beta.index] * beta).sum(axis=0)
    intercept = model.params["const"] * len(idx)
    residual = model.resid.loc[idx].sum()

    out = contributions.copy()
    out["Intercept"] = intercept
    out["Residual"] = residual

    total_alpha = alpha.loc[idx].sum()
    table = pd.DataFrame({"contribution": out})
    table["pct_of_total_alpha"] = table["contribution"] / total_alpha
    return table


def variance_inflation_factors(
    factors: pd.DataFrame,
    cols: list[str] | None = None,
) -> pd.Series:
    """
    VIF per regressor. VT and TS both spike in the same crisis windows, so
    they're the main multicollinearity risk in this regression -- VIF > 5-10
    is the conventional concern threshold.
    """
    cols = cols or list(factors.columns)
    X = sm.add_constant(factors[cols].dropna())
    vifs = {
        c: variance_inflation_factor(X.values, X.columns.get_loc(c))
        for c in cols
    }
    return pd.Series(vifs)


def orthogonalize(target: pd.Series, against: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Gram-Schmidt residualization: regress `target` on `against` (+
    intercept) and return the residuals. Used to strip the VT-TS correlation
    out of TS as a robustness check on the decomposition.
    """
    against_df = against.to_frame() if isinstance(against, pd.Series) else against
    aligned = pd.concat([target, against_df], axis=1).dropna()
    y = aligned.iloc[:, 0]
    X = sm.add_constant(aligned.iloc[:, 1:])
    resid = sm.OLS(y, X).fit().resid
    return resid.rename(f"{target.name}_orth")


def subperiod_contribution_table(
    model: RegressionResultsWrapper,
    factors: pd.DataFrame,
    alpha: pd.Series,
    periods: dict[str, tuple[str | None, str | None]] | None = None,
) -> pd.DataFrame:
    """
    pct_of_total_alpha per factor, side by side across named sub-periods
    (default: full sample, 2020, 2022).
    """
    periods = periods or DEFAULT_PERIODS
    cols = {}
    for label, period in periods.items():
        table = cumulative_contribution(model, factors, alpha, period=period)
        cols[label] = table["pct_of_total_alpha"]
    return pd.DataFrame(cols)
