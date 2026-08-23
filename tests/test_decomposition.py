"""
Tests for alpha decomposition (VT/TS/FF regression, R^2 decomposition,
cumulative contribution attribution, VIF, orthogonalization).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spo.risk.decomposition import (
    build_factor_frame,
    cumulative_contribution,
    orthogonalize,
    r2_decomposition,
    run_decomposition,
    subperiod_contribution_table,
    tail_shape_factor,
    variance_inflation_factors,
)

FF_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def _synthetic_factors(n: int = 1000, seed: int = 0, corr_vt_ts: float = 0.0) -> pd.DataFrame:
    """
    Synthetic VT/TS/FF factor panel over a business-day index. corr_vt_ts
    controls how correlated TS is with VT, so tests can dial multicollinearity
    up or down.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    vt = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    ts = corr_vt_ts * vt + np.sqrt(max(1 - corr_vt_ts ** 2, 0.0)) * noise
    ff = rng.standard_normal((n, len(FF_COLS))) * 0.005
    df = pd.DataFrame(ff, index=idx, columns=FF_COLS)
    df.insert(0, "TS", ts)
    df.insert(0, "VT", vt)
    return df


def _synthetic_alpha(
    factors: pd.DataFrame,
    betas: dict[str, float],
    beta0: float = 0.0002,
    noise_std: float = 0.001,
    seed: int = 1,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    alpha = pd.Series(beta0, index=factors.index)
    for col, b in betas.items():
        alpha = alpha + b * factors[col]
    alpha = alpha + rng.normal(0, noise_std, len(factors))
    return alpha.rename("alpha")


def test_run_decomposition_recovers_true_betas():
    factors = _synthetic_factors(n=2000, corr_vt_ts=0.0)
    true_betas = {"VT": 0.01, "TS": 0.005}
    alpha = _synthetic_alpha(factors, true_betas, noise_std=0.0005)
    model = run_decomposition(alpha, factors)
    assert model.params["VT"] == pytest.approx(true_betas["VT"], abs=0.003)
    assert model.params["TS"] == pytest.approx(true_betas["TS"], abs=0.003)


def test_r2_decomposition_sums_to_one():
    factors = _synthetic_factors(n=1500, corr_vt_ts=0.2)
    alpha = _synthetic_alpha(factors, {"VT": 0.01, "TS": 0.005, "Mkt-RF": 0.3})
    shares = r2_decomposition(alpha, factors)
    assert shares.sum() == pytest.approx(1.0, abs=1e-8)


def test_cumulative_contribution_reconciles_full_sample():
    factors = _synthetic_factors(n=1000, corr_vt_ts=0.1)
    alpha = _synthetic_alpha(factors, {"VT": 0.01, "TS": 0.005})
    model = run_decomposition(alpha, factors)
    table = cumulative_contribution(model, factors, alpha)
    assert table["contribution"].sum() == pytest.approx(alpha.sum(), rel=1e-6)
    assert table.loc["Residual", "contribution"] == pytest.approx(0.0, abs=1e-6)


def test_cumulative_contribution_reconciles_subperiod():
    factors = _synthetic_factors(n=1000, corr_vt_ts=0.1)
    alpha = _synthetic_alpha(factors, {"VT": 0.01, "TS": 0.005})
    model = run_decomposition(alpha, factors)
    start, end = factors.index[100], factors.index[300]
    table = cumulative_contribution(
        model, factors, alpha, period=(str(start.date()), str(end.date()))
    )
    period_alpha = alpha.loc[start:end].sum()
    assert table["contribution"].sum() == pytest.approx(period_alpha, rel=1e-6)


def test_vif_high_when_regressors_correlated():
    correlated = _synthetic_factors(n=1000, corr_vt_ts=0.95)
    independent = _synthetic_factors(n=1000, corr_vt_ts=0.0)
    vif_corr = variance_inflation_factors(correlated, cols=["VT", "TS"])
    vif_indep = variance_inflation_factors(independent, cols=["VT", "TS"])
    assert vif_corr["VT"] > 5
    assert vif_corr["TS"] > 5
    assert vif_indep["VT"] < 1.5
    assert vif_indep["TS"] < 1.5


def test_orthogonalization_reduces_vif():
    factors = _synthetic_factors(n=1000, corr_vt_ts=0.95)
    ts_orth = orthogonalize(factors["TS"], factors["VT"])
    rebuilt = factors.drop(columns=["TS"]).assign(TS=ts_orth)
    vif_after = variance_inflation_factors(rebuilt, cols=["VT", "TS"])
    assert vif_after["VT"] < 1.5
    assert vif_after["TS"] < 1.5
    assert abs(np.corrcoef(factors["VT"], ts_orth)[0, 1]) < 1e-6


def test_tail_shape_factor_shape_and_direction():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=60)
    calm = rng.standard_normal((30, 20)) * 0.01
    fat_tail = rng.standard_t(df=2, size=(30, 20)) * 0.01
    panel = pd.DataFrame(
        np.vstack([calm, fat_tail]), index=idx, columns=[f"a{i}" for i in range(20)]
    )

    ts = tail_shape_factor(panel, window=21)
    assert len(ts) == len(panel)
    assert ts.iloc[:20].isna().all()
    assert ts.iloc[-1] > ts.iloc[29]


def test_subperiod_contribution_table_columns_sum_to_one():
    factors = _synthetic_factors(n=1000, corr_vt_ts=0.1)
    alpha = _synthetic_alpha(factors, {"VT": 0.01, "TS": 0.005})
    model = run_decomposition(alpha, factors)
    periods = {
        "Full sample": (None, None),
        "First half": (str(factors.index[0].date()), str(factors.index[499].date())),
    }
    table = subperiod_contribution_table(model, factors, alpha, periods=periods)
    for col in table.columns:
        assert table[col].sum() == pytest.approx(1.0, abs=1e-6)


def test_build_factor_frame_aligns_and_drops_rf():
    idx = pd.bdate_range("2015-01-01", periods=50)
    rng = np.random.default_rng(0)
    alpha = pd.Series(rng.standard_normal(50), index=idx, name="alpha")
    vt = pd.Series(rng.standard_normal(50), index=idx, name="VT")
    ts = pd.Series(rng.standard_normal(50), index=idx, name="TS")
    ff = pd.DataFrame(
        rng.standard_normal((50, 6)) * 0.005,
        index=idx, columns=FF_COLS + ["RF"],
    )
    ts.iloc[:5] = np.nan  # warm-up window

    aligned_alpha, aligned_factors = build_factor_frame(alpha, vt, ts, ff)
    assert "RF" not in aligned_factors.columns
    assert len(aligned_alpha) == len(aligned_factors) == 45
