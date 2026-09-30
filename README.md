# Stochastic Portfolio Optimization

Research codebase for building and walk-forward backtesting equity portfolio strategies, comparing classical mean-variance approaches (Ledoit-Wolf min-variance, Black-Litterman, robust mean-variance) against stochastic-model-based CVaR optimization (GBM and Heston Monte Carlo scenarios). Includes an alpha-decomposition module that attributes the outperformance of one strategy over another to volatility-timing and tail-shape factors, controlling for Fama-French.

## Overview

The project walks through a full quant research pipeline on the S&P 500 (2015–2025, daily data), also configurable for the CAC 40:

1. **Data** — fetch, clean and align adjusted daily prices from Yahoo Finance, with manual review of return anomalies (splits, spin-offs).
2. **Classical baselines** — equal-weight, Ledoit-Wolf minimum-variance, Black-Litterman, and robust mean-variance (ellipsoidal uncertainty on the mean).
3. **Stochastic models** — multivariate Geometric Brownian Motion and Heston (stochastic volatility, calibrated via indirect inference/GMM) are simulated forward and turned into Monte Carlo scenarios for CVaR portfolio optimization (Rockafellar-Uryasev LP).
4. **Backtesting** — a walk-forward engine rebalances monthly, applies transaction costs, and reports Sharpe/Sortino/Calmar/PSR/DSR performance metrics.
5. **Alpha decomposition** — regresses the daily outperformance of the best stochastic strategy over the best classical one on a vol-timing factor (ΔVIX), a pooled tail-shape (kurtosis) factor, and the Fama-French 5 factors, with HAC standard errors and sequential R² attribution.

## Project structure

```
config/                 YAML config: universes (tickers), experiment defaults, reviewed return anomalies
data/
  raw/                  Unmodified data pulled from source APIs
  processed/            Cleaned prices, log/simple returns, VIX, Fama-French factors
notebooks/              Sequential research notebooks (01 -> 07, see below)
scripts/
  populate_tickers.py   Scrape current index constituents (Wikipedia) into config/universes.yaml
  fetch_data.py         Fetch + clean prices, compute returns, apply reviewed anomalies
  fetch_factors.py      Fetch VIX and Fama-French factors
  run_experiment.py     Run the full multi-strategy walk-forward backtest from a config file
src/spo/
  data/                 Fetching, cleaning/alignment, universe & anomaly loaders
  models/               GBM and Heston calibration + Monte Carlo simulation
  optim/                Mean-variance, Black-Litterman, robust MV, CVaR (Rockafellar-Uryasev)
  risk/                 VaR/CVaR utilities, alpha decomposition & factor regression
  backtest/             Walk-forward engine, strategy registry, performance metrics
results/                Saved backtest outputs (pickled results, net returns, summary stats)
tests/                  pytest unit tests for each module
```

### Notebooks

| Notebook | Content |
|---|---|
| `01_Data_Analysis` | Data exploration, cleaning, anomaly detection |
| `02_Mean_Var_Baseline` | Classical Markowitz / min-variance / efficient frontier |
| `03_GBM` | GBM calibration and Monte Carlo simulation |
| `04_Heston` | Heston calibration (GMM/indirect inference), Feller condition checks |
| `05_CVaR_vs_MV` | CVaR-optimized portfolios vs. mean-variance |
| `06_Strategies_Backtest` | Full walk-forward comparison of all strategies |
| `07_Alpha_Decomposition` | Vol-timing / tail-shape / Fama-French attribution of strategy alpha |

## Strategies

Registered in [src/spo/backtest/strategies.py](src/spo/backtest/strategies.py) and selectable by name in `config/experiments.yaml`:

| Name | Description |
|---|---|
| `equal_weight` | 1/N baseline |
| `min_variance` | Minimum-variance with Ledoit-Wolf shrinkage (or sample / hierarchical-clustering cleaned covariance) |
| `black_litterman` | Black-Litterman posterior returns (He-Litterman formulation) fed into a mean-variance solve |
| `robust_mv` | Robust max-Sharpe under an ellipsoidal uncertainty set on the mean (Ceria & Stubbs) |
| `gbm_cvar` | Multivariate GBM Monte Carlo scenarios, CVaR-minimized |
| `heston_cvar` | Multivariate Heston (stochastic vol) Monte Carlo scenarios, CVaR-minimized |

All optimizers are solved with `cvxpy` and support long-only and max-weight constraints.

## Getting started

### Installation

Requires Python >= 3.11.

```bash
pip install -e ".[dev]"
```

### Data pipeline

```bash
# Optional: refresh index constituents from Wikipedia
python -m scripts.populate_tickers --universe sp500

# Fetch, clean and compute returns for a universe
python -m scripts.fetch_data --universe sp500 --start-date 2015-01-01 --end-date 2025-12-31

# Fetch VIX and Fama-French factors (needed for notebook 07)
python -m scripts.fetch_factors --start-date 2015-01-01 --end-date 2025-12-31
```

### Run the backtest

```bash
python scripts/run_experiment.py --config config/experiments.yaml
```

Strategy list, lookback window, rebalance frequency and transaction costs are set in [config/experiments.yaml](config/experiments.yaml). Results (pickled full output, net returns, summary stats) are written to `results/<universe>/`.

### Tests

```bash
pytest
```

## Sample results

S&P 500, 2015–2025, monthly rebalance, 504-day lookback, 10 bps cost (from `results/sp500/summary.csv`):

| Metric | Equal-Weight | Min-Variance | Black-Litterman | Robust-MV | GBM-CVaR | Heston-CVaR |
|---|---|---|---|---|---|---|
| Ann. Return | 9.30% | 6.51% | 8.66% | 9.46% | 7.30% | 6.67% |
| Ann. Vol | 17.17% | 12.60% | 16.30% | 22.90% | 14.03% | 13.76% |
| Sharpe | 0.60 | 0.56 | 0.59 | 0.51 | 0.57 | 0.54 |
| Sortino | 0.66 | 0.60 | 0.64 | 0.58 | 0.62 | 0.58 |
| Max DD | -41.0% | -35.0% | -40.5% | -41.9% | -35.4% | -35.3% |
| Calmar | 0.23 | 0.19 | 0.21 | 0.23 | 0.21 | 0.19 |

Exact figures will vary by config: see notebook 06 for the full walk-forward comparison and notebook 07 for factor attribution of strategy alpha.

## Key methodology notes

- **Covariance estimation** defaults to Ledoit-Wolf shrinkage throughout, with an optional hierarchical-clustering block-averaging cleaner.
- **Heston calibration** uses per-asset indirect inference (GMM on empirical moments, simulated-kurtosis matching via Brent's method), with `rho` and `kappa` fixed from literature priors since they are not identifiable from daily returns alone; correlation across assets comes from a Ledoit-Wolf return-correlation matrix.
- **CVaR optimization** uses the Rockafellar-Uryasev linear program on Monte Carlo horizon scenarios, minimizing CVaR directly or maximizing return under a CVaR budget (efficient frontier).
- **Performance metrics** include the Probabilistic and Deflated Sharpe Ratio (Bailey & López de Prado) to account for non-normal returns and multiple-testing bias across strategies.
- **Alpha decomposition** uses Newey-West HAC standard errors (TS is a rolling-window construction, alpha exhibits vol clustering) and sequential (Type-I) R² so factor-group contributions sum exactly to total R².
