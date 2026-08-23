"""
Run the full backtest experiments from a config file.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --config config/experiments.yaml
    python scripts/run_experiment.py --universe sp500 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
import yaml

from spo.backtest import get_strategy, summary, backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_experiment")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiments.yaml")
    parser.add_argument("--universe", default=None, help="Override universe from config")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    defaults = cfg["default"]
    universe = args.universe or defaults["universe"]
    start = args.start or defaults["start"]
    end = args.end or defaults["end"]

    # Load the log-returns panel
    returns_path = Path("data/processed") / f"{universe}_log_returns.parquet"
    if not returns_path.exists():
        raise FileNotFoundError(
            f"{returns_path} not found. Run scripts/fetch_data.py first."
        )
    returns = pd.read_parquet(returns_path)
    returns = returns.loc[start:end]
    log.info("Loaded %s: %d days x %d assets", universe, *returns.shape)

    out_dir = Path(args.out) / universe
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    net_returns_by_strategy = {}
    for spec in cfg["strategies"]:
        name = spec["name"]
        label = spec.get("label", name)
        params = spec.get("params", {})
        log.info("Running strategy: %s (%s)", label, name)

        strategy_fn = get_strategy(name, **params)
        result = backtest(
            returns, strategy_fn,
            rebalance_freq=defaults["rebalance_freq"],
            lookback=defaults["lookback_days"],
            cost=defaults["cost_bps"]
        )
        all_results[label] = result
        net_returns_by_strategy[label] = result["net_returns"]

        s = summary(result["net_returns"])
        log.info("Ann.Ret=%.2f%%  Ann.Vol=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%",
                s["Ann. Returns"]*100, s["Ann. Vol"]*100,
                s["Sharpe"], s["Max DD"]*100)
    
    # Save pickled full results
    with open(out_dir / "backtest_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    pd.DataFrame(net_returns_by_strategy).to_csv(out_dir / "net_returns.csv")
    summary_df = pd.concat(
        {label: summary(r) for label, r in net_returns_by_strategy.items()},
        axis=1,
    )
    summary_df.to_csv(out_dir / "summary.csv")

    print("\n=== Summary ===")
    print(summary_df.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
