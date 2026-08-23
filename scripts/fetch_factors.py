"""
Fetch VIX and Fama-French factor data for the alpha-decomposition analysis (Week 7),
caching to data/processed as parquet to avoid re-hitting yfinance/pandas_datareader
on every notebook run.

Usage:
    python -m scripts.fetch_factors
    python -m scripts.fetch_factors --start-date 2015-01-01 --end-date 2025-12-31 --ff-model 5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spo.data import fetch_vix, fetch_ff_factors

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("fetch_factors")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch VIX and Fama-French factor data.")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--ff-model", default="5", choices=["3", "5"])
    args = parser.parse_args()

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching VIX from {args.start_date} to {args.end_date}")
    vix = fetch_vix(args.start_date, args.end_date)
    vix.to_frame().to_parquet(raw_dir / "vix_raw.parquet")
    vix.to_frame().to_parquet(processed_dir / "vix.parquet")
    logger.info(f"VIX: {len(vix)} obs, {vix.index.min().date()} to {vix.index.max().date()}")

    logger.info(f"Fetching Fama-French {args.ff_model}-factor daily data")
    ff = fetch_ff_factors(args.start_date, args.end_date, model=args.ff_model)
    ff.to_parquet(raw_dir / f"ff{args.ff_model}_factors_raw.parquet")
    ff.to_parquet(processed_dir / f"ff{args.ff_model}_factors.parquet")
    logger.info(f"FF{args.ff_model}: {len(ff)} obs, {ff.index.min().date()} to {ff.index.max().date()}")
    logger.info(f"Columns: {list(ff.columns)}")


if __name__ == "__main__":
    main()
