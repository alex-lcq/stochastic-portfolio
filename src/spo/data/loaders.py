"""
Price and universe loaders
"""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import yaml
from typing import Literal

logger = logging.getLogger(__name__)

SOURCE = Literal["yfinance"] # can add more sources in the future

def load_universe(name: str, config_path: str | Path="config/universes.yaml") -> dict:
    """
    Load a universe from a YAML configuration file
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if name not in config:
        available = ", ".join(config.keys())
        raise ValueError(f"Universe '{name}' not found in config. Available universes: {available}")
    return config[name]

def _fetch_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices data from yfinance
    """
    import yfinance as yf

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Normalize data to have a single level of columns with tickers as column names
    if isinstance(data.columns, pd.MultiIndex):
        close = pd.DataFrame(
            {ticker: data[ticker]["Close"] for ticker in tickers if ticker in data.columns.levels[0]}
        )
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def fetch_prices(tickers: list[str], start: str, end: str, source: SOURCE="yfinance") -> pd.DataFrame:
    """
    Fetch adjusted close prices data for a list of tickers between start and end dates
    """
    if source == "yfinance":
        prices = _fetch_yfinance(tickers, start, end)
    else:
        raise ValueError(f"Unsupported data source: {source}")
    
    return prices.sort_index()