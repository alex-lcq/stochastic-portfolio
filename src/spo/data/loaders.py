"""
Price and universe loaders
"""
from __future__ import annotations
import io
import logging
import urllib.request
import zipfile
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


def fetch_vix(start: str, end: str) -> pd.Series:
    """
    Fetch the CBOE VIX index level from yfinance.
    """
    import yfinance as yf

    data = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    vix = data["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    return vix.rename("VIX").sort_index()


_FF_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{dataset}_CSV.zip"

_FF_COLUMNS = {
    "3": ["Mkt-RF", "SMB", "HML", "RF"],
    "5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
}


def fetch_ff_factors(start: str, end: str, model: str = "5") -> pd.DataFrame:
    """
    Fetch daily Fama-French factors directly from Ken French's data library
    (bypassing pandas-datareader's famafrench reader, which is currently
    broken against pandas>=2.2 -- pandas-datareader#933). model="3": Mkt-RF,
    SMB, HML, RF. model="5": adds RMW, CMA. Values are converted from
    percent to decimal.
    """
    dataset = {
        "3": "F-F_Research_Data_Factors_daily",
        "5": "F-F_Research_Data_5_Factors_2x3_daily",
    }[model]
    columns = _FF_COLUMNS[model]

    with urllib.request.urlopen(_FF_BASE_URL.format(dataset=dataset)) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")

    # Data rows are "YYYYMMDD,val,val,...". Header/footer lines aren't.
    rows = [
        [v.strip() for v in line.split(",")]
        for line in text.splitlines()
        if line[:8].strip().isdigit() and len(line[:8].strip()) == 8
    ]
    ff = pd.DataFrame(rows, columns=["date", *columns])
    ff["date"] = pd.to_datetime(ff["date"], format="%Y%m%d")
    ff = ff.set_index("date").astype(float) / 100.0
    return ff.sort_index().loc[start:end]