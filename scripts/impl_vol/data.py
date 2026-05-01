"""
Fetch SPY option chains from Yahoo Finance via yfinance.

Results are cached to cache/spy_{type}_{date}.pkl so that repeated runs
with the same reference date are reproducible without re-fetching.
Delete the cache file to force a fresh download.
"""

from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd
import yfinance as yf


_CACHE_DIR = Path(__file__).parent / "cache"


def _spy_dividend_yield(ticker: yf.Ticker) -> float:
    """Return SPY forward dividend yield. Raises if the key is missing."""
    val = ticker.info.get("dividendYield")
    if not val:
        raise ValueError("SPY dividend yield unavailable via yfinance (key: dividendYield)")
    return float(val) / 100  # yfinance returns percentage form (e.g., 1.14 → 0.0114)


def fetch_spy_options(
    reference_date: date,
    r_curve: Callable[[float], float],
    max_T: float = 1.0,
    min_T: float = 4 / 52,
    min_volume: int = 10,
    option_type: str = "call",
) -> pd.DataFrame:
    """
    Fetch SPY option data for all expirations within [min_T, max_T] years.

    reference_date: date used to label and cache the snapshot.
    r_curve:        T -> continuously-compounded risk-free rate (from rates.py).
    option_type:    "call" or "put".

    Note: yfinance only provides current option chains; the reference_date
    controls the cache filename so the same run is reusable, not historical.
    """
    cache_path = _CACHE_DIR / f"spy_{option_type}_{reference_date}.parquet"
    if cache_path.exists():
        print(f"  Loading {option_type}s from cache ({cache_path.name})")
        return pd.read_parquet(cache_path)

    ticker = yf.Ticker("SPY")
    hist = ticker.history(period="1d")
    S = float(hist["Close"].iloc[-1])
    q = _spy_dividend_yield(ticker)
    today = date.today()

    records = []
    for exp_str in ticker.options:
        exp_date = date.fromisoformat(exp_str)
        T = (exp_date - today).days / 365.0
        if T < min_T or T > max_T:
            continue

        chain = ticker.option_chain(exp_str)
        opts = chain.calls.copy() if option_type == "call" else chain.puts.copy()
        opts = opts[opts["volume"] >= min_volume]
        opts["openInterest"] = opts["openInterest"].fillna(0).astype(int)

        has_quote = (opts["bid"] > 0) & (opts["ask"] > 0)
        opts["mid"] = (
            has_quote * (opts["bid"] + opts["ask"]) / 2.0
            + (~has_quote) * opts["lastPrice"]
        )
        opts = opts[opts["mid"] > 0]

        r = r_curve(T)
        for _, row in opts.iterrows():
            records.append({
                "K":            float(row["strike"]),
                "T":            T,
                "mid":          float(row["mid"]),
                "openInterest": int(row["openInterest"]),
                "S":            S,
                "r":            r,
                "q":            q,
                "expiration":   exp_str,
                "option_type":  option_type,
            })

    df = pd.DataFrame(records)
    _CACHE_DIR.mkdir(exist_ok=True)
    df.to_parquet(cache_path)
    print(f"  Fetched {len(df)} {option_type} quotes  (q={q:.2%}, cached)")
    return df
