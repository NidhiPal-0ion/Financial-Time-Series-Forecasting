"""
Generate a synthetic sample financial dataset for offline use.
Simulates daily closing prices using a geometric Brownian motion model.
Run this script once to (re)create data/sample_prices.csv.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_sample(
    start: str = "2018-01-01",
    end: str = "2023-12-31",
    ticker: str = "SAMPLE",
    initial_price: float = 100.0,
    mu: float = 0.0002,       # daily drift
    sigma: float = 0.012,      # daily volatility
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame of daily OHLCV-like data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)  # business days only
    n = len(dates)

    # Geometric Brownian Motion for close prices
    log_returns = rng.normal(mu - 0.5 * sigma ** 2, sigma, size=n)
    close = initial_price * np.exp(np.cumsum(log_returns))

    # Derive simple OHLV from close
    daily_range = rng.uniform(0.005, 0.02, size=n) * close
    open_  = close - rng.uniform(-0.5, 0.5, size=n) * daily_range
    high   = np.maximum(close, open_) + rng.uniform(0, 0.5, size=n) * daily_range
    low    = np.minimum(close, open_) - rng.uniform(0, 0.5, size=n) * daily_range
    volume = rng.integers(500_000, 5_000_000, size=n).astype(float)

    df = pd.DataFrame({
        "Date": dates,
        "Open": np.round(open_, 2),
        "High": np.round(high, 2),
        "Low": np.round(low, 2),
        "Close": np.round(close, 2),
        "Volume": volume,
        "Ticker": ticker,
    })
    return df


if __name__ == "__main__":
    out_path = Path(__file__).parent / "sample_prices.csv"
    df = generate_sample()
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.head())
