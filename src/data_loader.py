"""
data_loader.py – Load, validate and preprocess financial time-series data.

Key design decisions
---------------------
* Dates are parsed and used as an ordered index — never shuffled.
* Preprocessing (scaling) is fit ONLY on training rows to prevent data leakage.
* Missing-value handling is explained so students can reason about it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Constants / defaults                                                         #
# --------------------------------------------------------------------------- #

SAMPLE_DATA_PATH = Path(__file__).parent.parent / "data" / "sample_prices.csv"
DEFAULT_TARGET_COL = "Close"


# --------------------------------------------------------------------------- #
#  1. Loading                                                                   #
# --------------------------------------------------------------------------- #


def load_csv(path: str | Path, date_col: str = "Date") -> pd.DataFrame:
    """Load a CSV file and return a DataFrame with a sorted DatetimeIndex."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, parse_dates=[date_col])

    # Sort chronologically — crucial for time-series work
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)
    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df


# --------------------------------------------------------------------------- #
#  2. Validation                                                                #
# --------------------------------------------------------------------------- #


def validate_dataframe(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL) -> list[str]:
    """
    Run basic sanity checks and return a list of warning messages.
    Raises ValueError for fatal issues.
    """
    warnings: list[str] = []

    # --- Date index ---
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex after load_csv().")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Dates are not in ascending order — check your data.")

    # --- Target column ---
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Available: {list(df.columns)}"
        )

    # --- Missing values ---
    missing = df[target_col].isna().sum()
    if missing > 0:
        warnings.append(
            f"Target column '{target_col}' has {missing} missing value(s). "
            "They will be forward-filled."
        )

    # --- Non-positive prices ---
    if (df[target_col] <= 0).any():
        warnings.append(
            f"Target column '{target_col}' contains non-positive values. "
            "Log-transformation may be unreliable."
        )

    # --- Duplicate dates ---
    dups = df.index.duplicated().sum()
    if dups:
        warnings.append(f"Found {dups} duplicate date(s). Only the last will be kept.")

    return warnings


# --------------------------------------------------------------------------- #
#  3. Cleaning                                                                  #
# --------------------------------------------------------------------------- #


def clean_dataframe(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COL) -> pd.DataFrame:
    """Apply light cleaning: drop duplicate dates, forward-fill missing values."""
    df = df[~df.index.duplicated(keep="last")]
    df[target_col] = df[target_col].ffill()
    return df


# --------------------------------------------------------------------------- #
#  4. Chronological train/test split                                            #
# --------------------------------------------------------------------------- #


def train_test_split_ts(
    series: pd.Series,
    test_fraction: float = 0.2,
) -> tuple[pd.Series, pd.Series]:
    """
    Split a time series into train and test portions WITHOUT shuffling.

    The split point is determined by the fraction of total observations so
    that no future data ever leaks into training.

    Args:
        series:        1-D time series (pd.Series with DatetimeIndex).
        test_fraction: Fraction of data reserved for testing (e.g. 0.2 = 20 %).

    Returns:
        (train_series, test_series)
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be strictly between 0 and 1.")

    split_idx = int(len(series) * (1 - test_fraction))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    logger.info(
        "Train: %d rows (%s – %s), Test: %d rows (%s – %s)",
        len(train), train.index[0].date(), train.index[-1].date(),
        len(test), test.index[0].date(), test.index[-1].date(),
    )
    return train, test


# --------------------------------------------------------------------------- #
#  5. Scaling (fit on train only)                                              #
# --------------------------------------------------------------------------- #


def fit_scaler(train: pd.Series) -> MinMaxScaler:
    """Fit a MinMaxScaler on training data only and return it."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train.values.reshape(-1, 1))
    return scaler


def scale(scaler: MinMaxScaler, series: pd.Series) -> np.ndarray:
    """Apply a pre-fitted scaler to a series and return a 1-D NumPy array."""
    return scaler.transform(series.values.reshape(-1, 1)).flatten()


def inverse_scale(scaler: MinMaxScaler, arr: np.ndarray) -> np.ndarray:
    """Invert the scaling transformation to recover original price units."""
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


# --------------------------------------------------------------------------- #
#  6. Lag-feature builder (for Linear Regression)                              #
# --------------------------------------------------------------------------- #


def make_lag_features(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised learning features from a 1-D array using lagged values.

    For each time step t (starting at `lookback`), the input row is
    [series[t-lookback], ..., series[t-1]] and the target is series[t].

    This construction ensures no future information is used: at prediction
    time for step t, we only look at steps up to t-1.

    Args:
        series:   1-D array of (scaled) values.
        lookback: Number of past time steps to use as features.

    Returns:
        X: shape (n_samples, lookback)
        y: shape (n_samples,)
    """
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback : i])
        y.append(series[i])
    return np.array(X), np.array(y)


# --------------------------------------------------------------------------- #
#  7. Convenience wrapper                                                       #
# --------------------------------------------------------------------------- #


def load_and_prepare(
    path: str | Path = SAMPLE_DATA_PATH,
    target_col: str = DEFAULT_TARGET_COL,
    test_fraction: float = 0.2,
) -> dict:
    """
    End-to-end helper that loads, validates, cleans, splits, and scales data.

    Returns a dict with keys:
        train_raw, test_raw   – original-scale pd.Series
        train_scaled, test_scaled – scaled np.ndarray
        scaler                – fitted MinMaxScaler
        target_col            – name of target column
        warnings              – list of validation warning strings
    """
    df = load_csv(path)
    warnings = validate_dataframe(df, target_col)
    df = clean_dataframe(df, target_col)

    series = df[target_col]
    train_raw, test_raw = train_test_split_ts(series, test_fraction)

    scaler = fit_scaler(train_raw)
    train_scaled = scale(scaler, train_raw)
    test_scaled = scale(scaler, test_raw)

    return {
        "train_raw": train_raw,
        "test_raw": test_raw,
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "scaler": scaler,
        "target_col": target_col,
        "warnings": warnings,
    }
