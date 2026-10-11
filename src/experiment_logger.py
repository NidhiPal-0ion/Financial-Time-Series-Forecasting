"""
experiment_logger.py – Save and load experiment results to/from a local JSON file.

Each experiment run stores:
  - timestamp
  - model name
  - configuration (lookback, epochs, etc.)
  - evaluation metrics (MAE, RMSE)
  - dataset path used

No external services required — just a plain JSON file in the project root.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path(__file__).parent.parent / "experiments" / "history.json"


def log_experiment(
    model_name: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    dataset: str = "sample_prices.csv",
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> dict:
    """
    Append a single experiment record to the history file.

    Args:
        model_name: Human-readable model identifier.
        config:     Dict of hyperparameters / settings used.
        metrics:    Dict of metric name → value (MAE, RMSE …).
        dataset:    Name or path of the dataset used.
        log_path:   Path to the JSON history file.

    Returns:
        The newly added record dict.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "dataset": dataset,
        "config": config,
        "metrics": metrics,
    }

    # Load existing records or start fresh
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as fh:
            history = json.load(fh)
    else:
        history = []

    history.append(record)

    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    logger.info("Logged experiment: %s  MAE=%.4f  RMSE=%.4f",
                model_name, metrics.get("MAE", 0), metrics.get("RMSE", 0))
    return record


def load_history(log_path: Path | str = DEFAULT_LOG_PATH) -> pd.DataFrame:
    """
    Load the experiment history as a flat DataFrame.

    Returns an empty DataFrame if no history file exists yet.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return pd.DataFrame(columns=["timestamp", "model", "dataset", "MAE", "RMSE"])

    with open(log_path, "r", encoding="utf-8") as fh:
        history = json.load(fh)

    rows = []
    for rec in history:
        row = {
            "timestamp": rec.get("timestamp", ""),
            "model": rec.get("model", ""),
            "dataset": rec.get("dataset", ""),
            **rec.get("metrics", {}),
            **{f"cfg_{k}": v for k, v in rec.get("config", {}).items()},
        }
        rows.append(row)

    return pd.DataFrame(rows)


def clear_history(log_path: Path | str = DEFAULT_LOG_PATH) -> None:
    """Wipe the experiment history (used for testing)."""
    log_path = Path(log_path)
    if log_path.exists():
        log_path.unlink()
