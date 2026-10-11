"""
metrics.py – Evaluation metrics and result plotting for time-series forecasts.

Only MAE and RMSE are computed — both are in the original price units after
inverse-scaling, making them directly interpretable.

Disclaimer
----------
Low MAE/RMSE does NOT imply a profitable trading strategy. These metrics
measure how closely the model tracks historical prices, not future alpha.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend for server/test environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


# --------------------------------------------------------------------------- #
#  Metric functions                                                             #
# --------------------------------------------------------------------------- #


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error — average magnitude of errors in price units."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Root Mean Squared Error — penalises large errors more than MAE.
    Still in the same units as the target (e.g. dollars).
    """
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return a dict with both MAE and RMSE."""
    return {"MAE": mae(actual, predicted), "RMSE": rmse(actual, predicted)}


# --------------------------------------------------------------------------- #
#  Plotting                                                                     #
# --------------------------------------------------------------------------- #


def plot_predictions(
    test_index: pd.DatetimeIndex,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    target_col: str = "Close",
    title: str = "Actual vs Predicted",
) -> plt.Figure:
    """
    Plot actual vs predicted prices for one or more models.

    Args:
        test_index:   DatetimeIndex for the x-axis.
        actual:       1-D array of true values (original scale).
        predictions:  Dict mapping model name → 1-D array of predictions.
        target_col:   Column name (used for y-axis label).
        title:        Chart title.

    Returns:
        matplotlib Figure object (can be displayed or saved by the caller).
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_index, actual, label="Actual", color="black", linewidth=1.5, zorder=5)

    colors = ["steelblue", "tomato", "seagreen", "orange", "purple"]
    for i, (model_name, preds) in enumerate(predictions.items()):
        ax.plot(
            test_index,
            preds,
            label=model_name,
            color=colors[i % len(colors)],
            linewidth=1.2,
            linestyle="--",
            alpha=0.85,
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{target_col} (original scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_metrics_bar(metrics: dict[str, dict[str, float]]) -> plt.Figure:
    """
    Bar chart comparing MAE and RMSE across models.

    Args:
        metrics: dict of {model_name: {"MAE": ..., "RMSE": ...}}
    """
    model_names = list(metrics.keys())
    mae_vals  = [metrics[m]["MAE"]  for m in model_names]
    rmse_vals = [metrics[m]["RMSE"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width / 2, mae_vals,  width, label="MAE",  color="steelblue")
    bars2 = ax.bar(x + width / 2, rmse_vals, width, label="RMSE", color="tomato")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Error (price units)")
    ax.set_title("Model Comparison – MAE & RMSE")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bar heights
    for bar in (*bars1, *bars2):
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    return fig


def plot_loss_curve(losses: list[float], model_name: str = "LSTM") -> plt.Figure:
    """Plot training loss over epochs for the LSTM."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(1, len(losses) + 1), losses, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (scaled)")
    ax.set_title(f"{model_name} – Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
