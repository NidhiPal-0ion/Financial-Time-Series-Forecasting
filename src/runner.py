"""
runner.py – Programmatic (non-UI) pipeline for running all three models.

Use this for CLI runs, automated testing, or batch experiments.

Example
-------
    python -m src.runner --data data/sample_prices.csv --lookback 20 --epochs 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from src.data_loader import load_and_prepare, inverse_scale, make_lag_features
from src.experiment_logger import log_experiment
from src.metrics import compute_metrics, plot_predictions, plot_metrics_bar, plot_loss_curve
from src.models import NaiveModel, LinearModel, LSTMModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    data_path: str | Path = "data/sample_prices.csv",
    target_col: str = "Close",
    test_fraction: float = 0.2,
    lookback: int = 20,
    lstm_hidden: int = 64,
    lstm_epochs: int = 30,
    save_plots: bool = True,
    plots_dir: str | Path = "outputs/plots",
) -> dict:
    """
    Full end-to-end pipeline:
      1. Load & preprocess data
      2. Train Naive, Linear, LSTM
      3. Evaluate on test set (MAE, RMSE)
      4. Save metrics to experiment history
      5. Optionally save plots to disk

    Returns a results dict with predictions and metrics.
    """

    # ------------------------------------------------------------------ #
    #  1. Data preparation                                                  #
    # ------------------------------------------------------------------ #
    logger.info("Loading data from %s", data_path)
    data = load_and_prepare(data_path, target_col=target_col, test_fraction=test_fraction)

    for w in data["warnings"]:
        logger.warning(w)

    train_s = data["train_scaled"]
    test_s = data["test_scaled"]
    scaler = data["scaler"]
    test_raw = data["test_raw"]
    test_index = test_raw.index
    actual = test_raw.values

    # ------------------------------------------------------------------ #
    #  2. Naive model                                                       #
    # ------------------------------------------------------------------ #
    logger.info("--- Naive Baseline ---")
    naive = NaiveModel().fit(train_s, lookback)
    naive_preds_scaled = naive.predict(test_s, lookback)
    naive_preds = inverse_scale(scaler, naive_preds_scaled)
    naive_metrics = compute_metrics(actual, naive_preds)
    log_experiment(
        "Naive Baseline",
        {"lookback": lookback, "test_fraction": test_fraction},
        naive_metrics,
        dataset=str(Path(data_path).name),
    )
    logger.info("Naive  MAE=%.4f  RMSE=%.4f", naive_metrics["MAE"], naive_metrics["RMSE"])

    # ------------------------------------------------------------------ #
    #  3. Linear Regression                                                 #
    # ------------------------------------------------------------------ #
    logger.info("--- Linear Regression ---")
    lr_model = LinearModel().fit_with_tail(train_s, lookback)
    lr_preds_scaled = lr_model.predict(test_s, lookback)
    lr_preds = inverse_scale(scaler, lr_preds_scaled)
    lr_metrics = compute_metrics(actual, lr_preds)
    log_experiment(
        "Linear Regression",
        {"lookback": lookback, "test_fraction": test_fraction},
        lr_metrics,
        dataset=str(Path(data_path).name),
    )
    logger.info("LinReg MAE=%.4f  RMSE=%.4f", lr_metrics["MAE"], lr_metrics["RMSE"])

    # ------------------------------------------------------------------ #
    #  4. LSTM                                                              #
    # ------------------------------------------------------------------ #
    logger.info("--- LSTM (hidden=%d, epochs=%d) ---", lstm_hidden, lstm_epochs)

    def _progress(epoch, total, loss):
        if epoch % 5 == 0 or epoch == total:
            logger.info("  Epoch %d/%d  loss=%.6f", epoch, total, loss)

    lstm = LSTMModel(hidden_size=lstm_hidden).fit(
        train_s,
        lookback=lookback,
        epochs=lstm_epochs,
        progress_callback=_progress,
    )
    lstm_preds_scaled = lstm.predict(test_s, lookback)
    lstm_preds = inverse_scale(scaler, lstm_preds_scaled)
    lstm_metrics = compute_metrics(actual, lstm_preds)
    log_experiment(
        "LSTM",
        {"lookback": lookback, "hidden_size": lstm_hidden, "epochs": lstm_epochs,
         "test_fraction": test_fraction},
        lstm_metrics,
        dataset=str(Path(data_path).name),
    )
    logger.info("LSTM   MAE=%.4f  RMSE=%.4f", lstm_metrics["MAE"], lstm_metrics["RMSE"])

    # ------------------------------------------------------------------ #
    #  5. Summary                                                           #
    # ------------------------------------------------------------------ #
    all_metrics = {
        "Naive Baseline": naive_metrics,
        "Linear Regression": lr_metrics,
        "LSTM": lstm_metrics,
    }
    all_preds = {
        "Naive Baseline": naive_preds,
        "Linear Regression": lr_preds,
        "LSTM": lstm_preds,
    }

    logger.info("\n=== Results ===")
    for name, m in all_metrics.items():
        logger.info("%-20s  MAE=%.4f  RMSE=%.4f", name, m["MAE"], m["RMSE"])

    # ------------------------------------------------------------------ #
    #  6. Save plots                                                        #
    # ------------------------------------------------------------------ #
    if save_plots:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig_pred = plot_predictions(test_index, actual, all_preds, target_col)
        fig_pred.savefig(plots_dir / "predictions.png", dpi=150)
        logger.info("Saved predictions plot → %s/predictions.png", plots_dir)

        fig_bar = plot_metrics_bar(all_metrics)
        fig_bar.savefig(plots_dir / "metrics_comparison.png", dpi=150)
        logger.info("Saved metrics bar chart → %s/metrics_comparison.png", plots_dir)

        fig_loss = plot_loss_curve(lstm._train_losses, "LSTM")
        fig_loss.savefig(plots_dir / "lstm_loss.png", dpi=150)
        logger.info("Saved LSTM loss curve → %s/lstm_loss.png", plots_dir)

        import matplotlib.pyplot as plt
        plt.close("all")

    return {
        "metrics": all_metrics,
        "predictions": all_preds,
        "actual": actual,
        "test_index": test_index,
        "lstm_losses": lstm._train_losses,
    }


# --------------------------------------------------------------------------- #
#  CLI entry point                                                              #
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Financial Time-Series Forecasting pipeline")
    p.add_argument("--data",      default="data/sample_prices.csv")
    p.add_argument("--target",    default="Close")
    p.add_argument("--lookback",  type=int, default=20)
    p.add_argument("--epochs",    type=int, default=30)
    p.add_argument("--hidden",    type=int, default=64)
    p.add_argument("--test-frac", type=float, default=0.2, dest="test_frac")
    p.add_argument("--no-plots",  action="store_true")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_pipeline(
        data_path=args.data,
        target_col=args.target,
        test_fraction=args.test_frac,
        lookback=args.lookback,
        lstm_hidden=args.hidden,
        lstm_epochs=args.epochs,
        save_plots=not args.no_plots,
    )
