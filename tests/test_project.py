"""
Tests for Financial Time-Series Forecasting project.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------------- #
#  Fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _make_series(n: int = 200, seed: int = 0) -> pd.Series:
    """Create a synthetic price series with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="Close")


def _make_csv(tmp_path: Path, n: int = 200) -> Path:
    """Write a minimal CSV file and return its path."""
    series = _make_series(n)
    df = series.reset_index()
    df.columns = ["Date", "Close"]
    df["Open"] = series.values + 0.5
    path = tmp_path / "test_prices.csv"
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
#  1. Data loading & validation                                                 #
# --------------------------------------------------------------------------- #


class TestDataLoader:
    def test_load_csv_returns_datetime_index(self, tmp_path):
        from src.data_loader import load_csv
        csv_path = _make_csv(tmp_path)
        df = load_csv(csv_path)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_dates_are_sorted(self, tmp_path):
        from src.data_loader import load_csv
        csv_path = _make_csv(tmp_path)
        df = load_csv(csv_path)
        assert df.index.is_monotonic_increasing

    def test_missing_target_raises(self, tmp_path):
        from src.data_loader import load_csv, validate_dataframe
        csv_path = _make_csv(tmp_path)
        df = load_csv(csv_path)
        with pytest.raises(ValueError, match="not found"):
            validate_dataframe(df, target_col="NonExistent")

    def test_missing_values_warning(self, tmp_path):
        from src.data_loader import load_csv, validate_dataframe
        csv_path = _make_csv(tmp_path)
        df = load_csv(csv_path)
        # Inject a NaN
        df.iloc[5, df.columns.get_loc("Close")] = float("nan")
        warnings = validate_dataframe(df, "Close")
        assert any("missing" in w.lower() for w in warnings)

    def test_file_not_found_raises(self):
        from src.data_loader import load_csv
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent_file_xyz.csv")


# --------------------------------------------------------------------------- #
#  2. Chronological split                                                        #
# --------------------------------------------------------------------------- #


class TestChronologicalSplit:
    def test_no_overlap(self):
        from src.data_loader import train_test_split_ts
        series = _make_series(100)
        train, test = train_test_split_ts(series, test_fraction=0.2)
        # Training dates must all be before test dates
        assert train.index.max() < test.index.min()

    def test_correct_sizes(self):
        from src.data_loader import train_test_split_ts
        series = _make_series(100)
        train, test = train_test_split_ts(series, test_fraction=0.2)
        assert len(train) + len(test) == len(series)
        assert len(test) == 20

    def test_order_preserved(self):
        from src.data_loader import train_test_split_ts
        series = _make_series(100)
        train, test = train_test_split_ts(series, test_fraction=0.2)
        assert train.index.is_monotonic_increasing
        assert test.index.is_monotonic_increasing

    def test_invalid_fraction_raises(self):
        from src.data_loader import train_test_split_ts
        series = _make_series(100)
        with pytest.raises(ValueError):
            train_test_split_ts(series, test_fraction=0.0)
        with pytest.raises(ValueError):
            train_test_split_ts(series, test_fraction=1.5)


# --------------------------------------------------------------------------- #
#  3. Scaler – fit on train only                                                 #
# --------------------------------------------------------------------------- #


class TestScaler:
    def test_scaler_range(self):
        from src.data_loader import fit_scaler, scale, train_test_split_ts
        series = _make_series(200)
        train, test = train_test_split_ts(series, 0.2)
        scaler = fit_scaler(train)
        train_scaled = scale(scaler, train)
        # Training data should be in [0, 1]
        assert train_scaled.min() >= 0.0 - 1e-9
        assert train_scaled.max() <= 1.0 + 1e-9

    def test_inverse_scale_recovers_original(self):
        from src.data_loader import fit_scaler, scale, inverse_scale
        series = _make_series(200)
        scaler = fit_scaler(series)
        scaled = scale(scaler, series)
        recovered = inverse_scale(scaler, scaled)
        np.testing.assert_allclose(recovered, series.values, rtol=1e-5)


# --------------------------------------------------------------------------- #
#  4. Lag features – no lookahead                                               #
# --------------------------------------------------------------------------- #


class TestLagFeatures:
    def test_shape(self):
        from src.data_loader import make_lag_features
        arr = np.arange(50.0)
        X, y = make_lag_features(arr, lookback=5)
        assert X.shape == (45, 5)
        assert y.shape == (45,)

    def test_no_lookahead(self):
        """Each row of X must only contain values strictly before y."""
        from src.data_loader import make_lag_features
        arr = np.arange(20.0)
        X, y = make_lag_features(arr, lookback=3)
        for i, (row, target) in enumerate(zip(X, y)):
            # All feature values must be strictly less than the target index value
            assert (row < target).all(), f"Row {i} contains future information!"


# --------------------------------------------------------------------------- #
#  5. Metrics                                                                   #
# --------------------------------------------------------------------------- #


class TestMetrics:
    def test_mae_perfect(self):
        from src.metrics import mae
        a = np.array([1.0, 2.0, 3.0])
        assert mae(a, a) == pytest.approx(0.0)

    def test_rmse_perfect(self):
        from src.metrics import rmse
        a = np.array([1.0, 2.0, 3.0])
        assert rmse(a, a) == pytest.approx(0.0)

    def test_mae_known_value(self):
        from src.metrics import mae
        actual = np.array([10.0, 20.0, 30.0])
        pred   = np.array([12.0, 18.0, 33.0])
        # |−2| + |2| + |−3| = 7 → mean = 7/3
        assert mae(actual, pred) == pytest.approx(7 / 3, rel=1e-6)

    def test_rmse_known_value(self):
        from src.metrics import rmse
        actual = np.array([0.0, 0.0])
        pred   = np.array([3.0, 4.0])
        # sqrt((9+16)/2) = sqrt(12.5)
        assert rmse(actual, pred) == pytest.approx(np.sqrt(12.5), rel=1e-6)

    def test_rmse_ge_mae(self):
        from src.metrics import mae, rmse
        rng = np.random.default_rng(7)
        a = rng.uniform(0, 100, 50)
        p = rng.uniform(0, 100, 50)
        assert rmse(a, p) >= mae(a, p)


# --------------------------------------------------------------------------- #
#  6. Naive Model                                                               #
# --------------------------------------------------------------------------- #


class TestNaiveModel:
    def test_output_length(self):
        from src.models import NaiveModel
        train = np.linspace(0, 1, 100)
        test  = np.linspace(1, 2, 20)
        model = NaiveModel().fit(train)
        preds = model.predict(test)
        assert len(preds) == len(test)

    def test_first_pred_is_last_train(self):
        from src.models import NaiveModel
        train = np.array([1.0, 2.0, 3.0])
        test  = np.array([4.0, 5.0])
        model = NaiveModel().fit(train)
        preds = model.predict(test)
        # First prediction should be the last training value
        assert preds[0] == pytest.approx(train[-1])


# --------------------------------------------------------------------------- #
#  7. Linear Regression Model                                                   #
# --------------------------------------------------------------------------- #


class TestLinearModel:
    def test_output_length(self):
        from src.models import LinearModel
        train = np.linspace(0, 1, 100)
        test  = np.linspace(1, 2, 20)
        model = LinearModel().fit_with_tail(train, lookback=5)
        preds = model.predict(test, lookback=5)
        assert len(preds) == len(test)

    def test_predictions_are_finite(self):
        from src.models import LinearModel
        rng = np.random.default_rng(99)
        train = rng.uniform(0, 1, 150)
        test  = rng.uniform(0, 1, 30)
        model = LinearModel().fit_with_tail(train, lookback=10)
        preds = model.predict(test, lookback=10)
        assert np.all(np.isfinite(preds))


# --------------------------------------------------------------------------- #
#  8. LSTM Model                                                                #
# --------------------------------------------------------------------------- #


class TestLSTMModel:
    def test_output_length(self):
        from src.models import LSTMModel
        rng = np.random.default_rng(0)
        train = rng.uniform(0, 1, 100)
        test  = rng.uniform(0, 1, 20)
        model = LSTMModel(hidden_size=16).fit(train, lookback=5, epochs=2)
        preds = model.predict(test, lookback=5)
        assert len(preds) == len(test)

    def test_predictions_are_finite(self):
        from src.models import LSTMModel
        rng = np.random.default_rng(1)
        train = rng.uniform(0, 1, 100)
        test  = rng.uniform(0, 1, 20)
        model = LSTMModel(hidden_size=16).fit(train, lookback=5, epochs=2)
        preds = model.predict(test, lookback=5)
        assert np.all(np.isfinite(preds))

    def test_loss_decreases(self):
        """Training loss should generally decrease over many epochs on a signal."""
        from src.models import LSTMModel
        # Simple linear trend — easy to learn
        train = np.linspace(0, 1, 200)
        model = LSTMModel(hidden_size=32).fit(train, lookback=10, epochs=20)
        losses = model._train_losses
        # Loss at end of training should be lower than at start
        assert losses[-1] < losses[0], "Loss did not decrease during training"

    def test_predict_before_fit_raises(self):
        from src.models import LSTMModel
        model = LSTMModel()
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(np.array([1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------- #
#  9. Experiment Logger                                                         #
# --------------------------------------------------------------------------- #


class TestExperimentLogger:
    def test_log_and_load(self, tmp_path):
        from src.experiment_logger import log_experiment, load_history
        log_path = tmp_path / "history.json"
        log_experiment(
            "TestModel",
            {"lookback": 10},
            {"MAE": 1.23, "RMSE": 2.34},
            log_path=log_path,
        )
        hist = load_history(log_path)
        assert len(hist) == 1
        assert hist.iloc[0]["model"] == "TestModel"
        assert hist.iloc[0]["MAE"] == pytest.approx(1.23)

    def test_multiple_runs_accumulate(self, tmp_path):
        from src.experiment_logger import log_experiment, load_history
        log_path = tmp_path / "history.json"
        for i in range(3):
            log_experiment(
                f"Model{i}", {"run": i}, {"MAE": float(i), "RMSE": float(i)},
                log_path=log_path
            )
        hist = load_history(log_path)
        assert len(hist) == 3

    def test_empty_history_returns_dataframe(self, tmp_path):
        from src.experiment_logger import load_history
        hist = load_history(tmp_path / "nonexistent.json")
        assert isinstance(hist, pd.DataFrame)
