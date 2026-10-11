"""
models.py – Three forecasting models: Naive baseline, Linear Regression, LSTM.

Each model exposes a common interface:
    fit(train_scaled, lookback)  →  self
    predict(test_scaled, lookback)  →  1-D np.ndarray (scaled predictions)

This makes it easy to compare them on the same test window.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from src.data_loader import make_lag_features


# --------------------------------------------------------------------------- #
#  1. Naive Persistence Baseline                                                #
# --------------------------------------------------------------------------- #


class NaiveModel:
    """
    Persistence (last-value) baseline.

    Predicts that the next value equals the most recent observed value.
    This is the simplest sensible forecast and acts as a lower-bound
    benchmark — any learned model should beat it.

    No training is required.
    """

    name = "Naive Baseline"

    def fit(self, train_scaled: np.ndarray, lookback: int = 1) -> "NaiveModel":
        """Store the last training value so we can start predictions."""
        self._last_train_value = train_scaled[-1]
        return self

    def predict(self, test_scaled: np.ndarray, lookback: int = 1) -> np.ndarray:
        """
        Return predictions where each prediction = the previous actual value.
        The first prediction uses the last training value.
        """
        # Prepend last training value to get the 'previous' for step 0
        source = np.concatenate([[self._last_train_value], test_scaled[:-1]])
        return source


# --------------------------------------------------------------------------- #
#  2. Linear Regression on Lagged Features                                     #
# --------------------------------------------------------------------------- #


class LinearModel:
    """
    Ordinary Least Squares regression on lag features.

    Features: [t-lookback, ..., t-2, t-1]  →  target: t

    Why lagged features?
    --------------------
    Instead of learning a sequence, we treat each prediction as a tabular
    regression problem: predict the current price from the past `lookback`
    prices. This is a classical auto-regressive (AR) approach.

    Limitation: it cannot capture long-range temporal dependencies, and it
    assumes a linear relationship — strong assumptions for financial prices.
    """

    name = "Linear Regression"

    def __init__(self) -> None:
        self._model = LinearRegression()

    def fit(self, train_scaled: np.ndarray, lookback: int = 20) -> "LinearModel":
        X, y = make_lag_features(train_scaled, lookback)
        self._model.fit(X, y)
        self._lookback = lookback
        self._tail = train_scaled[-lookback:]   # needed to seed features for 1st test step
        return self

    def predict(self, test_scaled: np.ndarray, lookback: int = 20) -> np.ndarray:
        """
        Build lagged features from *test* data and predict.

        Note: to generate features for the very first test step we need the
        last `lookback` training values. These are stored as self._tail during fit().
        """
        combined = np.concatenate([self._tail, test_scaled])
        X, _ = make_lag_features(combined, lookback)
        return self._model.predict(X)

    def fit_with_tail(
        self, train_scaled: np.ndarray, lookback: int = 20
    ) -> "LinearModel":
        """Alias for fit() — kept for backwards compatibility."""
        return self.fit(train_scaled, lookback)


# --------------------------------------------------------------------------- #
#  3. LSTM Neural Network                                                       #
# --------------------------------------------------------------------------- #


class _LSTMNet(nn.Module):
    """
    A small single-layer LSTM followed by a linear output head.

    Architecture
    ------------
    Input  →  LSTM(hidden_size, num_layers=1)  →  Linear(hidden_size, 1)

    Only the hidden state of the LAST time step is used for prediction.
    This is the standard 'many-to-one' LSTM pattern for sequence regression.
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,   # input shape: (batch, seq_len, features)
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)           # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]     # take only the last step
        return self.fc(last_hidden).squeeze(-1)


class LSTMModel:
    """
    PyTorch LSTM forecaster with configurable lookback and hidden size.

    Training details
    ----------------
    * Loss:      Mean Squared Error (MSE)
    * Optimizer: Adam
    * Data:      Sliding window of length `lookback` from scaled training series
    * Device:    CPU (no GPU needed for this dataset size)

    Why LSTM?
    ---------
    LSTMs have a gating mechanism (input, forget, output gates) that lets the
    network learn WHICH past information to carry forward and which to discard.
    This makes them better at capturing non-linear temporal patterns than
    plain linear regression, at the cost of more parameters and training time.
    """

    name = "LSTM"

    def __init__(self, hidden_size: int = 64, num_layers: int = 1) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._net: _LSTMNet | None = None
        self._tail: np.ndarray | None = None

    def fit(
        self,
        train_scaled: np.ndarray,
        lookback: int = 20,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
        progress_callback=None,
    ) -> "LSTMModel":
        """
        Train the LSTM on sliding windows of the scaled training series.

        Args:
            train_scaled:      Scaled 1-D training array.
            lookback:          Sequence length fed into the LSTM.
            epochs:            Number of full passes over the training data.
            lr:                Adam learning rate.
            batch_size:        Mini-batch size.
            progress_callback: Optional callable(epoch, total, loss) for UI.
        """
        self._lookback = lookback
        self._tail = train_scaled[-lookback:]

        X, y = make_lag_features(train_scaled, lookback)

        # Convert to tensors: X needs shape (batch, seq_len, 1)
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(y, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
            # shuffle=True is correct here: we're shuffling WINDOWS (each window
            # is already chronologically ordered internally). We are NOT shuffling
            # raw time-series observations.
        )

        self._net = _LSTMNet(self.hidden_size, self.num_layers)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self._net.train()
        self._train_losses: list[float] = []
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            avg_loss = epoch_loss / len(dataset)
            self._train_losses.append(avg_loss)
            if progress_callback:
                progress_callback(epoch, epochs, avg_loss)

        self._net.eval()
        return self

    def predict(self, test_scaled: np.ndarray, lookback: int = 20) -> np.ndarray:
        """
        Generate one-step-ahead predictions for the test period.

        We prepend the last `lookback` training values to the test array so
        that we can build a window for the very first test observation.
        No future test values are used when predicting step t.
        """
        if self._net is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")

        combined = np.concatenate([self._tail, test_scaled])
        X, _ = make_lag_features(combined, lookback)

        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self._net.eval()
        with torch.no_grad():
            preds = self._net(X_t).numpy()
        return preds
