# Financial Time-Series Forecasting

> A small, complete, and interview-ready financial time-series forecasting project.
> Built for a B.Tech student applying for quantitative AI/ML roles.

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Models](#models)
5. [Evaluation Approach](#evaluation-approach)
6. [Setup](#setup)
7. [Running the Project](#running-the-project)
8. [Tests](#tests)
9. [Experiment History](#experiment-history)
10. [Limitations & Disclaimer](#limitations--disclaimer)

---

## Overview

This project demonstrates a clean, reproducible pipeline for forecasting financial
time-series data using three progressively more complex models:

| Model | Approach | Parameters |
|---|---|---|
| Naive Baseline | Persistence (last value) | None |
| Linear Regression | Auto-regressive lag features (OLS) | Lookback window |
| LSTM | Single-layer recurrent neural network | Hidden size, lookback, epochs |

All models are compared on the **same held-out test period** using **MAE** and **RMSE**
in the original price units. A Streamlit dashboard makes it easy to explore the data,
run experiments, and review results interactively.

---

## Project Structure

```
Financial-Time-Series-Forecasting/
│
├── app.py                      # Streamlit dashboard (entry point)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Loading, validation, scaling, chronological split
│   ├── models.py               # Naive, LinearRegression, LSTM models
│   ├── metrics.py              # MAE, RMSE, Matplotlib plots
│   ├── experiment_logger.py    # JSON-based experiment history
│   └── runner.py               # CLI pipeline (no Streamlit required)
│
├── data/
│   ├── generate_sample.py      # Script to regenerate sample data
│   └── sample_prices.csv       # Built-in synthetic OHLCV dataset (1 565 business days)
│
├── tests/
│   ├── __init__.py
│   └── test_project.py         # pytest suite (~30 tests)
│
├── experiments/
│   └── history.json            # Auto-created; stores experiment records
│
├── outputs/
│   └── plots/                  # Auto-created; saves PNG plots after CLI run
│
├── .github/
│   └── workflows/
│       └── tests.yml           # GitHub Actions CI
│
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Dataset

**Built-in sample** (`data/sample_prices.csv`):
- 1 565 synthetic business-day records from **2018-01-01 to 2023-12-31**
- Generated using Geometric Brownian Motion (GBM) — the standard model for stock prices
- Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Ticker`

**Bring your own data**:
- Any CSV with a `Date` column (parseable by pandas) and a numeric price column
- Upload it directly in the Streamlit sidebar

> **Why GBM?** GBM assumes log-returns are normally distributed and independent —
> a useful benchmark, even though real markets show fat tails, autocorrelation, and volatility clustering.

---

## Models

### 1. Naive Persistence Baseline
- **Prediction**: `ŷ(t) = y(t-1)` — the next value equals the last observed value
- **Why include it?** Any learned model should outperform this. If LSTM barely beats naive, the signal is weak.
- No training required.

### 2. Linear Regression (Auto-Regressive)
- **Features**: `[y(t-L), y(t-L+1), ..., y(t-1)]` where L = lookback window
- **Fits**: Ordinary Least Squares on these lag features
- **Limitation**: Assumes a linear relationship — very strong assumption for prices
- **In an interview**: "This is essentially an AR(L) model fit with OLS instead of Yule-Walker."

### 3. LSTM (Long Short-Term Memory)
- **Architecture**: 1 LSTM layer → Linear output head (many-to-one)
- **Input**: Sequence of length `lookback` fed as `(batch, seq_len, 1)` tensors
- **Why LSTM?** The gating mechanism (input/forget/output gates) lets the network selectively
  remember or forget past information — better at non-linear temporal patterns than linear regression
- **Training**: Adam optimizer, MSE loss, CPU-only (no GPU needed for this size)
- **Data leakage prevention**: Windows are shuffled for mini-batch training (each window is internally
  chronological), but raw test observations are never seen during training

---

## Evaluation Approach

**Chronological split** — the single most important rule in time-series ML:

```
|─────────── Training (80%) ──────────|─── Test (20%) ───|
2018-01-01                         2022-10-17         2023-12-31
```

- The scaler (`MinMaxScaler`) is fitted **only on training data** and then applied to test data
- Lag features for the first test window use training tail values — no test data is peeked at
- **MAE** (Mean Absolute Error): average error magnitude in original price units
- **RMSE** (Root Mean Squared Error): penalises large errors more; also in original units

**Formula reference**:

$$\text{MAE} = \frac{1}{n}\sum_{t=1}^{n}|y_t - \hat{y}_t|$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t - \hat{y}_t)^2}$$

---

## Setup

### Requirements
- Python 3.11+ (tested on 3.13)
- All packages listed in `requirements.txt`

### Install

```bash
# Clone / navigate to the project folder
cd Financial-Time-Series-Forecasting

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Project

### Option A – Streamlit Dashboard (recommended)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

The dashboard has three tabs:
- **📊 Data** — inspect the dataset, see the train/test split boundary
- **🔮 Forecast** — choose a model, configure settings, run training, view predictions and metrics
- **📝 History** — browse all past experiment runs

### Option B – CLI Pipeline

```bash
# Run all three models with default settings
python -m src.runner

# Custom settings
python -m src.runner --lookback 30 --epochs 50 --hidden 128 --test-frac 0.2

# Skip saving plots
python -m src.runner --no-plots
```

All results are saved automatically:
- `experiments/history.json` — experiment records
- `outputs/plots/predictions.png` — actual vs predicted chart
- `outputs/plots/metrics_comparison.png` — bar chart of MAE/RMSE
- `outputs/plots/lstm_loss.png` — LSTM training loss curve

### Option C – Regenerate Sample Data

```bash
python data/generate_sample.py
```

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage (requires pytest-cov)
pytest tests/ -v --cov=src
```

**Test coverage** includes:
- Data loading and date validation
- Chronological split (no overlap, correct sizes, order preserved)
- Scaler fitted only on training data
- Lag feature construction with no lookahead
- MAE and RMSE correctness (including known-value cases)
- All three models: output shape, finite predictions, loss decrease, untrained error

---

## Experiment History

Every training run is logged to `experiments/history.json`. Each record contains:
- ISO 8601 timestamp
- Model name
- Configuration (lookback, hidden size, epochs, test fraction)
- MAE and RMSE in original price units
- Dataset filename

View history from the **History** tab in the dashboard or programmatically:

```python
from src.experiment_logger import load_history
df = load_history()
print(df.tail())
```

---

## Limitations & Disclaimer

> ⚠️ **Low MAE/RMSE does NOT imply a profitable trading strategy.**
> These metrics measure how closely a model tracks historical prices — not whether it
> can generate alpha in live markets. Real trading involves transaction costs, slippage,
> market impact, regime changes, and execution risk that are absent here.

Additional limitations:

1. **Synthetic data**: The included sample uses GBM, which does not capture real market
   microstructure (fat tails, volatility clustering, correlation breaks).

2. **One-step-ahead forecasting only**: All models predict one day forward. Multi-step
   forecasts accumulate errors rapidly and are much harder.

3. **No feature engineering**: Models use only the raw target series. Real quantitative
   strategies incorporate order book data, earnings, macro indicators, alternative data, etc.

4. **No transaction costs or risk**: A useful next step is to wrap predictions in a simple
   long/short backtest with realistic cost assumptions (e.g. using `vectorbt` or `backtrader`).

5. **Small LSTM**: 1 layer, modest hidden size, 30 epochs. This is intentional for
   laptop-friendly training but is far from state-of-the-art sequence modelling.

---

## Interview Talking Points

| Question | Answer |
|---|---|
| Why not shuffle time-series data? | Shuffling creates data leakage — the model would see future values during training, leading to optimistic metrics that don't generalise. |
| Why fit the scaler on training data only? | Fitting on all data uses test set statistics, which are unknown at prediction time in practice. |
| What does the LSTM forget gate do? | It learns which parts of the cell state to discard at each time step, allowing the network to reset irrelevant history. |
| What does MAE vs RMSE tell you? | MAE gives equal weight to all errors; RMSE penalises large errors more. If RMSE >> MAE, there are occasional large prediction failures. |
| Why is the naive baseline useful? | It sets a minimum benchmark. Beating naive by only a tiny margin suggests the model has learned very little useful signal. |
