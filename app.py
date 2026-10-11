"""
app.py – Streamlit dashboard for Financial Time-Series Forecasting.

Sections
--------
1. Sidebar  – dataset selection, model choice, hyperparameters
2. Data tab  – raw data table, price chart, validation warnings
3. Forecast tab – train → predict → plot actual vs predicted
4. History tab  – experiment log table
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on the path when launched via `streamlit run app.py`
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import (
    SAMPLE_DATA_PATH,
    clean_dataframe,
    fit_scaler,
    load_and_prepare,
    load_csv,
    scale,
    inverse_scale,
    train_test_split_ts,
    validate_dataframe,
)
from src.experiment_logger import load_history, log_experiment
from src.metrics import (
    compute_metrics,
    plot_loss_curve,
    plot_metrics_bar,
    plot_predictions,
)
from src.models import LSTMModel, LinearModel, NaiveModel

# --------------------------------------------------------------------------- #
#  Page config                                                                  #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Financial Time-Series Forecaster",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Financial Time-Series Forecasting")
st.caption(
    "Naive baseline · Linear Regression · LSTM  |  Chronological split · MAE · RMSE"
)

# --------------------------------------------------------------------------- #
#  Sidebar – configuration                                                      #
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("⚙️ Configuration")

    # -- Dataset --
    st.subheader("Dataset")
    use_sample = st.radio(
        "Source",
        ["Use built-in sample data", "Upload your own CSV"],
        index=0,
    )

    if use_sample == "Upload your own CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(uploaded.read())
            tmp.flush()
            data_path = tmp.name
            data_label = uploaded.name
        else:
            st.info("No file uploaded — using sample data.")
            data_path = str(SAMPLE_DATA_PATH)
            data_label = "sample_prices.csv"
    else:
        data_path = str(SAMPLE_DATA_PATH)
        data_label = "sample_prices.csv"

    target_col = st.selectbox("Target column", ["Close", "Open", "High", "Low"])
    test_fraction = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)

    # -- Model --
    st.subheader("Model")
    model_choice = st.selectbox(
        "Choose model",
        ["Naive Baseline", "Linear Regression", "LSTM", "All models (compare)"],
    )

    lookback = st.slider("Lookback window", 5, 60, 20, 5,
                          help="Number of past days used as input features.")

    lstm_hidden = 64
    lstm_epochs = 30
    if model_choice in ("LSTM", "All models (compare)"):
        lstm_hidden = st.slider("LSTM hidden size", 16, 256, 64, 16)
        lstm_epochs = st.slider("LSTM training epochs", 5, 100, 30, 5)

    run_btn = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
#  Tabs                                                                         #
# --------------------------------------------------------------------------- #

tab_data, tab_forecast, tab_history = st.tabs(["📊 Data", "🔮 Forecast", "📝 History"])

# =========================================================================== #
#  TAB 1 – Data inspection                                                      #
# =========================================================================== #

with tab_data:
    st.subheader(f"Dataset: `{data_label}`")
    try:
        df_raw = load_csv(data_path)
        warnings = validate_dataframe(df_raw, target_col)
        for w in warnings:
            st.warning(w)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df_raw):,}")
        col2.metric("Start date", str(df_raw.index.min().date()))
        col3.metric("End date", str(df_raw.index.max().date()))

        st.dataframe(df_raw.reset_index().tail(200), use_container_width=True)

        st.subheader(f"{target_col} Price History")
        st.line_chart(df_raw[target_col])

        # Show train/test split line
        df_clean = clean_dataframe(df_raw, target_col)
        series = df_clean[target_col]
        train_raw, test_raw = train_test_split_ts(series, test_fraction)
        split_date = test_raw.index[0]
        st.caption(
            f"**Train:** {train_raw.index[0].date()} → {train_raw.index[-1].date()} "
            f"({len(train_raw):,} rows) &nbsp;&nbsp; "
            f"**Test:** {test_raw.index[0].date()} → {test_raw.index[-1].date()} "
            f"({len(test_raw):,} rows)"
        )

    except Exception as exc:
        st.error(f"Could not load data: {exc}")

# =========================================================================== #
#  TAB 2 – Forecast                                                             #
# =========================================================================== #

with tab_forecast:
    if not run_btn:
        st.info("Configure settings in the sidebar and press **Run Forecast** to begin.")
    else:
        # ------------------------------------------------------------------ #
        #  Load & prepare                                                       #
        # ------------------------------------------------------------------ #
        with st.spinner("Loading and preparing data …"):
            try:
                data = load_and_prepare(data_path, target_col, test_fraction)
            except Exception as exc:
                st.error(str(exc))
                st.stop()

        train_s = data["train_scaled"]
        test_s  = data["test_scaled"]
        scaler  = data["scaler"]
        actual  = data["test_raw"].values
        test_index = data["test_raw"].index

        for w in data["warnings"]:
            st.warning(w)

        all_metrics: dict[str, dict] = {}
        all_preds:   dict[str, np.ndarray] = {}
        lstm_losses: list[float] = []

        models_to_run: list[str] = (
            ["Naive Baseline", "Linear Regression", "LSTM"]
            if model_choice == "All models (compare)"
            else [model_choice]
        )

        # ------------------------------------------------------------------ #
        #  Train each selected model                                            #
        # ------------------------------------------------------------------ #

        for mname in models_to_run:
            with st.spinner(f"Training **{mname}** …"):

                if mname == "Naive Baseline":
                    model = NaiveModel().fit(train_s, lookback)
                    preds_scaled = model.predict(test_s, lookback)

                elif mname == "Linear Regression":
                    model = LinearModel().fit_with_tail(train_s, lookback)
                    preds_scaled = model.predict(test_s, lookback)

                elif mname == "LSTM":
                    progress_bar = st.progress(0, text="LSTM epoch 0 / …")
                    loss_placeholder = st.empty()
                    lstm_loss_history: list[float] = []

                    def _cb(epoch, total, loss, _pb=progress_bar, _lp=loss_placeholder,
                            _lh=lstm_loss_history):
                        _pb.progress(epoch / total, text=f"LSTM epoch {epoch}/{total}  loss={loss:.6f}")
                        _lh.append(loss)

                    model = LSTMModel(hidden_size=lstm_hidden).fit(
                        train_s,
                        lookback=lookback,
                        epochs=lstm_epochs,
                        progress_callback=_cb,
                    )
                    lstm_losses = lstm_loss_history
                    preds_scaled = model.predict(test_s, lookback)
                    progress_bar.empty()
                else:
                    continue

                preds = inverse_scale(scaler, preds_scaled)
                metrics = compute_metrics(actual, preds)
                all_metrics[mname] = metrics
                all_preds[mname] = preds

                log_experiment(
                    mname,
                    {
                        "lookback": lookback,
                        "hidden_size": lstm_hidden if mname == "LSTM" else None,
                        "epochs": lstm_epochs if mname == "LSTM" else None,
                        "test_fraction": test_fraction,
                    },
                    metrics,
                    dataset=data_label,
                )

        # ------------------------------------------------------------------ #
        #  Display results                                                      #
        # ------------------------------------------------------------------ #

        st.subheader("📊 Evaluation Metrics")
        metric_cols = st.columns(len(all_metrics))
        for col, (mname, m) in zip(metric_cols, all_metrics.items()):
            col.metric(mname, "", border=True)
            col.write(f"**MAE:** {m['MAE']:.4f}")
            col.write(f"**RMSE:** {m['RMSE']:.4f}")

        st.subheader("Actual vs Predicted")
        fig_pred = plot_predictions(test_index, actual, all_preds, target_col)
        st.pyplot(fig_pred)

        if len(all_metrics) > 1:
            st.subheader("Model Comparison")
            fig_bar = plot_metrics_bar(all_metrics)
            st.pyplot(fig_bar)

        if lstm_losses:
            st.subheader("LSTM Training Loss")
            fig_loss = plot_loss_curve(lstm_losses, "LSTM")
            st.pyplot(fig_loss)

        st.caption(
            "⚠️ **Disclaimer:** Low MAE/RMSE does NOT imply a profitable trading "
            "strategy. These metrics quantify historical fit, not future alpha."
        )

# =========================================================================== #
#  TAB 3 – Experiment History                                                   #
# =========================================================================== #

with tab_history:
    st.subheader("📝 Experiment History")
    hist_df = load_history()
    if hist_df.empty:
        st.info("No experiments logged yet. Run a forecast first!")
    else:
        st.dataframe(hist_df.sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button(
            "⬇️ Download history as CSV",
            data=hist_df.to_csv(index=False),
            file_name="experiment_history.csv",
            mime="text/csv",
        )
