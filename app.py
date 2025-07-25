import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU to avoid CUDA errors

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from prophet import Prophet
import datetime

# Define consistent fonts and colors
FONT_FAMILY = "DejaVu Sans"
TITLE_FONT_SIZE = 20
AXIS_TITLE_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14

COLOR_ACTUAL = "#1f77b4"  # blue
COLOR_ARIMA = "#2ca02c"   # green
COLOR_PROPHET = "#ff7f0e" # orange
COLOR_LSTM = "#9467bd"    # purple
COLOR_BAR_ARIMA = "red"
COLOR_BAR_PROPHET = "orange"
COLOR_BAR_LSTM = "purple"

st.set_page_config(page_title="\U0001F4C8 Stock Price Forecasting Dashboard", layout="wide")
st.markdown(f"<h1 style='font-family:{FONT_FAMILY}; font-weight:bold;'>{chr(0x1F4C8)} Stock Price Forecasting Dashboard</h1>", unsafe_allow_html=True)

# Sidebar inputs
ticker = st.text_input("Enter Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.date_input("End Date", datetime.date(2024, 12, 31))
model_type = st.selectbox("Select Model", ["ARIMA", "Prophet", "LSTM", "Comparison"])

# Load Data
@st.cache_data(show_spinner=True)
def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end)
        if df.empty or "Close" not in df.columns:
            raise ValueError("No data found for symbol")
        df = df[["Close"]].dropna()
        df.reset_index(inplace=True)
        df.columns = ["ds", "y"]
        if len(df) < 20:
            raise ValueError("Insufficient data from yfinance")
        return df, False
    except Exception as e:
        # Load sample data fallback
        sample_path = "sample_aapl.csv"
        if os.path.exists(sample_path):
            df_sample = pd.read_csv(sample_path)
            df_sample.columns = ["ds", "y"]
            st.warning(f"Using sample data due to data unavailability or error: {e}")
            return df_sample, True
        else:
            st.error(f"Failed to load data and sample data not found: {e}")
            return pd.DataFrame(columns=["ds", "y"]), False

data, is_sample = load_data(ticker, start_date, end_date)
if data.shape[0] < 20:
    st.error("Insufficient data for modeling. Please select a longer date range or a different stock symbol.")
elif is_sample:
    st.info("Note: Sample data is being used for demonstration purposes.")
else:
    st.markdown(f"<h3 style='font-family:{FONT_FAMILY}; font-weight:bold;'>{chr(0x1F4C4)} Data for {ticker} from {start_date} to {end_date}</h3>", unsafe_allow_html=True)
    st.dataframe(data.tail())

# Line chart for closing price
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Close', line=dict(color=COLOR_ACTUAL)))
fig1.update_layout(
    title=dict(text=f"{ticker} Closing Price", font=dict(family=FONT_FAMILY, size=TITLE_FONT_SIZE, color="black"), x=0.5),
    xaxis_title=dict(text="Date", font=dict(family=FONT_FAMILY, size=AXIS_TITLE_FONT_SIZE, color="black")),
    yaxis_title=dict(text="Close", font=dict(family=FONT_FAMILY, size=AXIS_TITLE_FONT_SIZE, color="black")),
    font=dict(family=FONT_FAMILY, size=TICK_FONT_SIZE, color="black"),
    plot_bgcolor="white",
    margin=dict(l=40, r=40, t=60, b=40),
    xaxis=dict(showgrid=True, gridcolor="lightgray"),
    yaxis=dict(showgrid=True, gridcolor="lightgray")
)
st.plotly_chart(fig1, use_container_width=True)

# Train-Test split with minimum size checks
min_train_size = 30  # minimum training size for models
min_test_size = 10   # minimum test size

if len(data) < (min_train_size + min_test_size):
    st.error(f"Not enough data for modeling. Please select a longer date range or different stock symbol. Minimum required data points: {min_train_size + min_test_size}")
    train, test = None, None
else:
    train_size = max(min_train_size, int(len(data) * 0.8))
    test_size = len(data) - train_size
    if test_size < min_test_size:
        train_size = len(data) - min_test_size
    train, test = data[:train_size], data[train_size:]

# Forecasting Functions
def run_arima(train, test):
    train = train.dropna(subset=['y'])
    test = test.dropna(subset=['y'])
    if len(train) < 10:
        raise ValueError("Training data too small for ARIMA model.")
    try:
        model = ARIMA(train['y'], order=(5, 1, 0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test['y'], forecast))
        return forecast, rmse
    except Exception as e:
        raise RuntimeError(f"ARIMA model fitting failed: {e}")

def run_prophet(train, test):
    train = train.dropna(subset=['ds', 'y'])
    test = test.dropna(subset=['ds', 'y'])
    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(periods=len(test), freq='D')
    future = future.dropna(subset=['ds'])
    forecast = model.predict(future)
    # Ensure 'ds' columns are datetime for both dataframes before merging
    test = test.copy()
    forecast_subset = forecast[['ds', 'yhat']].copy()
    test['ds'] = pd.to_datetime(test['ds'])
    forecast_subset['ds'] = pd.to_datetime(forecast_subset['ds'])
    merged = pd.merge(test, forecast_subset, on='ds', how='left')
    # Drop NaNs in merged before RMSE calculation
    merged = merged.dropna(subset=['y', 'yhat'])
    if merged.empty:
        raise ValueError("Merged dataframe is empty after dropping NaNs, cannot compute RMSE.")
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    return merged['yhat'], rmse

def run_lstm(train, test):
    train = train.dropna(subset=['y'])
    test = test.dropna(subset=['y'])
    # Prepare series data for LSTM
    series = train['y'].values.reshape(-1, 1)
    gen = TimeseriesGenerator(series, series, length=10, batch_size=1)

    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(10, 1)))  # Changed activation to 'tanh'
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    try:
        model.fit(gen, epochs=20, verbose=0)
    except Exception as e:
        raise RuntimeError(f"LSTM model training failed: {e}")

    predictions = []
    # Use last 10 points from train as seed input
    curr_batch = series[-10:].reshape(1, 10, 1)
    for i in range(len(test)):
        pred = model.predict(curr_batch, verbose=0)[0][0]
        predictions.append(pred)
        curr_batch = np.append(curr_batch[:, 1:, :], [[[pred]]], axis=1)

    rmse = np.sqrt(mean_squared_error(test['y'], predictions))
    return predictions, rmse

# Visualization logic
def plot_forecast(title, actual, predicted, model_color, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual, mode='lines', name='Actual', line=dict(color=COLOR_ACTUAL)))
    fig.add_trace(go.Scatter(x=actual.index, y=predicted, mode='lines', name=f'{model_name} Forecast', line=dict(color=model_color)))
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT_FAMILY, size=TITLE_FONT_SIZE, color="black"), x=0.5),
        xaxis_title=dict(text="Date", font=dict(family=FONT_FAMILY, size=AXIS_TITLE_FONT_SIZE, color="black")),
        yaxis_title=dict(text="Price", font=dict(family=FONT_FAMILY, size=AXIS_TITLE_FONT_SIZE, color="black")),
        font=dict(family=FONT_FAMILY, size=TICK_FONT_SIZE, color="black"),
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray")
    )
    st.plotly_chart(fig, use_container_width=True)

# Run selected model
if data.shape[0] >= 20:
    if model_type == "ARIMA":
        st.markdown(f"<h3 style='font-family:{FONT_FAMILY}; font-weight:bold;'>{chr(0x1F4C8)} ARIMA Forecast</h3>", unsafe_allow_html=True)
        try:
            preds, rmse = run_arima(train, test)
            plot_forecast("ARIMA Forecast vs Actual", test.set_index('ds')['y'], preds, COLOR_ARIMA, "ARIMA")
            st.metric("RMSE", f"{rmse:.2f}")
        except Exception as e:
            st.error(f"ARIMA model error: {e}")

    elif model_type == "Prophet":
        st.markdown(f"<h3 style='font-family:{FONT_FAMILY}; font-weight:bold;'>{chr(0x1F916)} Prophet Forecast</h3>", unsafe_allow_html=True)
        try:
            preds, rmse = run_prophet(train, test)
            plot_forecast("Prophet Forecast vs Actual", test.set_index('ds')['y'], preds, COLOR_PROPHET, "Prophet")
            st.metric("RMSE", f"{rmse:.2f}")
        except Exception as e:
            st.error(f"Prophet model error: {e}")

    elif model_type == "LSTM":
        st.markdown(f"<h3 style='font-family:{FONT_FAMILY}; font-weight:bold;'>{chr(0x1F916)} LSTM Forecast</h3>", unsafe_allow_html=True)
        try:
            preds, rmse = run_lstm(train, test)
            plot_forecast("LSTM Forecast vs Actual", test.set_index('ds')['y'], preds, COLOR_LSTM, "LSTM")
            st.metric("RMSE", f"{rmse:.2f}")
        except Exception as e:
            st.error(f"LSTM model error: {e}")

    elif model_type == "Comparison":
        st.markdown(f"<h3 style='font-family:{FONT_FAMILY}; font-weight:bold;'>{chr(0x1F4CA)} Model Performance Comparison (Lower RMSE is Better)</h3>", unsafe_allow_html=True)
        try:
            preds_arima, rmse_arima = run_arima(train, test)
        except Exception as e:
            st.error(f"ARIMA model error: {e}")
            preds_arima, rmse_arima = None, None
        try:
            preds_prophet, rmse_prophet = run_prophet(train, test)
        except Exception as e:
            st.error(f"Prophet model error: {e}")
            preds_prophet, rmse_prophet = None, None
        try:
            preds_lstm, rmse_lstm = run_lstm(train, test)
        except Exception as e:
            st.error(f"LSTM model error: {e}")
            preds_lstm, rmse_lstm = None, None

        # Line chart comparison
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Actual', line=dict(color=COLOR_ACTUAL)))
        if preds_arima is not None:
            fig.add_trace(go.Scatter(x=test['ds'], y=preds_arima, mode='lines', name='ARIMA', line=dict(color=COLOR_ARIMA)))
        if preds_prophet is not None:
            fig.add_trace(go.Scatter(x=test['ds'], y=preds_prophet, mode='lines', name='Prophet', line=dict(color=COLOR_PROPHET)))
        if preds_lstm is not None:
            fig.add_trace(go.Scatter(x=test['ds'], y=preds_lstm, mode='lines', name='LSTM', line=dict(color=COLOR_LSTM)))
        fig.update_layout(
            title=dict(text="Model Forecast Comparison", font=dict(family=FONT_FAMILY, size=TITLE_FONT_SIZE, color="black"), x=0.5),
            xaxis_title=dict(text="Date", font=dict(family=FONT_FAMILY, size=AXIS_TITLE_FONT_SIZE, color="black")),
            yaxis_title=dict(text="Price", font=dict(family=FONT_FAMILY, size=AXIS_TITLE_FONT_SIZE, color="black")),
            font=dict(family=FONT_FAMILY, size=TICK_FONT_SIZE, color="black"),
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis=dict(showgrid=True, gridcolor="lightgray"),
            yaxis=dict(showgrid=True, gridcolor="lightgray")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart comparison
        fig_bar, ax = plt.subplots(figsize=(8, 5))
        model_names = []
        rmses = []
        colors = []
        if preds_arima is not None:
            model_names.append("ARIMA")
            rmses.append(rmse_arima)
            colors.append(COLOR_BAR_ARIMA)
        if preds_prophet is not None:
            model_names.append("Prophet")
            rmses.append(rmse_prophet)
            colors.append(COLOR_BAR_PROPHET)
        if preds_lstm is not None:
            model_names.append("LSTM")
            rmses.append(rmse_lstm)
            colors.append(COLOR_BAR_LSTM)
        bars = ax.bar(model_names, rmses, color=colors)
        ax.set_ylabel("RMSE", fontsize=AXIS_TITLE_FONT_SIZE, fontname=FONT_FAMILY)
        ax.set_title("Model Comparison", fontsize=TITLE_FONT_SIZE, fontweight='bold', fontname=FONT_FAMILY)
        ax.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
        ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname(FONT_FAMILY)
        st.pyplot(fig_bar)
