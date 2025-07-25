import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from prophet import Prophet
import datetime
import os

# Streamlit page setup
st.set_page_config(page_title="📈 Stock Forecasting", layout="wide")
st.title("📈 Stock Price Forecasting Dashboard")

# Date settings
today = datetime.date.today()
ticker = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2018, 1, 1))
end_date = st.date_input("End Date", datetime.date(2024, 12, 31))
model_type = st.selectbox("Model", ["ARIMA", "LSTM", "Prophet", "Comparison"])

# Warning for large date ranges
if (end_date - start_date).days > 365 * 5:
    st.warning("⚠️ Date range is more than 5 years. This may slow down LSTM training.")

# Data loader
@st.cache_data(show_spinner=True)
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)[["Close"]]
        df.columns = ["Close"]
        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("No data fetched.")
        return df
    except Exception:
        fallback = "sample_aapl.csv"
        if os.path.exists(fallback):
            df = pd.read_csv(fallback, parse_dates=["Date"], index_col="Date")
            df.columns = ["Close"]
            st.info("📁 Loaded fallback sample_aapl.csv")
            return df
        else:
            st.error("❌ No data and no fallback CSV found.")
            return pd.DataFrame()

safe_end_date = min(end_date, today)
data = load_data(ticker, start_date, safe_end_date)

if data.empty:
    st.stop()

# Plot historical prices
st.subheader("📉 Historical Closing Prices")
st.plotly_chart(px.line(data.reset_index(), x="Date", y="Close", title=f"{ticker} Closing Price"))

# Prepare dataset
df = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# --- Model Functions ---
def run_arima(train, test):
    ts_train = train["y"]
    model = ARIMA(ts_train, order=(5, 1, 0)).fit()
    forecast = model.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test["y"], forecast))
    return forecast, rmse

def run_lstm(train, test):
    series = np.concatenate([train["y"].values, test["y"].values]).reshape(-1, 1)
    gen = TimeseriesGenerator(series, series, length=10, batch_size=1)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(10, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(gen, epochs=5, verbose=0)

    predictions = []
    curr_batch = series[-len(test)-10:-len(test)].reshape(1, 10, 1)

    for _ in range(len(test)):
        pred = model.predict(curr_batch, verbose=0)[0]
        predictions.append(pred[0])
        curr_batch = np.append(curr_batch[:, 1:, :], [[pred]], axis=1)

    rmse = np.sqrt(mean_squared_error(test["y"], predictions))
    return predictions, rmse

def run_prophet(train, test):
    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    preds = forecast["yhat"][-len(test):].values
    rmse = np.sqrt(mean_squared_error(test["y"], preds))
    return preds, rmse

# --- Forecast Visualization ---
if model_type == "ARIMA":
    st.subheader("🔮 ARIMA Forecast")
    preds, rmse = run_arima(train, test)
    forecast_df = test.copy()
    forecast_df["Forecast"] = preds
    fig = px.line(
        forecast_df, x="ds", y=["y", "Forecast"],
        title="ARIMA Forecast vs Actual",
        labels={"value": "Price", "ds": "Date"},
        color_discrete_map={"y": "blue", "Forecast": "orange"}
    )
    st.plotly_chart(fig)
    st.metric("RMSE", f"{rmse:.2f}")

elif model_type == "LSTM":
    st.subheader("🔮 LSTM Forecast")
    preds, rmse = run_lstm(train, test)
    forecast_df = test.copy()
    forecast_df["Forecast"] = preds
    fig = px.line(
        forecast_df, x="ds", y=["y", "Forecast"],
        title="LSTM Forecast vs Actual",
        labels={"value": "Price", "ds": "Date"},
        color_discrete_map={"y": "blue", "Forecast": "green"}
    )
    st.plotly_chart(fig)
    st.metric("RMSE", f"{rmse:.2f}")

elif model_type == "Prophet":
    st.subheader("🔮 Prophet Forecast")
    preds, rmse = run_prophet(train, test)
    forecast_df = test.copy()
    forecast_df["Forecast"] = preds
    fig = px.line(
        forecast_df, x="ds", y=["y", "Forecast"],
        title="Prophet Forecast vs Actual",
        labels={"value": "Price", "ds": "Date"},
        color_discrete_map={"y": "blue", "Forecast": "red"}
    )
    st.plotly_chart(fig)
    st.metric("RMSE", f"{rmse:.2f}")

# --- Comparison Mode ---
elif model_type == "Comparison":
    st.subheader("🔀 Model Comparison")

    # Use the RMSE values from your report
    rmse_arima = 11.87
    rmse_prophet = 10.34
    rmse_lstm = 8.65

    # RMSE Bar Chart
    fig_bar = go.Figure(data=[
        go.Bar(name='ARIMA', x=["ARIMA"], y=[rmse_arima], marker_color='lightblue'),
        go.Bar(name='Prophet', x=["Prophet"], y=[rmse_prophet], marker_color='lightgreen'),
        go.Bar(name='LSTM', x=["LSTM"], y=[rmse_lstm], marker_color='salmon')
    ])
    fig_bar.update_layout(
        title="RMSE Comparison (Lower is Better)",
        yaxis_title="RMSE",
        xaxis_title="Model",
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Metrics
    st.metric("ARIMA RMSE", f"{rmse_arima:.2f}")
    st.metric("Prophet RMSE", f"{rmse_prophet:.2f}")
    st.metric("LSTM RMSE", f"{rmse_lstm:.2f}")
