import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
from prophet import Prophet
import datetime
import os

# Page setup
st.set_page_config(page_title="📈 Stock Forecasting", layout="wide")
st.title("📈 Stock Price Forecasting Dashboard")

# Date inputs
today = datetime.date.today()
three_years_ago = today - datetime.timedelta(days=3 * 365)

ticker = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2018, 1, 1))
end_date = st.date_input("End Date", datetime.date(2024, 12, 31))
model_type = st.selectbox("Model", ["ARIMA", "LSTM", "Prophet", "Comparison"])

if (end_date - start_date).days > 365 * 5:
    st.warning("⚠️ Long date ranges may slow forecasting, especially with LSTM.")

@st.cache_data(show_spinner=True)
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)[["Close"]]
        df.columns = ["Close"]
        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("Empty data")
        return df
    except:
        fallback = "sample_aapl.csv"
        if os.path.exists(fallback):
            df = pd.read_csv(fallback, parse_dates=["Date"], index_col="Date")
            df.columns = ["Close"]
            st.info("Loaded fallback CSV data.")
            return df
        else:
            st.error("Failed to fetch data and fallback CSV not found.")
            return pd.DataFrame()

safe_end_date = min(end_date, today)
data = load_data(ticker, start_date, safe_end_date)

if data.empty:
    st.stop()

# Plot raw data
st.subheader("📊 Historical Closing Price")
st.plotly_chart(px.line(data.reset_index(), x="Date", y="Close", title=f"{ticker} Closing Price"))

# Prepare DataFrame for models
df = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# ARIMA model
def run_arima(train, test):
    try:
        ts_train = train["y"]
        model = ARIMA(ts_train, order=(5, 1, 0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test["y"], forecast))
        return forecast, rmse
    except Exception as e:
        st.error(f"ARIMA failed: {e}")
        return np.array([]), None

# LSTM model
def run_lstm(train, test):
    try:
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
    except Exception as e:
        st.error(f"LSTM failed: {e}")
        return np.array([]), None

# Prophet model
def run_prophet(train, test):
    try:
        model = Prophet()
        model.fit(train)
        future = model.make_future_dataframe(periods=len(test), freq='D')
        forecast = model.predict(future)
        preds = forecast['yhat'][-len(test):].values
        rmse = np.sqrt(mean_squared_error(test["y"], preds))
        return preds, rmse
    except Exception as e:
        st.error(f"Prophet failed: {e}")
        return np.array([]), None

# Single model forecast
if model_type != "Comparison":
    if model_type == "ARIMA":
        st.subheader("🔮 ARIMA Forecast")
        preds, rmse = run_arima(train, test)
    elif model_type == "LSTM":
        st.subheader("🔮 LSTM Forecast")
        preds, rmse = run_lstm(train, test)
    elif model_type == "Prophet":
        st.subheader("🔮 Prophet Forecast")
        preds, rmse = run_prophet(train, test)

    if len(preds) > 0:
        forecast_df = test.copy()
        forecast_df["Forecast"] = preds

        st.metric("RMSE", f"{rmse:.2f}")

        # Bar chart for forecast vs actual
        melted = forecast_df[["ds", "y", "Forecast"]].melt(id_vars="ds", value_name="Price", var_name="Type")
        fig_bar = px.bar(melted, x="ds", y="Price", color="Type", barmode="group",
                         title=f"{model_type} Forecast vs Actual", labels={"ds": "Date"})
        st.plotly_chart(fig_bar, use_container_width=True)

# Comparison mode
else:
    st.subheader("🔍 Comparing All Models")
    preds_arima, rmse_arima = run_arima(train, test)
    preds_lstm, rmse_lstm = run_lstm(train, test)
    preds_prophet, rmse_prophet = run_prophet(train, test)

    comp_df = test.copy()
    comp_df["ARIMA"] = preds_arima
    comp_df["LSTM"] = preds_lstm
    comp_df["Prophet"] = preds_prophet

    fig = px.line(comp_df, x="ds", y=["y", "ARIMA", "LSTM", "Prophet"],
                  title="Forecast vs Actual by Model", labels={"value": "Price", "ds": "Date"})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("ARIMA RMSE", f"{rmse_arima:.2f}")
    col2.metric("LSTM RMSE", f"{rmse_lstm:.2f}")
    col3.metric("Prophet RMSE", f"{rmse_prophet:.2f}")
