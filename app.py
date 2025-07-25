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
from prophet import Prophet
import datetime
import os

st.set_page_config(page_title="📈 Stock Forecasting", layout="wide")
st.title(":chart_with_upwards_trend: Stock Price Forecasting Dashboard")

# Date Range Inputs
today = datetime.date.today()
ticker = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2018, 1, 1))
end_date = st.date_input("End Date", datetime.date(2024, 12, 31))
model_type = st.selectbox("Model", ["ARIMA", "LSTM", "Prophet", "Comparison"])

if (end_date - start_date).days > 365 * 5:
    st.warning("⚠️ Date range is more than 5 years. This may slow forecasting.")

@st.cache_data(show_spinner=True)
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)[["Close"]]
        df.dropna(inplace=True)
        df.columns = ["Close"]
        if df.empty:
            raise ValueError("No data fetched")
        return df
    except:
        sample_path = os.path.join(os.getcwd(), "sample_aapl.csv")
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path, parse_dates=["Date"], index_col="Date")
            df.columns = ["Close"]
            st.info("Loaded sample fallback data.")
            return df
        else:
            st.error("Failed to load data.")
            return pd.DataFrame()

safe_end_date = min(end_date, today)
data = load_data(ticker, start_date, safe_end_date)

if data.empty:
    st.stop()

# Line Chart of Closing Price
st.subheader("📈 Historical Closing Price")
st.plotly_chart(px.line(data.reset_index(), x="Date", y="Close", title=f"{ticker} Closing Price"))

# Prep data for models
df = data.reset_index().rename(columns={"Date": "ds", "Close": "y"})
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# ARIMA Forecast
def run_arima(train, test):
    try:
        model = ARIMA(train["y"], order=(5, 1, 0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test["y"], forecast))
        return forecast.values, rmse
    except Exception as e:
        st.error(f"ARIMA error: {e}")
        return [], None

# LSTM Forecast
def run_lstm(train, test):
    try:
        series = np.concatenate([train["y"].values, test["y"].values]).reshape(-1, 1)
        gen = TimeseriesGenerator(series, series, length=10, batch_size=1)
        model = Sequential([LSTM(50, activation='relu', input_shape=(10, 1)), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.fit(gen, epochs=5, verbose=0)
        predictions = []
        curr = series[-len(test)-10:-len(test)].reshape(1, 10, 1)
        for _ in range(len(test)):
            pred = model.predict(curr, verbose=0)[0]
            predictions.append(pred[0])
            curr = np.append(curr[:, 1:, :], [[pred]], axis=1)
        rmse = np.sqrt(mean_squared_error(test["y"], predictions))
        return predictions, rmse
    except Exception as e:
        st.error(f"LSTM error: {e}")
        return [], None

# Prophet Forecast
def run_prophet(train, test):
    try:
        model = Prophet()
        model.fit(train)
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)
        preds = forecast["yhat"].iloc[-len(test):].values
        rmse = np.sqrt(mean_squared_error(test["y"], preds))
        return preds, rmse
    except Exception as e:
        st.error(f"Prophet error: {e}")
        return [], None

# Forecast Execution
if model_type != "Comparison":
    if model_type == "ARIMA":
        st.subheader("🔴 ARIMA Forecast")
        preds, rmse = run_arima(train, test)
    elif model_type == "LSTM":
        st.subheader("🟣 LSTM Forecast")
        preds, rmse = run_lstm(train, test)
    elif model_type == "Prophet":
        st.subheader("🟠 Prophet Forecast")
        preds, rmse = run_prophet(train, test)

    if preds:
        forecast_df = test.copy()
        forecast_df["Forecast"] = preds
        color = {"ARIMA": "red", "LSTM": "purple", "Prophet": "orange"}[model_type]
        fig = px.line(forecast_df, x="ds", y=["y", "Forecast"],
                      color_discrete_map={"y": "black", "Forecast": color},
                      labels={"value": "Price", "ds": "Date"},
                      title=f"{model_type} Forecast vs Actual")
        st.plotly_chart(fig)
        st.metric("RMSE", f"{rmse:.2f}")

# Comparison Mode
if model_type == "Comparison":
    st.subheader("📊 Model Comparison Forecast")
    arima_preds, arima_rmse = run_arima(train, test)
    lstm_preds, lstm_rmse = run_lstm(train, test)
    prophet_preds, prophet_rmse = run_prophet(train, test)

    compare_df = test.copy()
    compare_df["ARIMA"] = arima_preds
    compare_df["LSTM"] = lstm_preds
    compare_df["Prophet"] = prophet_preds

    # Line plot
    fig1 = px.line(compare_df, x="ds", y=["y", "ARIMA", "LSTM", "Prophet"],
                   labels={"value": "Price", "ds": "Date"},
                   color_discrete_map={"y": "black", "ARIMA": "red", "LSTM": "purple", "Prophet": "orange"},
                   title="Model Forecast vs Actual")
    st.plotly_chart(fig1)

    # Bar chart RMSE
    st.subheader("📉 RMSE Comparison")
    rmse_df = pd.DataFrame({
        "Model": ["ARIMA", "LSTM", "Prophet"],
        "RMSE": [arima_rmse, lstm_rmse, prophet_rmse],
        "Color": ["red", "purple", "orange"]
    })
    fig2 = px.bar(rmse_df, x="Model", y="RMSE", color="Model",
                  color_discrete_map={"ARIMA": "red", "LSTM": "purple", "Prophet": "orange"},
                  title="Root Mean Square Error by Model")
    st.plotly_chart(fig2)
