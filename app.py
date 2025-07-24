import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.express as px
import warnings
import time
import os
import sys

st.set_page_config(page_title="📈 Stock Forecasting Dashboard", layout="wide")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
st.write("Python version:", sys.version)
st.title("📈 Stock Price Forecasting Dashboard")

ticker = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
model_choice = st.sidebar.selectbox("Model", ["ARIMA", "Prophet", "LSTM", "Compare All"])

if (end_date - start_date).days > 1825:
    st.warning("⚠️ Date range is more than 5 years. This may slow forecasting, especially with LSTM.")

st.markdown(f"📅 Fetching data for: `{ticker}` from `{start_date}` to `{end_date}`")

@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close'].to_frame()
            df.columns = ['Close']
        elif 'Close' not in df.columns:
            raise ValueError("Missing 'Close' column")
        return df
    except Exception:
        try:
            st.info("✅ Loaded fallback data from sample_aapl.csv")
            df = pd.read_csv("sample_aapl.csv", parse_dates=["Date"])
            df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
            df.set_index("Date", inplace=True)
            return df
        except Exception:
            return pd.DataFrame()

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("❌ No data found. Please check stock symbol and date range.")
    st.stop()

st.subheader("📈 Closing Price Chart")
st.plotly_chart(px.line(data, x=data.index, y="Close", title=f"{ticker} Closing Price"))

df = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

@st.cache_resource(show_spinner=False)
def run_prophet(train, test):
    model = Prophet(daily_seasonality=True)
    model.fit(train)
    future = model.make_future_dataframe(periods=len(test), freq="B")
    forecast = model.predict(future)
    preds = forecast["yhat"][-len(test):]
    rmse = np.sqrt(mean_squared_error(test["y"], preds))
    return preds, rmse

@st.cache_resource(show_spinner=False)
def run_arima(train, test):
    ts_train = train["y"].astype(float)
    model = ARIMA(ts_train, order=(5, 1, 0))
    model_fit = model.fit()
    preds = model_fit.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test["y"], preds))
    return preds, rmse

def create_sequences(arr, ts=60):
    X, y = [], []
    for i in range(ts, len(arr)):
        X.append(arr[i - ts:i, 0])
        y.append(arr[i, 0])
    return np.array(X), np.array(y)

@st.cache_resource(show_spinner=False)
def run_lstm(df, train_size, ts=60):
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(df[["y"]])
    train_arr, test_arr = arr[:train_size], arr[train_size:]
    if len(train_arr) < ts+1 or len(test_arr) < ts+1:
        return np.array([]), float("inf")
    X_train, y_train = create_sequences(train_arr, ts)
    X_test, y_test = create_sequences(test_arr, ts)
    if X_train.size == 0 or X_test.size == 0:
        return np.array([]), float("inf")
    X_train = X_train.reshape(-1, ts, 1)
    X_test = X_test.reshape(-1, ts, 1)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(ts, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile("adam", "mean_squared_error")
    model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
    preds = scaler.inverse_transform(model.predict(X_test))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(actual, preds))
    return preds, rmse

start = time.time()
with st.spinner("🔮 Forecasting..."):
    if model_choice == "Prophet":
        preds, rmse = run_prophet(train, test)
        st.subheader("📊 Prophet Results")
        st.write(f"RMSE: {rmse:.2f}")
        st.line_chart(pd.DataFrame({"Actual": test["y"], "Predicted": preds}, index=test["ds"]))

    elif model_choice == "ARIMA":
        preds, rmse = run_arima(train, test)
        st.subheader("📊 ARIMA Results")
        st.write(f"RMSE: {rmse:.2f}")
        st.line_chart(pd.DataFrame({"Actual": test["y"], "Predicted": preds}, index=test["ds"]))

    elif model_choice == "LSTM":
        preds, rmse = run_lstm(df, train_size)
        if rmse != float("inf"):
            st.subheader("📊 LSTM Results")
            st.write(f"RMSE: {rmse:.2f}")
            st.line_chart(pd.DataFrame({
                "Actual": df["y"][train_size+60:].values,
                "Predicted": preds.flatten()
            }, index=df["ds"][train_size+60:]))

    else:
        p1, r1 = run_prophet(train, test)
        p2, r2 = run_arima(train, test)
        p3, r3 = run_lstm(df, train_size)
        st.subheader("📊 RMSE Comparison")
        st.bar_chart(pd.Series({"Prophet": r1, "ARIMA": r2, "LSTM": r3}))

if time.time() - start > 45:
    st.warning("⚠️ Forecasting took a long time. Try a smaller range.")
