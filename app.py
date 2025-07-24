import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.express as px
import warnings
import time

warnings.filterwarnings("ignore")

# Streamlit config
st.set_page_config(page_title="📈 Stock Forecasting Dashboard", layout="wide")
st.title("📈 Stock Price Forecasting Dashboard")

# Sidebar Inputs
ticker = st.sidebar.text_input("Stock Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
model_choice = st.sidebar.selectbox("Model", ["ARIMA", "Prophet", "LSTM", "Compare All"])

# Load stock data
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.error("❌ No data found for this symbol and date range.")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

st.subheader("📈 Closing Price Chart")
st.plotly_chart(px.line(data, x=data.index, y="Close", title=f"{ticker} Closing Price"))

# Prepare data
df = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Prophet
def run_prophet(train, test):
    model = Prophet(daily_seasonality=True)
    model.fit(train)
    future = model.make_future_dataframe(periods=len(test), freq='B')
    forecast = model.predict(future)
    predicted = forecast["yhat"][-len(test):]
    rmse = np.sqrt(mean_squared_error(test["y"], predicted))
    return predicted, rmse, forecast

# ARIMA
def run_arima(train, test):
    train_series = train["y"].astype(float)
    test_series = test["y"].astype(float)
    model = ARIMA(train_series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_series))
    rmse = np.sqrt(mean_squared_error(test_series, forecast))
    return forecast, rmse

# LSTM helpers
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def run_lstm(df, train_size, time_step=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["y"]])
    train_scaled, test_scaled = scaled[:train_size], scaled[train_size:]

    if len(train_scaled) < time_step + 1 or len(test_scaled) < time_step + 1:
        st.error("❌ Not enough data for LSTM model. Try a longer date range.")
        return np.array([]), float("inf")

    X_train, y_train = create_sequences(train_scaled, time_step)
    X_test, y_test = create_sequences(test_scaled, time_step)

    if X_train.size == 0 or X_test.size == 0:
        st.error("❌ Failed to create training sequences for LSTM.")
        return np.array([]), float("inf")

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    pred_scaled = model.predict(X_test)
    pred = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(actual, pred))
    return pred, rmse

# Run selected model
start_time = time.time()

with st.spinner("🔮 Running selected forecasting model..."):

    if model_choice == "Prophet":
        predicted, rmse_prophet, _ = run_prophet(train, test)
        st.subheader("📊 Prophet Forecast Results")
        st.write(f"📉 RMSE: {rmse_prophet:.2f}")
        st.line_chart(pd.DataFrame({"Actual": test["y"].values, "Predicted": predicted.values}, index=test["ds"]))

    elif model_choice == "ARIMA":
        forecast, rmse_arima = run_arima(train, test)
        st.subheader("📊 ARIMA Forecast Results")
        st.write(f"📉 RMSE: {rmse_arima:.2f}")
        st.line_chart(pd.DataFrame({"Actual": test["y"].values, "Predicted": forecast.values}, index=test["ds"]))

    elif model_choice == "LSTM":
        pred_lstm, rmse_lstm = run_lstm(df, train_size)
        if rmse_lstm != float("inf"):
            st.subheader("📊 LSTM Forecast Results")
            st.write(f"📉 RMSE: {rmse_lstm:.2f}")
            st.line_chart(pd.DataFrame({
                "Actual": df["y"].values[train_size + 60:],
                "Predicted": pred_lstm.flatten()
            }))

    elif model_choice == "Compare All":
        predicted, rmse_prophet, _ = run_prophet(train, test)
        forecast, rmse_arima = run_arima(train, test)
        pred_lstm, rmse_lstm = run_lstm(df, train_size)

        st.subheader("📊 RMSE Comparison Across Models")
        rmse_df = pd.DataFrame({
            "Model": ["ARIMA", "Prophet", "LSTM"],
            "RMSE": [rmse_arima, rmse_prophet, rmse_lstm]
        })
        st.write(rmse_df)
        st.bar_chart(rmse_df.set_index("Model"))

# Notify if long run time
if time.time() - start_time > 45:
    st.warning("⚠️ Forecasting took longer than expected. Try a shorter date range or simpler model.")
