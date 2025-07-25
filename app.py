import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from prophet import Prophet
import datetime

# Page config
st.set_page_config(page_title="📈 Stock Price Forecasting Dashboard", layout="wide")
st.title("📈 Stock Price Forecasting Dashboard")
st.markdown("This dashboard displays ARIMA, Prophet, and LSTM forecasts for a selected stock.")

# Inputs
ticker = st.text_input("Enter Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.date_input("End Date", datetime.date(2024, 12, 31))

# Load data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df[["Date", "Close"]]

df = load_data(ticker, start_date, end_date)
st.subheader(f"📊 Data for {ticker} from {start_date} to {end_date}")
st.dataframe(df.tail())

# Plot close price
st.subheader(f"{ticker} Closing Price")
plt.figure(figsize=(10, 3))
plt.plot(df["Date"], df["Close"])
plt.xlabel("Date")
plt.ylabel("Close")
st.pyplot(plt)

# Train-test split
df = df.rename(columns={"Date": "ds", "Close": "y"})
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# --------- ARIMA ---------
def run_arima(train, test):
    model = ARIMA(train["y"], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test["y"], forecast))
    return forecast, rmse

st.subheader("📊 ARIMA Forecast")
arima_pred, arima_rmse = run_arima(train, test)
plt.figure(figsize=(10, 3))
plt.plot(train["ds"], train["y"], label="Train", color="blue")
plt.plot(test["ds"], test["y"], label="Actual", color="orange")
plt.plot(test["ds"], arima_pred, label="ARIMA Forecast", color="green")
plt.legend()
st.pyplot(plt)
st.write(f"**RMSE**: {arima_rmse:.2f}")

# --------- LSTM ---------
def run_lstm(train, test):
    full_data = np.concatenate([train["y"], test["y"]])
    full_data = full_data.reshape(-1, 1)
    generator = TimeseriesGenerator(full_data, full_data, length=10, batch_size=1)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(10, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=5, verbose=0)

    test_seq = full_data[len(full_data)-len(test)-10:-len(test)].reshape(1, 10, 1)
    predictions = []
    for i in range(len(test)):
        pred = model.predict(test_seq, verbose=0)[0]
        predictions.append(pred[0])
        test_seq = np.append(test_seq[:, 1:, :], [[pred]], axis=1)

    rmse = np.sqrt(mean_squared_error(test["y"], predictions))
    return predictions, rmse

st.subheader("📊 LSTM Forecast")
lstm_pred, lstm_rmse = run_lstm(train, test)
plt.figure(figsize=(10, 3))
plt.plot(test["ds"], test["y"], label="Actual", color="blue")
plt.plot(test["ds"], lstm_pred, label="LSTM Forecast", color="orange")
plt.legend()
st.pyplot(plt)
st.write(f"**RMSE**: {lstm_rmse:.2f}")

# --------- Prophet ---------
def run_prophet(train, test):
    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(periods=len(test), freq='D')
    forecast = model.predict(future)
    preds = forecast[['ds', 'yhat']].tail(len(test))
    rmse = np.sqrt(mean_squared_error(test["y"], preds["yhat"]))
    return forecast, preds, rmse

st.subheader("📊 Prophet Forecast")
full_prophet, prophet_preds, prophet_rmse = run_prophet(train, test)
fig = Prophet().plot(full_prophet)
st.pyplot(fig)
st.write(f"**RMSE**: {prophet_rmse:.2f}")

# --------- Bar chart comparison ---------
st.subheader("📊 Model Performance Comparison (Lower RMSE is Better)")
models = ["ARIMA", "Prophet", "LSTM"]
rmses = [arima_rmse, prophet_rmse, lstm_rmse]
colors = ["red", "orange", "purple"]

plt.figure(figsize=(6, 4))
bars = plt.bar(models, rmses, color=colors)
plt.ylabel("RMSE")
plt.title("Model Comparison")
for bar, rmse in zip(bars, rmses):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{rmse:.2f}', ha='center', va='bottom')
st.pyplot(plt)
