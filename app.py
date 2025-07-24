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
import datetime
import os

# Set page
st.set_page_config(page_title="📈 Stock Forecasting", layout="wide")
st.title("📈 Stock Price Forecasting Dashboard")

# Inputs
ticker = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2018, 1, 1))
end_date = st.date_input("End Date", datetime.date(2024, 12, 31))
model_type = st.selectbox("Model", ["ARIMA", "LSTM"])

if (end_date - start_date).days > 365 * 5:
    st.warning("⚠️ Date range is more than 5 years. This may slow forecasting, especially with LSTM.")

# Fetch Data
@st.cache_data
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)[["Close"]]
        df.columns = ["Close"]
        df.dropna(inplace=True)
        return df
    except Exception:
        sample_path = os.path.join(os.getcwd(), "sample_aapl.csv")
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path, parse_dates=["Date"], index_col="Date")
            df.columns = ["Close"]
            st.success("✅ Loaded fallback data from sample_aapl.csv")
            return df
        else:
            st.error("❌ Failed to fetch data and fallback CSV not found.")
            return pd.DataFrame()

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.stop()

# Plot
st.subheader("📈 Closing Price Chart")
fig = px.line(data.reset_index(), x="Date", y="Close", title=f"{ticker} Closing Price")
st.plotly_chart(fig)

# Prepare for modeling
df = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Run Models
def run_arima(train, test):
    ts_train = train["y"]
    model = ARIMA(ts_train, order=(5, 1, 0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test["y"], forecast))
    return forecast, rmse

def run_lstm(train, test):
    series = np.concatenate([train["y"].values, test["y"].values])
    series = series.reshape(-1, 1)

    gen = TimeseriesGenerator(series, series, length=10, batch_size=1)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(10, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(gen, epochs=5, verbose=0)

    predictions = []
    curr_batch = series[-len(test)-10:-len(test)].reshape(1, 10, 1)

    for i in range(len(test)):
        pred = model.predict(curr_batch, verbose=0)[0]
        predictions.append(pred[0])
        curr_batch = np.append(curr_batch[:, 1:, :], [[pred]], axis=1)

    rmse = np.sqrt(mean_squared_error(test["y"], predictions))
    return predictions, rmse

# Run and display
if model_type == "ARIMA":
    st.subheader("🔮 ARIMA Forecast")
    preds, rmse = run_arima(train, test)
else:
    st.subheader("🔮 LSTM Forecast")
    preds, rmse = run_lstm(train, test)

# Plot Forecast
forecast_df = test.copy()
forecast_df["Forecast"] = preds

fig2 = px.line(forecast_df, x="ds", y=["y", "Forecast"], labels={"value": "Price", "ds": "Date"}, title=f"{model_type} Forecast vs Actual")
st.plotly_chart(fig2)

st.metric("RMSE", f"{rmse:.2f}")
