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
import datetime
import os

# Set page
st.set_page_config(page_title="📈 Stock Forecasting", layout="wide")
st.title("📈 Stock Price Forecasting Dashboard")

# Default to last 3 years
today = datetime.date.today()
three_years_ago = today - datetime.timedelta(days=3 * 365)

ticker = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2018, 1, 1))
end_date = st.date_input("End Date", datetime.date(2024, 12, 31))
model_type = st.selectbox("Model", ["ARIMA", "LSTM"])

if (end_date - start_date).days > 365 * 5:
    st.warning("⚠️ Date range is more than 5 years. This may slow forecasting, especially with LSTM.")

@st.cache_data(show_spinner=True)
def load_data(ticker, start, end):
    """Load stock data from yfinance or fallback CSV."""
    try:
        df = yf.download(ticker, start=start, end=end)[["Close"]]
        df.columns = ["Close"]
        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("No data fetched from yfinance for the selected range.")
        return df
    except Exception:
        sample_path = os.path.join(os.getcwd(), "sample_aapl.csv")
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path, parse_dates=["Date"], index_col="Date")
            df.columns = ["Close"]
            st.info("ℹ️ Loaded fallback data from sample_aapl.csv due to data unavailability.")
            return df
        else:
            st.error("❌ Failed to fetch data and fallback CSV not found.")
            return pd.DataFrame()

safe_end_date = min(end_date, today)
data = load_data(ticker, start_date, safe_end_date)

if data.empty:
    st.error("❌ No historical data found for selected range.")
    st.stop()

# Plot closing price chart
st.subheader("📈 Closing Price Chart")
fig = px.line(data.reset_index(), x="Date", y="Close", title=f"{ticker} Closing Price")
st.plotly_chart(fig)

# Prepare for modeling
df = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Run Models with error handling and progress indication
def run_arima(train, test):
    try:
        ts_train = train["y"]
        model = ARIMA(ts_train, order=(5, 1, 0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test["y"], forecast))
        return forecast, rmse
    except Exception as e:
        st.error(f"ARIMA model failed: {e}")
        return np.array([]), None

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
elif model_type == "LSTM":
    st.subheader("🔮 LSTM Forecast")
    preds, rmse = run_lstm(train, test)

# Plot Forecast
forecast_df = test.copy()
forecast_df["Forecast"] = preds

fig2 = px.line(forecast_df, x="ds", y=["y", "Forecast"], labels={"value": "Price", "ds": "Date"}, title=f"{model_type} Forecast vs Actual")
st.plotly_chart(fig2)

st.metric("RMSE", f"{rmse:.2f}")
