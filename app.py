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

# Try importing Prophet
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False
    st.warning("⚠️ Prophet library is not installed.")

# Set page
st.set_page_config(page_title="📈 Stock Forecasting", layout="wide")
st.title("📈 Stock Price Forecasting Dashboard")

# Default to last 3 years
today = datetime.date.today()
three_years_ago = today - datetime.timedelta(days=3 * 365)

ticker = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", three_years_ago)
end_date = st.date_input("End Date", today)

# Prevent future end date
if end_date > today:
    st.warning("⚠️ End date cannot be in the future. Resetting to today.")
    end_date = today

model_type = st.selectbox("Model", ["ARIMA", "LSTM", "Prophet"])

# Prevent large forecast durations
if (end_date - start_date).days > 365 * 3:
    st.warning("⚠️ Forecasting with more than 3 years of data may be slow, especially for LSTM.")

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

safe_end_date = min(end_date, today)
data = load_data(ticker, start_date, safe_end_date)

if data.empty:
    st.error("❌ No historical data found for selected range.")
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
    for _ in range(len(test)):
        pred = model.predict(curr_batch, verbose=0)[0]
        predictions.append(pred[0])
        curr_batch = np.append(curr_batch[:, 1:, :], [[pred]], axis=1)
    rmse = np.sqrt(mean_squared_error(test["y"], predictions))
    return predictions, rmse

def run_prophet(train, test):
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(train)
    future = prophet_model.make_future_dataframe(periods=len(test), freq='B')
    forecast = prophet_model.predict(future)
    yhat = forecast["yhat"][-len(test):].values
    rmse = np.sqrt(mean_squared_error(test["y"].values, yhat))
    return yhat, rmse

# Main Forecast Execution
if model_type == "ARIMA":
    st.subheader("🔮 ARIMA Forecast")
    preds, rmse = run_arima(train, test)
elif model_type == "LSTM":
    st.subheader("🔮 LSTM Forecast")
    preds, rmse = run_lstm(train, test)
elif model_type == "Prophet" and prophet_available:
    st.subheader("🔮 Prophet Forecast")
    preds, rmse = run_prophet(train, test)

# Plot Main Forecast
if len(preds) != len(test):
    st.warning(f"⚠️ Forecast length mismatch. Truncating.")
    min_len = min(len(preds), len(test))
    preds = preds[:min_len]
    test = test[:min_len]

forecast_df = test.copy().reset_index(drop=True)
forecast_df["Forecast"] = preds

fig2 = px.line(
    forecast_df,
    x="ds",
    y=["y", "Forecast"],
    labels={"value": "Price", "ds": "Date"},
    title=f"{model_type} Forecast vs Actual"
)
st.plotly_chart(fig2)
st.metric("RMSE", f"{rmse:.2f}")

# Download Forecast CSV
csv = forecast_df.to_csv(index=False).encode()
st.download_button(
    label="📥 Download Forecast CSV",
    data=csv,
    file_name=f"{ticker}_{model_type}_forecast.csv",
    mime="text/csv"
)

# RMSE Comparison Chart
rmse_scores = {}

try:
    _, rmse_arima = run_arima(train, test)
    rmse_scores["ARIMA"] = rmse_arima
except:
    rmse_scores["ARIMA"] = None

if prophet_available:
    try:
        _, rmse_prophet = run_prophet(train, test)
        rmse_scores["Prophet"] = rmse_prophet
    except:
        rmse_scores["Prophet"] = None

try:
    _, rmse_lstm = run_lstm(train, test)
    rmse_scores["LSTM"] = rmse_lstm
except:
    rmse_scores["LSTM"] = None

# Plot RMSE comparison
rmse_valid = {k: v for k, v in rmse_scores.items() if v is not None}
if rmse_valid:
    st.subheader("📉 RMSE Comparison Across Models")
    rmse_df = pd.DataFrame(list(rmse_valid.items()), columns=["Model", "RMSE"])
    fig_rmse = px.bar(rmse_df, x="Model", y="RMSE", color="Model",
                      title="Model RMSE Comparison (Lower is Better)",
                      text_auto=".2f")
    st.plotly_chart(fig_rmse)
else:
    st.info("ℹ️ RMSE comparison could not be completed. Some models may have failed.")
