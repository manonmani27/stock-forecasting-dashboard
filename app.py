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

# Default to last 3 years
today = datetime.date.today()
three_years_ago = today - datetime.timedelta(days=3 * 365)

ticker = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", three_years_ago)
end_date = st.date_input("End Date", today)
model_type = st.selectbox("Model", ["ARIMA", "LSTM"])

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

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.stop()

st.write("✅ Data shape:", data.shape)

# Plot
st.subheader("📈 Closing Price Chart")
fig = px.line(data.reset_index(), x="Date", y="Close", title=f"{ticker} Closing Price")
st.plotly_chart(fig)

# Prepare for modeling
df = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Debug test/train sizes
st.write("📊 Train size:", len(train), "Test size:", len(test))

# Run Models
def run_arima(train, test):
    try:
        ts_train = train["y"]
        model = ARIMA(ts_train, order=(5, 1, 0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test["y"], forecast))
        return forecast, rmse
    except Exception as e:
        st.error(f"❌ ARIMA error: {e}")
        return [], 0

def run_lstm(train, test):
    try:
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
    except Exception as e:
        st.error(f"❌ LSTM error: {e}")
        return [], 0

# Run and display
if model_type == "ARIMA":
    st.subheader("🔮 ARIMA Forecast")
    preds, rmse = run_arima(train, test)
else:
    st.subheader("🔮 LSTM Forecast")
    preds, rmse = run_lstm(train, test)

# Validate and plot
if not isinstance(preds, (list, np.ndarray)) or len(preds) == 0:
    st.error("❌ No forecast results to display.")
else:
    if len(preds) != len(test):
        st.warning(f"⚠️ Mismatched lengths: preds={len(preds)}, test={len(test)} — truncating.")
        min_len = min(len(preds), len(test))
        preds = preds[:min_len]
        test = test[:min_len]

    forecast_df = test.copy().reset_index(drop=True)
    forecast_df["Forecast"] = preds

    st.write("📈 Forecast Data Sample:")
    st.dataframe(forecast_df.head())

    fig2 = px.line(
        forecast_df,
        x="ds",
        y=["y", "Forecast"],
        labels={"value": "Price", "ds": "Date"},
        title=f"{model_type} Forecast vs Actual"
    )
    st.plotly_chart(fig2)
    st.metric("RMSE", f"{rmse:.2f}")
