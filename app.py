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
import matplotlib.pyplot as plt
import datetime
import os

# Streamlit page config
st.set_page_config(page_title="📈 Stock Forecasting", layout="wide")
st.title("📈 Stock Price Forecasting Dashboard")

# Date selection
today = datetime.date.today()
three_years_ago = today - datetime.timedelta(days=3 * 365)
ticker = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2018, 1, 1))
end_date = st.date_input("End Date", datetime.date(2024, 12, 31))
model_type = st.selectbox("Model", ["ARIMA", "LSTM", "Prophet", "Comparison"])

if (end_date - start_date).days > 365 * 5:
    st.warning("⚠️ Date range is more than 5 years. This may slow forecasting, especially with LSTM.")

@st.cache_data(show_spinner=True)
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)[["Close"]]
        df.columns = ["Close"]
        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("No data fetched from yfinance.")
        return df
    except Exception:
        fallback_path = os.path.join(os.getcwd(), "sample_aapl.csv")
        if os.path.exists(fallback_path):
            df = pd.read_csv(fallback_path, parse_dates=["Date"], index_col="Date")
            df.columns = ["Close"]
            st.info("ℹ️ Loaded fallback data from sample_aapl.csv.")
            return df
        else:
            st.error("❌ Failed to fetch data and fallback CSV not found.")
            return pd.DataFrame()

safe_end_date = min(end_date, today)
data = load_data(ticker, start_date, safe_end_date)

if data.empty:
    st.stop()

# Plot Closing Price
st.subheader("📈 Closing Price Chart")
fig = px.line(data.reset_index(), x="Date", y="Close", title=f"{ticker} Closing Price")
st.plotly_chart(fig)

# Prepare data
df = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# --- Model Runners ---

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

def run_prophet(train, test):
    try:
        model = Prophet()
        model.fit(train)
        future = model.make_future_dataframe(periods=len(test), freq='D')
        forecast = model.predict(future)
        preds = forecast['yhat'][-len(test):].values
        rmse = np.sqrt(mean_squared_error(test['y'], preds))
        return preds, rmse
    except Exception as e:
        st.error(f"Prophet model failed: {e}")
        return np.array([]), None

# --- Model Execution ---

if model_type == "ARIMA":
    st.subheader("🔮 ARIMA Forecast")
    preds, rmse = run_arima(train, test)
elif model_type == "LSTM":
    st.subheader("🔮 LSTM Forecast")
    preds, rmse = run_lstm(train, test)
elif model_type == "Prophet":
    st.subheader("🔮 Prophet Forecast")
    preds, rmse = run_prophet(train, test)

if model_type != "Comparison":
    forecast_df = test.copy()
    forecast_df["Forecast"] = preds
    fig2 = px.line(forecast_df, x="ds", y=["y", "Forecast"],
                   labels={"value": "Price", "ds": "Date"},
                   title=f"{model_type} Forecast vs Actual")
    st.plotly_chart(fig2)

# --- Model Comparison ---

if model_type == "Comparison":
    st.subheader("🔮 Model Comparison Forecast")
    preds_arima, rmse_arima = run_arima(train, test)
    preds_lstm, rmse_lstm = run_lstm(train, test)
    preds_prophet, rmse_prophet = run_prophet(train, test)

    comp_df = test.copy()
    comp_df["ARIMA"] = preds_arima
    comp_df["LSTM"] = preds_lstm
    comp_df["Prophet"] = preds_prophet

    fig_comp = px.line(
        comp_df,
        x="ds",
        y=["y", "ARIMA", "LSTM", "Prophet"],
        labels={"value": "Price", "ds": "Date"},
        title="Model Comparison Forecast vs Actual"
    )
    st.plotly_chart(fig_comp)

    st.metric("ARIMA RMSE", f"{rmse_arima:.2f}")
    st.metric("LSTM RMSE", f"{rmse_lstm:.2f}")
    st.metric("Prophet RMSE", f"{rmse_prophet:.2f}")

    # --- Matplotlib RMSE Comparison Bar Chart ---
    st.markdown("### 📊 RMSE Comparison Across Models")
    rmse_values = {
        "ARIMA": rmse_arima,
        "Prophet": rmse_prophet,
        "LSTM": rmse_lstm
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['red', 'orange', 'purple']
    ax.bar(rmse_values.keys(), rmse_values.values(), color=colors)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14)
    ax.set_ylim(0, max(rmse_values.values()) + 10)

    for i, (model, value) in enumerate(rmse_values.items()):
        ax.text(i, value + 2, f"{value:.1f}", ha='center', va='bottom', fontsize=10)

    st.pyplot(fig)
