import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from prophet import Prophet
import datetime
import os

# Define consistent fonts and colors
FONT_FAMILY = "Arial, Helvetica, sans-serif"
TITLE_FONT_SIZE = 20
AXIS_TITLE_FONT_SIZE = 16
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14

COLOR_ACTUAL = "#1f77b4"  # blue
COLOR_ARIMA = "#2ca02c"   # green
COLOR_PROPHET = "#ff7f0e" # orange
COLOR_LSTM = "#9467bd"    # purple
COLOR_BAR_ARIMA = "red"
COLOR_BAR_PROPHET = "orange"
COLOR_BAR_LSTM = "purple"

st.set_page_config(page_title="\U0001F4C8 Stock Price Forecasting Dashboard", layout="wide")
st.markdown(f"<h1 style='font-family:{FONT_FAMILY}; font-weight:bold;'>{chr(0x1F4C8)} Stock Price Forecasting Dashboard</h1>", unsafe_allow_html=True)

# Sidebar inputs
ticker = st.text_input("Enter Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.date_input("End Date", datetime.date(2024, 12, 31))
model_type = st.selectbox("Select Model", ["ARIMA", "Prophet", "LSTM", "Comparison"])

# Load Data
@st.cache_data(show_spinner=True)
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df = df[["Close"]].dropna()
    df.reset_index(inplace=True)
    df.columns = ["ds", "y"]
    return df

data = load_data(ticker, start_date, end_date)
st.markdown(f"<h3 style='font-family:{FONT_FAMILY}; font-weight:bold;'>{chr(0x1F4C4)} Data for {ticker} from {start_date} to {end_date}</h3>", unsafe_allow_html=True)
st.dataframe(data.tail())

# Line chart for closing price
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Close', line=dict(color=COLOR_ACTUAL)))
fig1.update_layout(
    title=dict(text=f"{ticker} Closing Price", font=dict(family=FONT_FAMILY, size=TITLE_FONT_SIZE, color="black"), x=0.5),
    xaxis_title=dict(text="Date", font=dict(family=FONT_FAMILY, size=AXIS_TITLE_FONT_SIZE, color="black")),
    yaxis_title=dict(text="Close", font=dict(family=FONT_FAMILY, size=AXIS_TITLE_FONT_SIZE, color="black")),
    font=dict(family=FONT_FAMILY, size=TICK_FONT_SIZE, color="black"),
    plot_bgcolor="white",
    margin=dict(l=40, r=40, t=60, b=40),
    xaxis=dict(showgrid=True, gridcolor="lightgray"),
    yaxis=dict(showgrid=True, gridcolor="lightgray")
)
st.plotly_chart(fig1, use_container_width=True)

# Train-Test split
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Forecasting Functions
def run_arima(train, test):
    model = ARIMA(train['y'], order=(5, 1, 0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test['y'], forecast))
    return forecast, rmse

def run_prophet(train, test):
    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(periods=len(test), freq='D')
    forecast = model.predict(future)
    forecast = forecast[['ds', 'yhat']].set_index('ds').loc[test['ds']]
    rmse = np.sqrt(mean_squared_error(test['y'], forecast['yhat']))
    return forecast['yhat'], rmse

def run_lstm(train, test):
    series = pd.concat([train['y'], test['y']]).values.reshape(-1, 1)
    gen = TimeseriesGenerator(series, series, length=10, batch_size=1)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(10, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(gen, epochs=5, verbose=0)

    predictions = []
    curr_batch = series[train_size - 10:train_size].reshape(1, 10, 1)
    for i in range(len(test)):
        pred = model.predict(curr_batch, verbose=0)[0][0]
        predictions.append(pred)
        curr_batch = np.append(curr_batch[:, 1:, :], [[[pred]]], axis=1)

    rmse = np.sqrt(mean_squared_error(test['y'], predictions))
    return predictions, rmse

# Visualization logic
def plot_forecast(title, actual, predicted, model_color, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual, mode='lines', name='Actual', line=dict(color=COLOR_ACTUAL)))
    fig.add_trace(go.Scatter(x=actual.index, y=predicted, mode='lines', name=f'{model_name} Forecast', line=dict(color=model_color)))
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT_FAMILY, size=TITLE_FONT_SIZE, color="black"), x=0.5),
        xaxis_title=dict(text="Date", font=dict(family=FONT_FAMILY, size=AXIS_TITLE_FONT_SIZE, color="black")),
        yaxis_title=dict(text="Price", font=dict(family=FONT_FAMILY, size=AXIS_TITLE_FONT_SIZE, color="black")),
        font=dict(family=FONT_FAMILY, size=TICK_FONT_SIZE, color="black"),
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray")
    )
    st.plotly_chart(fig, use_container_width=True)

# Run selected model
if model_type == "ARIMA":
    st.markdown(f"<h3 style='font-family:{FONT_FAMILY}; font-weight:bold;'>{chr(0x1F4C8)} ARIMA Forecast</h3>", unsafe_allow_html=True)
    preds, rmse = run_arima(train, test)
    plot_forecast("ARIMA Forecast vs Actual", test.set_index('ds')['y'], preds, COLOR_ARIMA, "ARIMA")
    st.metric("RMSE", f"{rmse:.2f}")

elif model_type == "Prophet":
    st.markdown(f"<h3 style='font-family:{FONT_FAMILY}; font-weight:bold;'>{chr(0x1F916)} Prophet Forecast</h3>", unsafe_allow_html=True)
    preds, rmse = run_prophet(train, test)
    plot_forecast("Prophet Forecast vs Actual", test.set_index('ds')['y'], preds, COLOR_PROPHET, "Prophet")
    st.metric("RMSE", f"{rmse:.2f}")

elif model_type == "LSTM":
    st.markdown(f"<h3 style='font-family:{FONT_FAMILY}; font-weight:bold;'>{chr(0x1F916)} LSTM Forecast</h3>", unsafe_allow_html=True)
    preds, rmse = run_lstm(train, test)
    plot_forecast("LSTM Forecast vs Actual", test.set_index('ds')['y'], preds, COLOR_LSTM, "LSTM")
    st.metric("RMSE", f"{rmse:.2f}")

elif model_type == "Comparison":
    st.markdown(f"<h3 style='font-family:{FONT_FAMILY}; font-weight:bold;'>{chr(0x1F4CA)} Model Performance Comparison (Lower RMSE is Better)</h3>", unsafe_allow_html=True)
    preds_arima, rmse_arima = run_arima(train, test)
    preds_prophet, rmse_prophet = run_prophet(train, test)
    preds_lstm, rmse_lstm = run_lstm(train, test)

    # Line chart comparison
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Actual', line=dict(color=COLOR_ACTUAL)))
    fig.add_trace(go.Scatter(x=test['ds'], y=preds_arima, mode='lines', name='ARIMA', line=dict(color=COLOR_ARIMA)))
    fig.add_trace(go.Scatter(x=test['ds'], y=preds_prophet, mode='lines', name='Prophet', line=dict(color=COLOR_PROPHET)))
    fig.add_trace(go.Scatter(x=test['ds'], y=preds_lstm, mode='lines', name='LSTM', line=dict(color=COLOR_LSTM)))
    fig.update_layout(
        title=dict(text="Model Forecast Comparison", font=dict(family=FONT_FAMILY, size=TITLE_FONT_SIZE, color="black"), x=0.5),
        xaxis_title=dict(text="Date", font=dict(family=FONT_FAMILY, size=AXIS_TITLE_FONT_SIZE, color="black")),
        yaxis_title=dict(text="Price", font=dict(family=FONT_FAMILY, size=AXIS_TITLE_FONT_SIZE, color="black")),
        font=dict(family=FONT_FAMILY, size=TICK_FONT_SIZE, color="black"),
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bar chart comparison
    fig_bar, ax = plt.subplots(figsize=(8, 5))
    model_names = ["ARIMA", "Prophet", "LSTM"]
    rmses = [rmse_arima, rmse_prophet, rmse_lstm]
    colors = [COLOR_BAR_ARIMA, COLOR_BAR_PROPHET, COLOR_BAR_LSTM]
    bars = ax.bar(model_names, rmses, color=colors)
    ax.set_ylabel("RMSE", fontsize=AXIS_TITLE_FONT_SIZE, fontname=FONT_FAMILY)
    ax.set_title("Model Comparison", fontsize=TITLE_FONT_SIZE, fontweight='bold', fontname=FONT_FAMILY)
    ax.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname(FONT_FAMILY)
    st.pyplot(fig_bar)
