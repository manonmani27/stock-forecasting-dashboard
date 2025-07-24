# 📈 Stock Forecasting Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

An interactive dashboard for forecasting stock prices using **ARIMA**, **LSTM**, and optionally **Prophet** models. Built with Python and Streamlit during my internship at **Zidio**.

---

## 📌 Features

- 💡 Input any stock symbol (e.g., AAPL, TSLA, GOOGL)
- 📅 Select custom date ranges
- 🔮 Choose from ARIMA, LSTM, or Compare Both
- 📊 RMSE comparison of model performance
- 📈 Visualize actual vs predicted closing prices
- 🔄 Real-time data from Yahoo Finance
- ☁️ Deployable via Streamlit Cloud

---

## 🛠 Tech Stack

- **Python**
- **Streamlit** – UI and deployment
- **yFinance** – real-time stock data
- **ARIMA** – `statsmodels` for classical time series modeling
- **LSTM** – deep learning with TensorFlow/Keras
- **Plotly** – interactive charts
- *(Optional: Prophet – for advanced forecasting)*

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/manonmani27/stock-forecasting-dashboard.git
cd stock-forecasting-dashboard

# Create and activate virtual environment (optional)
python -m venv .venv
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
