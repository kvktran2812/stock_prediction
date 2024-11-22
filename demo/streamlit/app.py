import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf
import datetime
import yfinance as yf
import torch
from model import StockModel


model = StockModel()
model.load_state_dict(torch.load("../../models/jingle_bells_v1.pth", weights_only=True))
model.eval()

dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=65), periods=65, freq="D")
ticker = yf.Ticker("AAPL")
data = ticker.history(interval="1d", period="3mo")
data = data.reset_index()
data = data.drop(["Date", "Dividends", "Stock Splits"], axis=1)
data = data.tail(65)
data["Date"] = dates
data.set_index("Date", inplace=True) 



st.title("Candlestick Chart with mplfinance")
st.sidebar.header("Candlestick Chart Settings")
chart_type = st.sidebar.selectbox("Chart Type", options=["Candlestick", "OHLC"])
volume_display = st.sidebar.checkbox("Show Volume", value=False)

# Plot candlestick or OHLC chart
if chart_type == "Candlestick":
    fig, ax = mpf.plot(data, type='candle', volume=volume_display, returnfig=True, style='charles')
else:
    fig, ax = mpf.plot(data, type='ohlc', volume=volume_display, returnfig=True, style='charles')


model_data = data.pct_change().dropna().to_numpy()
model_data = model_data.reshape(1, 64, 5)
model_data = torch.tensor(model_data, dtype=torch.float32)
prediction = model(model_data)
prediction = prediction.detach().numpy().reshape((8,))

future_x = np.arange(len(data.index), len(data.index) + 8)  # Extend x-axis for predictions
future_y = np.zeros((8,))  # Predictive trend prices

for i in range(8):
    future_y[i] = data["Close"][-1] * (1 + prediction[i])

ax[0].plot(future_x, future_y, label='Predictive Trend', linestyle='--', color='red')
ax[0].legend()  # Add a legend

# Display the chart in Streamlit
st.pyplot(fig)