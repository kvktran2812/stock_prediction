import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf
import datetime
import yfinance as yf


dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=100), periods=100, freq="D")
ticker = yf.Ticker("AAPL")
data = ticker.history(interval="1d", period="1y")
data = data.reset_index()
data = data.drop(["Date", "Dividends", "Stock Splits"], axis=1)
data = data.tail(100)
data["Date"] = dates
data.set_index("Date", inplace=True) 



st.title("Candlestick Chart with mplfinance")
st.sidebar.header("Candlestick Chart Settings")
chart_type = st.sidebar.selectbox("Chart Type", options=["Candlestick", "OHLC"])
volume_display = st.sidebar.checkbox("Show Volume", value=False)

# Plot candlestick or OHLC chart
if chart_type == "Candlestick":
    fig, ax = mpf.plot(data, type='candle', volume=volume_display, returnfig=True, style='yahoo')
else:
    fig, ax = mpf.plot(data, type='ohlc', volume=volume_display, returnfig=True, style='yahoo')

# Display the chart in Streamlit
st.pyplot(fig)
