import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from stock import *

st.title("Demo Stock Models Prediction")
st.markdown("""
    This page demonstrates the prediction of stock models on real-time stock data. The stock ticker is **AAPL**.   
    There are two models used here to predict the future price:  
    - kv_stock_v1: Capture trends and patterns of short-term price movements and predict short-term price changes 
        **(64 time steps to predict 8 time steps in the future.)**
    - kv_stock_v2: Capture trends and patterns of long-term price movements and predict long-term price changes. 
        **(128 time steps to predict 32 time steps in the future.)**
""")


# load data:
data = load_raw_stock_data("AAPL", interval="1d", period="1y")
process_data = clean_data(data)


# load model kv_stock_v1:
kv_stock_v1 = load_kv_stock_v1("models/kv_stock_v1.pth")
kv_stock_v2 = load_kv_stock_v2("models/kv_stock_v2.pth")
kv_stock_v1_predicted = kv_stock_v1_predict(kv_stock_v1, process_data, close_data=data["Close"][-1])
kv_stock_v2_predicted = kv_stock_v2_predict(kv_stock_v2, process_data, close_data=data["Close"][-1])


# visualize data:
show_volume = st.toggle("Show Volume")
if show_volume:
    fig, ax = mpf.plot(
        data=data.iloc[-64:],
        type='candle',
        style='charles',
        volume=True,
        returnfig=True,
    )
else:
    fig, ax = mpf.plot(
        data=data.iloc[-64:],
        type='candle',
        style='charles',
        volume=False,
        returnfig=True,
    )



# plot prediction from models
size = len(data.iloc[-64:].index)
kv_stock_v1_plot(ax, kv_stock_v1_predicted, size)
kv_stock_v2_plot(ax, kv_stock_v2_predicted, size)

st.pyplot(fig)