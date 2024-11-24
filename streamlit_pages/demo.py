import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from stock import *

st.title("This is demo page")


# load data:
data = load_raw_stock_data("AAPL", interval="1d", period="1y")
process_data = clean_data(data)


# load model kv_stock_v1:
kv_stock_v1 = load_kv_stock_v1("models/kv_stock_v1.pth")
kv_stock_v2 = load_kv_stock_v2("models/kv_stock_v2.pth")
kv_stock_v1_predicted = kv_stock_v1_predict(kv_stock_v1, process_data, close_data=data["Close"][-1])
kv_stock_v2_predicted = kv_stock_v2_predict(kv_stock_v2, process_data, close_data=data["Close"][-1])


data = data.iloc[-64:]
# visualize data:
show_volume = st.toggle("Show Volume")
if show_volume:
    fig, ax = mpf.plot(
        data=data,
        type='candle',
        style='charles',
        volume=True,
        returnfig=True,
    )
else:
    fig, ax = mpf.plot(
        data=data,
        type='candle',
        style='charles',
        volume=False,
        returnfig=True,
    )



# plot prediction from models
size = len(data.index)
kv_stock_v1_plot(ax, kv_stock_v1_predicted, size)
kv_stock_v2_plot(ax, kv_stock_v2_predicted, size)

st.pyplot(fig)