import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from stock import *

st.title("This is demo page")


# load data:
data = load_raw_stock_data("AAPL", interval="1d", period="3mo")
process_data = clean_data(data)


# load model kv_stock_v1:
kv_stock_v1 = load_kv_stock_v1("models/kv_stock_v1.pth")
predicted = kv_stock_v1(torch.tensor(process_data[-64:].reshape(1, 64, 5), dtype=torch.float32)).detach().numpy().reshape(8,)

for i in range(len(predicted)):
    predicted[i] = (1 + predicted[i]) * data["Close"][-1]


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

future_x = np.arange(len(data.index), len(data.index) + 8)  # Extend x-axis for predictions
future_y = predicted  # Predictive trend prices

ax[0].plot(future_x, future_y)

st.pyplot(fig)