import streamlit as st
from datetime import datetime, timedelta
import mplfinance as mpf
import pandas as pd
import numpy as np

st.title("Stock Prediction Model Info")

st.subheader("Intro")
st.markdown(
    """
    In stock prediction, there are various approaches, and my models focus primarily on technical analysis rather than fundamental analysis.\
    These models analyze historical data, including Open, High, Low, Close, and Volume, to forecast future price movements. \
    To preserve the richness of raw data and avoid losing crucial patterns, I have chosen not to rely on traditional indicators. \
    Instead, leveraging LSTM and CNNs, I have developed four distinct models, each tailored to address different market scenarios.
    """
)


# First model kv_stock_v1
st.subheader("kv_stock_v1")




# Second model kv_stock_v2
st.subheader("kv_stock_v2")




# Third model kv_stock_v3
st.subheader("kv_stock_v3")