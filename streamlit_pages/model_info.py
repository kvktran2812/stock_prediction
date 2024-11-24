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
show_v1 = st.toggle("Show kv_stock_v1 Model Architecture")
st.markdown(
    """
    This model uses LSTM compute latent representation of the input data. Then go through an Dense layer to predict the future price.
    - **Model Type**: LSTM
    - **Input**: 64 time steps of percentage change of Open, High, Low, Close, Volume (64, 5)
    - **Output**: 8 time steps of percentage change of Close (8, 1)
    """
)

if show_v1:
    st.image("images/kv_stock_v1.png", caption="kv_stock_v1 Model Architecture")



# Second model kv_stock_v2
st.subheader("kv_stock_v2")
st.markdown(
    """
    Similar to kv_stock_v1, this model also uses LSTM, but with slightly that it use input of 128 time steps and predict 32 times steps. \
    The idea behind this architecture is to make the model look at longer patterns or trends and predict further to the future.
    - **Model Type**: LSTM
    - **Input**: 128 time steps of percentage change of Open, High, Low, Close, Volume (128, 5)
    - **Output**: 32 time steps of percentage change of Close (32, 1)
    """ 
)



# Third model kv_stock_v3
st.subheader("kv_stock_v3")