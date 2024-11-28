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

data = data.to_numpy()

n_windows = 64
n_size = len(process_data)
n_features = len(process_data[0])
n_futures = 8


# load model kv_stock_v1:
kv_stock_v1 = load_kv_stock_v1("models/kv_stock_v1.pth")
kv_stock_v2 = load_kv_stock_v2("models/kv_stock_v2.pth")
# kv_stock_v1_predicted = kv_stock_v1_predict(kv_stock_v1, process_data, close_data=data["Close"][-1])
# kv_stock_v2_predicted = kv_stock_v2_predict(kv_stock_v2, process_data, close_data=data["Close"][-1])

X = np.zeros((n_size - n_windows - n_futures, n_windows, n_features))

for i in range(n_size - n_windows - n_futures):
    X[i] = process_data[i:i+n_windows]


test_x = torch.tensor(X[-64:], dtype=torch.float32)
kv_stock_v1.eval()

y_pred = kv_stock_v1(test_x)
y_pred = y_pred.detach().numpy()
y_pred.shape

close = data[-65:-1,3]

for i in range(8):
    for j in range(64):
        y_pred[j][i] = close[j] * (1 + y_pred[j][i])

plot_data = data[-64:]
df = pd.DataFrame(plot_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
df.index = pd.date_range("2023-01-01", periods=len(df), freq="D")

fig, ax = mpf.plot(
    df,
    type='candle',
    style='charles',
    datetime_format='',
    volume=False,
    returnfig=True,
)

plot_y = y_pred[:,0]
plot_x = np.arange(0, 64)

ax[0].plot(plot_x, plot_y, color="blue")
plt.show()


# visualize data:
# show_volume = st.toggle("Show Volume")
# if show_volume:
#     fig, ax = mpf.plot(
#         data=data.iloc[-64:],
#         type='candle',
#         style='charles',
#         volume=True,
#         returnfig=True,
#     )
# else:
#     fig, ax = mpf.plot(
#         data=data.iloc[-64:],
#         type='candle',
#         style='charles',
#         volume=False,
#         returnfig=True,
#     )



# # plot prediction from models
# size = len(data.iloc[-64:].index)
# kv_stock_v1_plot(ax, kv_stock_v1_predicted, size)
# kv_stock_v2_plot(ax, kv_stock_v2_predicted, size)

st.pyplot(fig)