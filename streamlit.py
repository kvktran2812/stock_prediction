import streamlit as st
from datetime import datetime, timedelta
import mplfinance as mpf
import pandas as pd
import numpy as np

# Streamlit Title
st.title("Candlestick Chart with Predictive Trend")

# Example candlestick data
data = {
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
    'Open': [430, 435, 440, 450],
    'High': [440, 445, 450, 460],
    'Low': [425, 430, 435, 440],
    'Close': [435, 440, 445, 455],
    'Volume': [1000, 1200, 1500, 1600]
}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Add predictive trend line
trend_dates = pd.date_range(start=df.index[-1], periods=4, freq='D')
trend_prices = [455, 460, 465, 470]  # Example trend prices

# Create candlestick chart and retrieve Figure and Axes
fig, ax = mpf.plot(
    df,
    type='candle',
    style='charles',
    volume=False,
    returnfig=True
)

# Add trend line to the chart
x = np.arange(len(df.index), len(df.index) + len(trend_prices))
ax[0].plot(x, trend_prices, label='Predictive Trend', linestyle='--', color='red')
ax[0].legend()

# Adjust x-axis ticks
new_dates = list(df.index) + list(trend_dates)
ax[0].set_xticks(range(len(new_dates)))
ax[0].set_xticklabels([date.strftime('%Y-%m-%d') for date in new_dates], rotation=45)

# Display the chart in Streamlit
st.pyplot(fig)
