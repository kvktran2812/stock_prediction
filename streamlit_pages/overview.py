import streamlit as st
from datetime import datetime, timedelta
import mplfinance as mpf
import pandas as pd
import numpy as np

st.title("Stock Prediction Project")

st.markdown(
    """
    Hello, my name is **Khuong Tran**, and this is my **Stock Prediction Project**  
    This project is designed to analyze stock market trends and make data-driven predictions using machine learning models.
    Explore insights, data, and predictions to make informed investment decisions.
    """
)

# Project Goals
st.subheader("Project Goals")
st.markdown(
    """
    - Build several deep learning models (LSTM, CNNs, etc) to predict stock prices.
    - Model information on how each model architecture works and how to train it.
    - User will be able to designs and craft their own models for different stock tickers.
    - Demo the model on real-time stock data.
    """
)

# Navigation Links
st.subheader("Navigation")
st.markdown(
    """
    Use the sidebar to navigate through the project:
    - **Overview**: Summary of the project.
    - **Model Info**: Details about each model architecture.
    - **Model Crafting**: Design and train your own model for a specific stock ticker.
    - **Demo**: Try out the model on real-time stock data.
    """
)

# Add a footer or note
st.write("___")
st.markdown(
    """
    **Note**: This project is for educational purposes only and should not be used for actual financial decision-making.  
    For any questions, feel free to reach out!
    - Email: kvktran@example.com
    - GitHub: https://github.com/kvktran2812
    - LinkedIn: https://www.linkedin.com/in/khuong-vinh-khang-tran-a078aa1b3/
    """
)