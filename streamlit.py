import streamlit as st
from datetime import datetime, timedelta
import mplfinance as mpf
import pandas as pd
import numpy as np

models_info = st.Page("model_info.py", title="Model Info")
model_crafting = st.Page("model_crafting.py", title="Model Crafting")

pg = st.navigation([models_info, model_crafting])
st.set_page_config(
    page_title="Stock Prediction Models",
)
pg.run()
