import streamlit as st

# Setup pages
overview_page = st.Page("pages/overview.py", title="Overview")
model_info_page = st.Page("pages/model_info.py", title="Model Info")
model_crafting_page = st.Page("pages/model_crafting.py", title="Model Crafting")



# Setup main pages 
pg = st.navigation([overview_page, model_info_page, model_crafting_page])
st.set_page_config(page_title="Stock Prediction Project")
pg.run()