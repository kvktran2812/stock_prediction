import streamlit as st

# Setup pages
overview_page = st.Page("streamlit_pages/overview.py", title="1. Overview")
model_info_page = st.Page("streamlit_pages/model_info.py", title="2. Model Info")
model_crafting_page = st.Page("streamlit_pages/model_crafting.py", title="3. Model Crafting")
demo_page = st.Page("streamlit_pages/demo.py", title="4. Demo")



# Setup main pages 
pg = st.navigation([overview_page, model_info_page, model_crafting_page, demo_page])
st.set_page_config(page_title="Stock Prediction Project")
pg.run()