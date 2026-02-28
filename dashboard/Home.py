import subprocess
import streamlit as st

st.set_page_config(page_title="Instacart Retail Analytics", layout="wide")

st.sidebar.success("✅ Choose a page here 👇")

st.title("🏠 Instacart Retail Analytics")
st.caption("Multi-page Streamlit app: Dashboard + Business pages")

st.markdown("### Open the dashboard")
st.markdown("Go to the left sidebar and choose **Dashboard** or **Basket Recommender**.")

st.info("Tip: Run the app with: `streamlit run Home.py`")
