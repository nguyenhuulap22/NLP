import streamlit as st

st.set_page_config(
    page_title="IT Translator Pro",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

from translator.translator import Translator
from ui import render_app


@st.cache_resource
def load_translator():
    return Translator()


with st.sidebar:
    if st.button("🔄 Reset cache / Reload model"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()


translator = load_translator()

render_app(translator)