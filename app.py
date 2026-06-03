import streamlit as st

from dictionary import load_model
from ui import render_app


def initialize_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = ""
    if "last_terms" not in st.session_state:
        st.session_state.last_terms = []


initialize_state()

with st.spinner("Đang tải mô hình dịch, vui lòng chờ..."):
    tokenizer, model = load_model()

render_app(tokenizer, model)
