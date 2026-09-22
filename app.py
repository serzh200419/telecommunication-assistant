import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("Electronic Communications Legal Assistant")

qa_tab, benchmark_tab = st.tabs(["Q&A", "Benchmark"])

with qa_tab:
    st.write("Question answering will be added later.")

with benchmark_tab:
    st.write("Provider benchmarking will be added later.")
