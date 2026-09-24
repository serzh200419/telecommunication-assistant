import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src import rag
from src.query_preprocessing import meta_answer
from src.ui import PROVIDERS, load_benchmark_tables


def show_qa():
    st.write("Ask in Armenian or English about the Law of the Republic of Armenia on Electronic Communications.")
    with st.form("law_question"):
        question = st.text_area("Question / Հարց")
        provider = st.selectbox("Provider", list(PROVIDERS), format_func=PROVIDERS.get)
        submitted = st.form_submit_button("Ask")
    if submitted:
        st.session_state.pop("law_response", None)
        if not question.strip():
            st.warning("Enter a question in Armenian or English.")
        elif meta_answer(question) is None and not all(os.getenv(f"{provider.upper()}_{field}", "").strip() for field in ("API_KEY", "MODEL")):
            st.error(f"{PROVIDERS[provider]} is not configured. Set {provider.upper()}_API_KEY and {provider.upper()}_MODEL in the environment or .env file.")
        else:
            with st.spinner("Reading the law and preparing an answer..."):
                try:
                    st.session_state.law_response = rag.ask(question, provider=provider)
                except Exception:
                    st.error("The request could not be completed. Check the local law index and provider configuration, then try again.")
    result = st.session_state.get("law_response")
    if result is None:
        return
    st.caption(f"Provider: {PROVIDERS.get(result['provider'], result['provider'])} | Model: {result['model'] or 'Not configured'}")
    if result["error"] is not None:
        st.error(result["error"])
        return
    st.subheader("Answer")
    st.text(result["answer"])
    if result["citations"]:
        st.write("Citations: " + ", ".join(f"Article {article}" for article in dict.fromkeys(result["citations"])))
    elif result["provider"] != "local":
        st.info("No citations returned because the provided law did not contain enough information.")
    if not result["retrieved_results"]:
        return
    with st.expander("Retrieved context"):
        for chunk in result["retrieved_results"]:
            st.text(f"Article {chunk['article_number']}: {chunk['title']}")
            st.text(chunk["text"])


def show_benchmark():
    st.write("Finalized results from the saved benchmark summary, using the same RAG pipeline across three required free-tier providers and one additional paid OpenAI model.")
    try:
        comparison, breakdown = load_benchmark_tables()
    except ValueError as error:
        st.error(str(error))
        return
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption("Gemini, Groq, and Mistral used free-tier API access. OpenAI was included as an additional paid comparison. Estimated paid cost uses the standard API rates recorded in the summary.")
    st.caption("Answer accuracy and hallucination were human-reviewed. Timing values are in seconds; costs are in USD. N/A indicates an unavailable measurement.")
    st.subheader("Answer accuracy by question type")
    st.dataframe(breakdown, hide_index=True, width="stretch")


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    st.set_page_config(page_title="Electronic Communications Legal Assistant", layout="wide")
    st.title("Electronic Communications Legal Assistant")
    qa_tab, benchmark_tab = st.tabs(["Ask the Law", "Benchmark"])
    with qa_tab:
        show_qa()
    with benchmark_tab:
        show_benchmark()


if __name__ == "__main__":
    main()
