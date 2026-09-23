import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src import rag
from src.evaluation import benchmark
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
    st.write("Completed benchmark results using the same RAG pipeline for all three providers.")
    try:
        comparison, breakdown = load_benchmark_tables()
    except ValueError as error:
        st.error(str(error))
        return
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption("Benchmark runs used free-tier API access. Estimated paid cost shows what the same token usage would cost under standard paid API rates.")
    st.caption("Answer accuracy and hallucination were human-reviewed. Timing values are in seconds; costs are in USD. N/A indicates an unavailable measurement.")
    st.subheader("Answer accuracy by question type")
    st.dataframe(breakdown, hide_index=True, width="stretch")
    st.divider()
    st.subheader("Run benchmark")
    st.write("Optional: running a benchmark makes live API calls and may take time or encounter free-tier rate limits. Saved successful responses are skipped; failed responses are retried.")
    st.caption("Generated responses require human review before becoming part of the finalized evaluation summary. Running here does not regenerate the saved comparison above.")
    with st.form("run_benchmark"):
        provider = st.selectbox("Benchmark provider", list(PROVIDERS), format_func=PROVIDERS.get)
        delay = st.number_input("Delay between requests (seconds)", min_value=0.0, value=0.0, step=1.0)
        submitted = st.form_submit_button("Run benchmark")
    if submitted:
        with st.spinner(f"Running {PROVIDERS[provider]} benchmark..."):
            try:
                result = benchmark.run_benchmark(provider=provider, delay_seconds=delay)
            except Exception:
                st.error("Benchmark stopped unexpectedly. Previously saved results are retained. Check the benchmark files and provider configuration before retrying.")
            else:
                if result["failed"]:
                    st.error(f"{PROVIDERS[provider]} benchmark recorded {result['failed']} failed calls out of {result['attempted']} attempts. Results were saved. Check provider configuration or rate limits before retrying.")
                elif result["attempted"] == 0:
                    st.success(f"{PROVIDERS[provider]} benchmark is already complete. Saved successes were reused; no API calls were needed.")
                else:
                    st.success(f"{PROVIDERS[provider]} benchmark finished: {result['successful']} successful calls. Responses were saved and require human review before finalization.")


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
