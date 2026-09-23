# telecommunication-assistant

Start the web application from the repository root:

```sh
python -m streamlit run app.py
```

Ask the Law accepts Armenian or English questions and uses the selected provider
through the existing RAG pipeline. Configure its API key and model in `.env` or
environment variables as shown in `.env.example`. Local processed law and index
files must already exist. Retrieved context is available in a collapsed expander.
The Benchmark tab reads `data/evaluation/final_benchmark_summary.json` and makes
no provider calls. Generate that file with
`python -m src.evaluation.evaluation_summary` if it is missing.

Run one benchmark question or the full fixed dataset for an explicit provider:

```sh
python -m src.evaluation.benchmark --provider groq --question-id hy_003
python -m src.evaluation.benchmark --provider groq
python -m src.evaluation.benchmark --provider mistral
python -m src.evaluation.benchmark --provider gemini
```

These commands make live provider calls using the existing configuration and
`src.rag.ask` pipeline. No provider is selected by default. Calls run sequentially.
Each completed call is saved immediately to
`data/evaluation/results/<provider>_benchmark.json` using a temporary file and
replacement, so an interrupted write does not truncate the previous results.

Reruns skip saved successes and retry failures automatically, replacing the
previous failed record. Unselected records remain saved. A rate-limit failure is
saved and stops the run without sleeping or retrying immediately. Other returned
provider errors are saved and the run continues. Unexpected exceptions stop the
run, retaining previously saved results. A run with failed calls exits nonzero.
Use one process per provider result file. Keep the same configured model when
resuming; archive the result file before benchmarking a different model.

Results include UTC call-start timestamps, configured model names, retrieved
chunks, provider metadata, raw answer/citation fields, and deterministic citation
scores following `data/evaluation/EVALUATION_RUBRIC.md`. Failed calls have null
citation scores. Human accuracy, hallucination, and rationale fields remain null
for manual review. Streaming adapters measure TTFT from request start to the first
non-empty generated text fragment, excluding role-only, usage-only, and reasoning
events. Total latency ends when the stream finishes; neither measurement includes
retrieval. TTFT stays null if no output text arrives. Usage comes only from provider
metadata. Groq uses JSON-object mode because its strict JSON-schema mode does not
support streaming; local JSON and citation validation still apply. Gemini and
Mistral retain their schema constraints. Existing saved successes are still skipped
on resume, so archive old result files before collecting fresh streaming timings.

The printed summary covers newly attempted calls in that invocation. Latency and
citation averages include successful responses only. Token totals include known
usage from successes and failures; reported-call counts show missing coverage.
Unavailable measurements display N/A. Saved successes are reported as skipped.
Results are ignored by Git. No cost or human-review scores are calculated.

Offline checks:

```sh
python -m src.evaluation.benchmark_validation
python -m unittest discover -s tests -v
```
