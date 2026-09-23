import json
import math
from pathlib import Path


SUMMARY_PATH = Path(__file__).resolve().parents[1] / "data/evaluation/final_benchmark_summary.json"
PROVIDERS = {"gemini": "Gemini", "groq": "Groq", "mistral": "Mistral", "openai": "OpenAI (paid)"}
GROUPS = {
    "armenian_answerable": "Armenian answerable", "english_answerable": "English answerable",
    "synthesis": "Synthesis", "unanswerable": "Unanswerable",
}
METRICS = (
    ("normalized_answer_accuracy", "Answer accuracy", "percent"),
    ("citation_precision", "Citation precision", "percent"),
    ("citation_recall", "Citation recall", "percent"),
    ("citation_f1", "Citation F1", "percent"),
    ("hallucination_rate", "Hallucination rate", "percent"),
    ("mean_ttft_ms", "Mean TTFT", "seconds"),
    ("mean_total_latency_ms", "Mean total latency", "seconds"),
    ("p95_total_latency_ms", "P95 total latency", "seconds"),
    ("total_prompt_tokens", "Prompt tokens", "tokens"),
    ("total_completion_tokens", "Completion tokens", "tokens"),
    ("actual_benchmark_cost_usd", "Actual benchmark cost", "money"),
    ("estimated_paid_cost_usd", "Estimated paid cost", "money"),
    ("failure_rate", "Failure rate", "percent"),
    ("abstention_accuracy", "Abstention accuracy", "percent"),
)


def format_metric(value, kind):
    if value is None:
        return "N/A"
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid numeric benchmark metric.")
    if kind == "percent":
        if value > 1:
            raise ValueError("Benchmark rate must be between 0 and 1.")
        return f"{value:.2%}"
    if kind == "seconds":
        return f"{value / 1000:.2f} s"
    if kind == "tokens":
        if type(value) is not int:
            raise ValueError("Token counts must be integers.")
        return f"{value:,}"
    return "$0.00" if value == 0 else f"${value:.7f}"


def load_benchmark_tables(path=SUMMARY_PATH):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        available = set(data["providers"])
        if available != set(PROVIDERS):
            raise ValueError("Expected exactly four providers: Gemini, Groq, Mistral, and OpenAI.")
        displayed = {provider: label.removesuffix(" (paid)") for provider, label in PROVIDERS.items()}
        comparison = []
        for provider, label in displayed.items():
            record = data["providers"][provider]
            if record["provider"] != provider or not isinstance(record["model"], str) or not record["model"].strip():
                raise ValueError("Invalid provider or model.")
            comparison.append({
                "Provider": label, "Model": record["model"],
                **{label: format_metric(record[field], kind) for field, label, kind in METRICS},
            })
        breakdown = [
            {"Question type": label, **{
                name: format_metric(data["by_question_type"][group][provider]["normalized_answer_accuracy"], "percent")
                for provider, name in displayed.items()
            }}
            for group, label in GROUPS.items()
        ]
        return comparison, breakdown
    except FileNotFoundError as error:
        raise ValueError("Benchmark summary is missing. Run python -m src.evaluation.evaluation_summary.") from error
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("Benchmark summary is malformed or unreadable. Regenerate it with python -m src.evaluation.evaluation_summary.") from error
