import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from time import sleep

from src import rag
from src.evaluation.benchmark_validation import ROOT, validate_benchmark_questions


DATASET_PATH = ROOT / "data/evaluation/benchmark_questions.json"
RESULTS_DIR = ROOT / "data/evaluation/results"
PROVIDERS = ("gemini", "groq", "mistral")
QUESTION_FIELDS = ("language", "type", "question", "expected_answer", "expected_articles")


def load_questions():
    questions = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    articles = json.loads((ROOT / "data/processed/law_articles.json").read_text(encoding="utf-8"))
    validate_benchmark_questions(questions, articles)
    return questions


def citation_metrics(returned, expected):
    returned, expected = set(returned), set(expected)
    if not returned or not expected:
        precision = recall = f1 = float(not returned and not expected)
    else:
        correct = len(returned & expected)
        precision = correct / len(returned)
        recall = correct / len(expected)
        f1 = 2 * precision * recall / (precision + recall) if correct else 0.0
    return {"citation_precision": precision, "citation_recall": recall, "citation_f1": f1}


def load_results(path, provider, questions):
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Saved benchmark results must be a list.")
    originals = {question["id"]: question for question in questions}
    results = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid saved benchmark record.")
        identifier = record.get("question_id")
        if not isinstance(identifier, str) or identifier not in originals or identifier in results:
            raise ValueError("Unknown or duplicate saved question ID.")
        if record.get("provider") != provider or record.get("status") not in ("success", "failed"):
            raise ValueError("Saved provider or status does not match this benchmark.")
        if any(record.get(field) != originals[identifier][field] for field in QUESTION_FIELDS):
            raise ValueError("Saved results do not match the benchmark dataset.")
        results[identifier] = record
    return results


def save_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(list(results.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def make_result(question, response, timestamp):
    failed = response.get("error") is not None
    record = {
        "question_id": question["id"],
        **{field: question[field] for field in QUESTION_FIELDS},
        **{field: response.get(field) for field in (
            "provider", "model", "answer", "citations", "retrieved_articles", "retrieved_results",
            "prompt_tokens", "completion_tokens", "total_latency_ms", "error",
            "raw_answer", "raw_citations", "ttft_ms",
        )},
        "timestamp": timestamp,
        "status": "failed" if failed else "success",
        "citation_precision": None,
        "citation_recall": None,
        "citation_f1": None,
        "answer_accuracy_score": None,
        "hallucination": None,
        "reviewer_rationale": None,
    }
    if not failed:
        record.update(citation_metrics(response["citations"], question["expected_articles"]))
    return record


def is_rate_limit(error):
    message = (error or "").lower().replace("_", "").replace("-", "").replace(" ", "")
    return any(marker in message for marker in (
        "status=429", "ratelimit", "quota", "resourceexhausted", "dailylimit",
    ))


def summarize(records):
    successful = [record for record in records if record["status"] == "success"]
    summary = {
        "attempted": len(records),
        "successful": len(successful),
        "failed": len(records) - len(successful),
    }
    for field in ("total_latency_ms", "citation_precision", "citation_recall", "citation_f1"):
        values = [record[field] for record in successful if record[field] is not None]
        summary["average_" + field] = mean(values) if values else None
    for field in ("prompt_tokens", "completion_tokens"):
        values = [record[field] for record in records if record[field] is not None]
        summary["total_" + field] = sum(values) if values else None
        summary[field + "_reported_calls"] = len(values)
    return summary


def run_benchmark(provider, question_id=None, results_dir=RESULTS_DIR, delay_seconds=0.0):
    if provider not in PROVIDERS:
        raise ValueError("Choose gemini, groq, or mistral.")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative.")
    questions = load_questions()
    selected = [question for question in questions if question_id is None or question["id"] == question_id]
    if not selected:
        raise ValueError("Unknown benchmark question ID.")
    path = Path(results_dir) / f"{provider}_benchmark.json"
    results = load_results(path, provider, questions)
    attempted = []
    skipped = 0
    for question in selected:
        previous = results.get(question["id"])
        if previous and previous["status"] == "success":
            skipped += 1
            continue
        if attempted and delay_seconds > 0:
            sleep(delay_seconds)
        timestamp = datetime.now(timezone.utc).isoformat()
        response = rag.ask(question["question"], provider=provider)
        record = make_result(question, response, timestamp)
        results[question["id"]] = record
        save_results(path, results)
        attempted.append(record)
        if is_rate_limit(record["error"]):
            print("Rate-limit failure saved. Stopping; rerun later to resume.")
            break
    summary = summarize(attempted)
    print(f"Provider: {provider}; this invocation only; skipped saved successes: {skipped}")
    for name, value in summary.items():
        print(f"{name}: {value if value is not None else 'N/A'}")
    print(f"Results: {path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run a resumable benchmark for one explicit provider.")
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--question-id")
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    args = parser.parse_args()
    try:
        summary = run_benchmark(
                provider=args.provider,
                question_id=args.question_id,
                delay_seconds=args.delay_seconds,
            )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Benchmark stopped ({type(error).__name__}); saved results were retained.\n")
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
