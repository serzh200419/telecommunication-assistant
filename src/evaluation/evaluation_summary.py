import csv
import json
import math
from statistics import mean, median, quantiles

from src.evaluation.benchmark_validation import ROOT, validate_benchmark_questions


EVALUATION_DIR = ROOT / "data/evaluation"
PROVIDERS = ("gemini", "groq", "mistral")
GROUPS = ("armenian_answerable", "english_answerable", "synthesis", "unanswerable")
PRICING = {
    "pricing_date": "2026-09-23",
    "currency": "USD",
    "actual_benchmark_tier": "free tier",
    "actual_cost_basis": "benchmark API usage incurred no charge",
    "estimated_cost_basis": "standard paid API rates",
    "providers": {
        "gemini": {
            "model": "gemini-3.5-flash-lite",
            "input_price_per_million_usd": 0.30,
            "output_price_per_million_usd": 2.50,
        },
        "groq": {
            "model": "openai/gpt-oss-120b",
            "input_price_per_million_usd": 0.15,
            "output_price_per_million_usd": 0.60,
        },
        "mistral": {
            "model": "voxtral-small-2507",
            "input_price_per_million_usd": 0.10,
            "output_price_per_million_usd": 0.40,
        },
    },
}
CSV_FIELDS = (
    "provider", "model", "normalized_answer_accuracy", "average_answer_score",
    "score_2_count", "score_1_count", "score_0_count", "hallucination_count",
    "hallucination_rate", "citation_precision", "citation_recall", "citation_f1",
    "mean_ttft_ms", "median_ttft_ms", "mean_total_latency_ms", "median_total_latency_ms",
    "p95_total_latency_ms", "total_prompt_tokens", "total_completion_tokens", "total_tokens",
    "successful", "failed", "failure_rate", "correct_abstentions",
    "unanswerable_questions", "abstention_accuracy",
    "actual_benchmark_cost_usd", "input_price_per_million_usd", "output_price_per_million_usd",
    "estimated_input_cost_usd", "estimated_output_cost_usd", "estimated_paid_cost_usd",
    "estimated_paid_cost_per_successful_question_usd",
)


def question_group(record):
    if record["type"] == "answerable":
        return "armenian_answerable" if record["language"] == "hy" else "english_answerable"
    return record["type"]


def validate_number(value, field, pair, nullable=False, integer=False, maximum=None):
    if nullable and value is None:
        return
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {field} for {pair}: {value!r}.")


def validate_inputs(benchmarks, reviews, questions):
    if not isinstance(benchmarks, dict) or set(benchmarks) != set(PROVIDERS):
        raise ValueError("Expected exactly three providers: gemini, groq, mistral.")
    originals = {question["id"]: question for question in questions}
    if len(questions) != 18 or len(originals) != 18:
        raise ValueError("Expected 18 unique benchmark questions.")
    pairs = set()
    for provider, records in benchmarks.items():
        if not isinstance(records, list) or len(records) != 18:
            raise ValueError(f"Expected exactly 18 benchmark records for {provider}.")
        models = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Malformed benchmark record for {provider}.")
            identifier = record.get("question_id")
            if not isinstance(identifier, str) or identifier not in originals:
                raise ValueError(f"Unknown question ID for {provider}: {identifier!r}.")
            pair = (provider, identifier)
            if record.get("provider") != provider or pair in pairs:
                raise ValueError(f"Incorrect provider or duplicate benchmark pair: {pair}.")
            pairs.add(pair)
            original = originals[identifier]
            for field in ("language", "type", "question", "expected_answer", "expected_articles"):
                if field not in record or record[field] != original[field]:
                    raise ValueError(f"Benchmark {field} does not match the dataset for {pair}.")
            status = record.get("status")
            if status not in ("success", "failed") or "error" not in record:
                raise ValueError(f"Invalid status or missing error field for {pair}.")
            if status == "success" and record["error"] is not None:
                raise ValueError(f"Successful record contains an error for {pair}.")
            if status == "failed" and (not isinstance(record["error"], str) or not record["error"].strip()):
                raise ValueError(f"Failed record needs an error for {pair}.")
            model = record.get("model")
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"Missing model for {pair}.")
            models.add(model)
            for field in ("citation_precision", "citation_recall", "citation_f1"):
                if field not in record:
                    raise ValueError(f"Missing {field} for {pair}.")
                validate_number(record[field], field, pair, nullable=status == "failed", maximum=1)
            for field in ("ttft_ms", "total_latency_ms", "prompt_tokens", "completion_tokens"):
                if field not in record:
                    raise ValueError(f"Missing {field} for {pair}.")
                validate_number(record[field], field, pair, nullable=True, integer=field.endswith("tokens"))
        if len(models) != 1:
            raise ValueError(f"Multiple models in final results for {provider}.")
    if not isinstance(reviews, list):
        raise ValueError("Manual reviews must be a list.")
    indexed_reviews = {}
    for review in reviews:
        if not isinstance(review, dict):
            raise ValueError("Malformed manual review record.")
        provider, identifier = review.get("provider"), review.get("question_id")
        if not isinstance(provider, str) or not isinstance(identifier, str):
            raise ValueError("Manual reviews need string provider and question_id fields.")
        pair = (provider, identifier)
        if pair in indexed_reviews:
            raise ValueError(f"Duplicate manual review: {pair}.")
        if pair not in pairs:
            raise ValueError(f"Unexpected manual review: {pair}.")
        score = review.get("answer_accuracy_score")
        if type(score) is not int or score not in (0, 1, 2):
            raise ValueError(f"answer_accuracy_score must be 0, 1, or 2 for {pair}.")
        if type(review.get("hallucination")) is not bool:
            raise ValueError(f"hallucination must be boolean for {pair}.")
        if "reviewer_rationale" not in review or (
            review["reviewer_rationale"] is not None and not isinstance(review["reviewer_rationale"], str)
        ):
            raise ValueError(f"Invalid reviewer_rationale for {pair}.")
        indexed_reviews[pair] = review
    missing = pairs - indexed_reviews.keys()
    if missing:
        raise ValueError(f"Missing manual review for: {sorted(missing)}.")
    if len(reviews) != 54:
        raise ValueError("Expected exactly 54 manual review records.")
    return indexed_reviews


def normalized_accuracy(records, reviews):
    if not records:
        return None
    scores = [reviews[(record["provider"], record["question_id"])]["answer_accuracy_score"] for record in records]
    return sum(scores) / (2 * len(scores))


def summarize_provider(provider, records, reviews):
    successful = [record for record in records if record["status"] == "success"]
    scores = [reviews[(provider, record["question_id"])]["answer_accuracy_score"] for record in successful]
    hallucinations = sum(reviews[(provider, record["question_id"])]["hallucination"] for record in successful)
    unanswerable = [record for record in records if record["type"] == "unanswerable"]
    correct = sum(
        record["status"] == "success" and reviews[(provider, record["question_id"])]["answer_accuracy_score"] == 2
        for record in unanswerable
    )
    summary = {
        "provider": provider, "model": records[0]["model"],
        "total_benchmark_records": len(records), "successful": len(successful),
        "failed": len(records) - len(successful),
        "failure_rate": (len(records) - len(successful)) / len(records),
        "reviewed_successful_questions": len(successful),
        "normalized_answer_accuracy": normalized_accuracy(successful, reviews),
        "average_answer_score": mean(scores) if scores else None,
        **{f"score_{score}_count": scores.count(score) for score in (2, 1, 0)},
        "hallucination_count": hallucinations,
        "hallucination_rate": hallucinations / len(successful) if successful else None,
        "unanswerable_questions": len(unanswerable), "correct_abstentions": correct,
        "abstention_accuracy": correct / len(unanswerable) if unanswerable else None,
    }
    for field in ("citation_precision", "citation_recall", "citation_f1"):
        summary[field] = mean(record[field] for record in successful) if successful else None
    coverage = {}
    for field in ("ttft_ms", "total_latency_ms"):
        values = [record[field] for record in successful if record[field] is not None]
        coverage[field] = len(values)
        summary["mean_" + field] = mean(values) if values else None
        summary["median_" + field] = median(values) if values else None
        if field == "total_latency_ms":
            summary["p95_" + field] = quantiles(values, n=100, method="inclusive")[94] if len(values) > 1 else (
                values[0] if values else None
            )
    for field in ("prompt_tokens", "completion_tokens"):
        values = [record[field] for record in records if record[field] is not None]
        successful_values = [record[field] for record in successful if record[field] is not None]
        coverage[field + "_all_records"] = len(values)
        coverage[field + "_successful"] = len(successful_values)
        summary["total_" + field] = sum(values) if values else None
        summary["average_" + field + "_per_successful_question"] = mean(successful_values) if successful_values else None
    totals = [summary["total_prompt_tokens"], summary["total_completion_tokens"]]
    summary["total_tokens"] = sum(totals) if all(value is not None for value in totals) else None
    summary["measurement_counts"] = coverage
    prices = PRICING["providers"][provider]
    input_tokens = summary["total_prompt_tokens"]
    output_tokens = summary["total_completion_tokens"]
    input_cost = input_tokens / 1_000_000 * prices["input_price_per_million_usd"] if input_tokens is not None else None
    output_cost = output_tokens / 1_000_000 * prices["output_price_per_million_usd"] if output_tokens is not None else None
    paid_cost = input_cost + output_cost if input_cost is not None and output_cost is not None else None
    summary.update({
        "actual_benchmark_cost_usd": 0.0,
        "input_price_per_million_usd": prices["input_price_per_million_usd"],
        "output_price_per_million_usd": prices["output_price_per_million_usd"],
        "estimated_input_cost_usd": input_cost,
        "estimated_output_cost_usd": output_cost,
        "estimated_paid_cost_usd": paid_cost,
        "estimated_paid_cost_per_successful_question_usd": (
            paid_cost / len(successful) if paid_cost is not None and successful else None
        ),
    })
    return summary


def aggregate_results(benchmarks, manual_reviews, questions):
    reviews = validate_inputs(benchmarks, manual_reviews, questions)
    providers = {provider: summarize_provider(provider, benchmarks[provider], reviews) for provider in PROVIDERS}
    by_type = {}
    for group in GROUPS:
        by_type[group] = {}
        for provider in PROVIDERS:
            records = [record for record in benchmarks[provider] if question_group(record) == group]
            successful = [record for record in records if record["status"] == "success"]
            by_type[group][provider] = {
                "questions": len(records), "reviewed_successful_questions": len(successful),
                "normalized_answer_accuracy": normalized_accuracy(successful, reviews),
            }
    return {
        "pricing": PRICING,
        "methodology": {
            "accuracy_and_hallucination": "Human reviews of successful final records only.",
            "citations": "Macro averages of stored metrics for successful records; no recalculation.",
            "latency": "Successful records with reported measurements only; milliseconds.",
            "p95": "Linear interpolation at rank (n - 1) * 0.95, equivalent to inclusive quantiles.",
            "tokens": "Totals include reported usage from all final records, including failures. Averages use successful records with reported usage. Measurement counts expose missing values.",
            "reliability": "Final saved state only; historical attempts and retries are not reconstructed.",
            "abstention": "Successful unanswerable records scoring 2 divided by all unanswerable records; failures are not correct abstentions.",
            "empty_groups": "Undefined means and rates are null, not zero.",
        },
        "providers": providers, "by_question_type": by_type,
    }


def main():
    paths = {provider: EVALUATION_DIR / "results" / f"{provider}_benchmark.json" for provider in PROVIDERS}
    try:
        questions = json.loads((EVALUATION_DIR / "benchmark_questions.json").read_text(encoding="utf-8"))
        articles = json.loads((ROOT / "data/processed/law_articles.json").read_text(encoding="utf-8"))
        validate_benchmark_questions(questions, articles)
        benchmarks = {provider: json.loads(path.read_text(encoding="utf-8")) for provider, path in paths.items()}
        reviews = json.loads((EVALUATION_DIR / "manual_review.json").read_text(encoding="utf-8"))
        summary = aggregate_results(benchmarks, reviews, questions)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Evaluation summary failed: {error}") from error
    summary["sources"] = {
        **{provider: str(path.relative_to(ROOT)).replace("\\", "/") for provider, path in paths.items()},
        "manual_review": "data/evaluation/manual_review.json",
        "questions": "data/evaluation/benchmark_questions.json",
    }
    (EVALUATION_DIR / "final_benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (EVALUATION_DIR / "final_benchmark_summary.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["providers"].values())
    print("Provider | Accuracy | Hallucination | Citation F1 | Mean TTFT ms | Mean latency ms | Actual cost USD | Estimated paid cost USD | Failure | Abstention")
    for row in summary["providers"].values():
        values = [row[field] for field in (
            "normalized_answer_accuracy", "hallucination_rate", "citation_f1", "mean_ttft_ms",
            "mean_total_latency_ms", "actual_benchmark_cost_usd", "estimated_paid_cost_usd",
            "failure_rate", "abstention_accuracy",
        )]
        print(row["provider"] + " | " + " | ".join("N/A" if value is None else f"{value:.4f}" for value in values))


if __name__ == "__main__":
    main()
