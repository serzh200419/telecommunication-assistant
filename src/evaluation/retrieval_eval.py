import argparse
import json
from pathlib import Path
from statistics import mean

from src.retrieval import retrieve


OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data/evaluation/retrieval_results.json"


def validate_questions(questions: list[dict]) -> None:
    if not isinstance(questions, list) or not questions:
        raise ValueError("Evaluation input must be a nonempty list of questions.")
    seen_ids = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each evaluation question must be an object.")
        for field in ("id", "language", "type", "question"):
            value = question.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Question field '{field}' must be a nonempty string.")
        if question["id"] in seen_ids:
            raise ValueError(f"Duplicate question ID: {question['id']}")
        seen_ids.add(question["id"])
        if question["language"] not in {"hy", "en"}:
            raise ValueError("Question language must be 'hy' or 'en'.")
        if question["type"] not in {"answerable", "synthesis", "unanswerable"}:
            raise ValueError("Unknown evaluation question type.")
        expected = question.get("expected_articles")
        if not isinstance(expected, list) or any(
            not isinstance(article, str) or not article.strip() for article in expected
        ):
            raise ValueError("expected_articles must be a list of nonempty article strings.")
        if len(set(expected)) != len(expected):
            raise ValueError("expected_articles must not contain duplicates.")
        if question["type"] == "unanswerable" and expected:
            raise ValueError("Unanswerable questions must have empty expected_articles.")
        if question["type"] != "unanswerable" and not expected:
            raise ValueError("Answerable and synthesis questions need expected_articles.")


def evaluate_question(question: dict) -> dict:
    retrieved = retrieve(question["question"], top_k=5)
    articles = list(dict.fromkeys(result["article_number"] for result in retrieved))
    result = {
        **question,
        "retrieved_results": retrieved,
        "unique_articles": articles,
        "top_result_score": retrieved[0]["score"] if retrieved else None,
        "metrics": None,
    }
    if question["type"] == "unanswerable":
        return result

    expected = question["expected_articles"]
    ranks = {article: articles.index(article) + 1 if article in articles else None for article in expected}
    found_ranks = [rank for rank in ranks.values() if rank is not None]
    metrics = {
        f"recall_at_{k}": sum(rank <= k for rank in found_ranks) / len(expected)
        for k in (1, 3, 5)
    }
    metrics["reciprocal_rank"] = 1 / min(found_ranks) if found_ranks else 0.0
    if question["type"] == "synthesis":
        metrics["expected_articles_found_at_5"] = len(found_ranks)
        metrics["coverage_at_5"] = len(found_ranks) / len(expected)
    result["expected_article_ranks"] = ranks
    result["metrics"] = metrics
    return result


def evaluate_questions(questions: list[dict]) -> dict:
    validate_questions(questions)
    results = [evaluate_question(question) for question in questions]
    applicable = [result for result in results if result["metrics"] is not None]
    synthesis = [result for result in applicable if result["type"] == "synthesis"]
    summary = {
        "question_count": len(results),
        "answerable_synthesis_count": len(applicable),
        "unanswerable_count": len(results) - len(applicable),
        **{
            f"recall_at_{k}": mean(result["metrics"][f"recall_at_{k}"] for result in applicable) if applicable else None
            for k in (1, 3, 5)
        },
        "mrr": mean(result["metrics"]["reciprocal_rank"] for result in applicable) if applicable else None,
        "average_synthesis_coverage_at_5": mean(result["metrics"]["coverage_at_5"] for result in synthesis) if synthesis else None,
        "no_expected_article_at_5": [result["id"] for result in applicable if result["metrics"]["recall_at_5"] == 0],
        "first_relevant_below_rank_1": [result["id"] for result in applicable if 0 < result["metrics"]["reciprocal_rank"] < 1],
    }
    return {"summary": summary, "results": results}


def print_summary(summary: dict) -> None:
    print(f"Evaluated questions: {summary['question_count']}")
    print(f"Answerable/synthesis: {summary['answerable_synthesis_count']}")
    print(f"Unanswerable: {summary['unanswerable_count']}")
    for label, key in [
        ("Recall@1", "recall_at_1"), ("Recall@3", "recall_at_3"),
        ("Recall@5", "recall_at_5"), ("MRR", "mrr"),
        ("Average synthesis coverage@5", "average_synthesis_coverage_at_5"),
    ]:
        value = summary[key]
        print(f"{label}: {value:.4f}" if value is not None else f"{label}: N/A")
    print("No expected article in Top-5: " + (", ".join(summary["no_expected_article_at_5"]) or "none"))
    print("First relevant article below rank 1: " + (", ".join(summary["first_relevant_below_rank_1"]) or "none"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval using unique article ranks from five chunks.")
    parser.add_argument("questions_path", type=Path)
    args = parser.parse_args()
    questions = json.loads(args.questions_path.read_text(encoding="utf-8"))
    report = evaluate_questions(questions)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(report["summary"])
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
