import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def validate_benchmark_questions(questions, articles):
    if not isinstance(questions, list) or len(questions) != 18:
        raise ValueError("The benchmark must contain exactly 18 questions.")
    article_numbers = {article["article_number"] for article in articles}
    seen_ids = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each question must be an object.")
        for field in ("id", "question"):
            value = question.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty string.")
        if question["id"] in seen_ids:
            raise ValueError(f"Duplicate question ID: {question['id']}")
        seen_ids.add(question["id"])
        if question.get("language") not in ("hy", "en"):
            raise ValueError("language must be hy or en.")
        if question.get("type") not in ("answerable", "synthesis", "unanswerable"):
            raise ValueError("Invalid question type.")
        expected = question.get("expected_articles")
        if not isinstance(expected, list) or any(
            not isinstance(number, str) or not number.strip() for number in expected
        ):
            raise ValueError("expected_articles must be a list of nonempty strings.")
        if len(expected) != len(set(expected)):
            raise ValueError("Duplicate expected articles.")
        if set(expected) - article_numbers:
            raise ValueError("Expected article does not exist in the processed law.")
        if "expected_answer" not in question:
            raise ValueError("Missing expected_answer.")
        answer = question["expected_answer"]
        if question["type"] == "unanswerable":
            if answer is not None or expected:
                raise ValueError("Unanswerable questions need null answers and no articles.")
        elif not isinstance(answer, str) or not answer.strip() or not expected:
            raise ValueError("Answerable and synthesis questions need answers and articles.")


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark ground-truth records offline.")
    parser.add_argument(
        "questions_path", nargs="?", type=Path,
        default=ROOT / "data/evaluation/benchmark_questions.json",
    )
    args = parser.parse_args()
    try:
        questions = json.loads(args.questions_path.read_text(encoding="utf-8"))
        articles = json.loads(
            (ROOT / "data/processed/law_articles.json").read_text(encoding="utf-8")
        )
        validate_benchmark_questions(questions, articles)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Validation failed: {error}\n")
    print("Validated 18 benchmark questions against the processed law.")


if __name__ == "__main__":
    main()
