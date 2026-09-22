import json
import unittest
from collections import Counter
from pathlib import Path

from src.evaluation.benchmark_validation import validate_benchmark_questions


def sample_questions():
    return [
        {
            "id": str(index), "language": "hy" if index < 6 else "en",
            "type": "answerable" if index < 12 else "synthesis" if index < 15 else "unanswerable",
            "question": "Synthetic validation fixture",
            "expected_answer": "Synthetic reference" if index < 15 else None,
            "expected_articles": ["1"] if index < 15 else [],
        }
        for index in range(18)
    ]


class BenchmarkValidationTests(unittest.TestCase):
    def test_final_dataset_preserves_retrieval_questions_and_matches_law(self):
        root = Path(__file__).resolve().parents[1]
        evaluation = root / "data/evaluation"
        original = json.loads((evaluation / "retrieval_questions.json").read_text(encoding="utf-8"))
        benchmark = json.loads((evaluation / "benchmark_questions.json").read_text(encoding="utf-8"))
        articles = json.loads((root / "data/processed/law_articles.json").read_text(encoding="utf-8"))
        validate_benchmark_questions(benchmark, articles)
        self.assertEqual(
            [{key: value for key, value in record.items() if key != "expected_answer"}
             for record in benchmark],
            original,
        )
        self.assertEqual(Counter(record["type"] for record in benchmark), {
            "answerable": 12, "synthesis": 3, "unanswerable": 3,
        })
        self.assertEqual(Counter(
            record["language"] for record in benchmark if record["type"] == "answerable"
        ), {"hy": 6, "en": 6})

    def test_valid_records(self):
        validate_benchmark_questions(sample_questions(), [{"article_number": "1"}])

    def test_invalid_count_and_records(self):
        for records in (None, {}, [], sample_questions()[:17], sample_questions() + [{}]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                validate_benchmark_questions(records, [{"article_number": "1"}])
        records = sample_questions()
        records[0] = None
        with self.assertRaises(ValueError):
            validate_benchmark_questions(records, [{"article_number": "1"}])

    def test_invalid_fields(self):
        for field, value in (
            ("id", "1"), ("id", ""), ("question", " "), ("language", "fr"),
            ("type", "other"), ("expected_answer", None), ("expected_answer", " "),
            ("expected_answer", 1), ("expected_articles", []),
            ("expected_articles", "1"), ("expected_articles", [1]),
            ("expected_articles", ["999"]), ("expected_articles", ["1", "1"]),
        ):
            records = sample_questions()
            records[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_benchmark_questions(records, [{"article_number": "1"}])

    def test_missing_fields(self):
        for field in sample_questions()[0]:
            records = sample_questions()
            del records[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_benchmark_questions(records, [{"article_number": "1"}])

    def test_synthesis_and_unanswerable_requirements(self):
        for index, field, value in (
            (12, "expected_answer", ""), (12, "expected_articles", []),
            (15, "expected_answer", "Outside answer"), (15, "expected_articles", ["1"]),
        ):
            records = sample_questions()
            records[index][field] = value
            with self.subTest(index=index, field=field), self.assertRaises(ValueError):
                validate_benchmark_questions(records, [{"article_number": "1"}])

    def test_article_existence_uses_supplied_law(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            validate_benchmark_questions(sample_questions(), [{"article_number": "2"}])


if __name__ == "__main__":
    unittest.main()
