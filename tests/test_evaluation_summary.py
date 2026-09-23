import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.evaluation import evaluation_summary as summary


class EvaluationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.questions = json.loads(
            (summary.EVALUATION_DIR / "benchmark_questions.json").read_text(encoding="utf-8")
        )
        self.benchmarks = {}
        self.reviews = []
        for provider in summary.PROVIDERS:
            records = []
            for index, question in enumerate(self.questions):
                records.append({
                    **{key: value for key, value in question.items() if key != "id"},
                    "question_id": question["id"], "provider": provider, "model": "test-model",
                    "status": "success", "error": None,
                    "citation_precision": 0.75, "citation_recall": 0.5, "citation_f1": 0.6,
                    "ttft_ms": index + 1.0, "total_latency_ms": (index + 1) * 10.0,
                    "prompt_tokens": 100, "completion_tokens": 20,
                })
                self.reviews.append({
                    "provider": provider, "question_id": question["id"],
                    "answer_accuracy_score": 2, "hallucination": False, "reviewer_rationale": None,
                })
            self.benchmarks[provider] = records

    def aggregate(self):
        return summary.aggregate_results(self.benchmarks, self.reviews, self.questions)

    def test_normalized_accuracy_scores_and_hallucination(self):
        self.reviews[1]["answer_accuracy_score"] = 1
        self.reviews[2]["answer_accuracy_score"] = 0
        for review in self.reviews[:3]:
            review["hallucination"] = True
        row = self.aggregate()["providers"]["gemini"]
        self.assertAlmostEqual(row["normalized_answer_accuracy"], 33 / 36)
        self.assertAlmostEqual(row["average_answer_score"], 33 / 18)
        self.assertEqual([row[f"score_{score}_count"] for score in (2, 1, 0)], [16, 1, 1])
        self.assertEqual(row["hallucination_count"], 3)
        self.assertAlmostEqual(row["hallucination_rate"], 3 / 18)

    def test_abstention_and_question_types(self):
        self.reviews[16]["answer_accuracy_score"] = 1
        self.reviews[17]["answer_accuracy_score"] = 0
        result = self.aggregate()
        row = result["providers"]["gemini"]
        self.assertEqual(row["unanswerable_questions"], 3)
        self.assertEqual(row["correct_abstentions"], 1)
        self.assertAlmostEqual(row["abstention_accuracy"], 1 / 3)
        groups = result["by_question_type"]
        self.assertEqual(groups["unanswerable"]["gemini"]["normalized_answer_accuracy"], 0.5)
        for group, count in zip(summary.GROUPS, (6, 6, 3, 3)):
            self.assertEqual(groups[group]["gemini"]["questions"], count)
        self.assertEqual(groups["armenian_answerable"]["gemini"]["normalized_answer_accuracy"], 1)

    def test_missing_review(self):
        self.reviews.pop()
        with self.assertRaisesRegex(ValueError, "Missing manual review"):
            self.aggregate()

    def test_duplicate_review(self):
        self.reviews[-1] = copy.deepcopy(self.reviews[0])
        with self.assertRaisesRegex(ValueError, "Duplicate manual review"):
            self.aggregate()

    def test_invalid_review_values(self):
        for field, value in (("answer_accuracy_score", None), ("answer_accuracy_score", True),
                             ("answer_accuracy_score", 3), ("answer_accuracy_score", 1.0),
                             ("hallucination", None), ("hallucination", 0), ("hallucination", "false")):
            with self.subTest(field=field, value=value):
                reviews = copy.deepcopy(self.reviews)
                reviews[0][field] = value
                with self.assertRaises(ValueError):
                    summary.aggregate_results(self.benchmarks, reviews, self.questions)

    def test_record_counts_providers_and_duplicates(self):
        invalid = []
        missing_provider = copy.deepcopy(self.benchmarks)
        del missing_provider["groq"]
        invalid.append(missing_provider)
        missing_record = copy.deepcopy(self.benchmarks)
        missing_record["groq"].pop()
        invalid.append(missing_record)
        duplicate = copy.deepcopy(self.benchmarks)
        duplicate["groq"][-1] = copy.deepcopy(duplicate["groq"][0])
        invalid.append(duplicate)
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                summary.aggregate_results(records, self.reviews, self.questions)

    def test_invalid_metrics_and_mismatched_questions(self):
        for field, value in (("citation_f1", None), ("citation_precision", 1.1),
                             ("total_latency_ms", float("nan")), ("ttft_ms", -1),
                             ("prompt_tokens", True), ("completion_tokens", 1.5),
                             ("status", "unknown"), ("language", "en"),
                             ("error", "unexpected error"), ("model", "other-model")):
            records = copy.deepcopy(self.benchmarks)
            records["gemini"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summary.aggregate_results(records, self.reviews, self.questions)

    def test_latency_tokens_and_stored_citations(self):
        row = self.aggregate()["providers"]["gemini"]
        self.assertEqual(row["mean_ttft_ms"], 9.5)
        self.assertEqual(row["median_ttft_ms"], 9.5)
        self.assertEqual(row["mean_total_latency_ms"], 95)
        self.assertEqual(row["median_total_latency_ms"], 95)
        self.assertAlmostEqual(row["p95_total_latency_ms"], 171.5)
        self.assertEqual(row["total_prompt_tokens"], 1800)
        self.assertEqual(row["total_completion_tokens"], 360)
        self.assertEqual(row["total_tokens"], 2160)
        self.assertEqual(row["average_prompt_tokens_per_successful_question"], 100)
        self.assertEqual(row["average_completion_tokens_per_successful_question"], 20)
        self.assertEqual(row["citation_precision"], 0.75)
        self.assertEqual(row["citation_recall"], 0.5)
        self.assertEqual(row["citation_f1"], 0.6)

    def test_failures_are_excluded_from_answer_metrics_not_reliability(self):
        record = self.benchmarks["gemini"][-1]
        record.update(status="failed", error="timeout", citation_f1=None,
                      citation_precision=None, citation_recall=None)
        self.reviews[17].update(answer_accuracy_score=0, hallucination=True)
        row = self.aggregate()["providers"]["gemini"]
        self.assertEqual(row["successful"], 17)
        self.assertEqual(row["failed"], 1)
        self.assertAlmostEqual(row["failure_rate"], 1 / 18)
        self.assertEqual(row["normalized_answer_accuracy"], 1)
        self.assertEqual(row["hallucination_rate"], 0)
        self.assertAlmostEqual(row["abstention_accuracy"], 2 / 3)
        self.assertEqual(row["total_tokens"], 2160)

    def test_no_successes_and_missing_measurements(self):
        for record in self.benchmarks["gemini"]:
            record.update(status="failed", error="timeout", ttft_ms=None, total_latency_ms=None,
                          prompt_tokens=None, completion_tokens=None)
        row = self.aggregate()["providers"]["gemini"]
        for field in ("normalized_answer_accuracy", "hallucination_rate", "citation_f1",
                      "mean_ttft_ms", "p95_total_latency_ms", "total_tokens"):
            self.assertIsNone(row[field])
        self.assertEqual(row["failure_rate"], 1)
        self.assertEqual(row["abstention_accuracy"], 0)
        self.assertEqual(row["measurement_counts"]["ttft_ms"], 0)

    def test_main_writes_json_and_csv_without_changing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            inputs = {root / "benchmark_questions.json": self.questions,
                      root / "manual_review.json": self.reviews}
            inputs.update({root / "results" / f"{provider}_benchmark.json": records
                           for provider, records in self.benchmarks.items()})
            for path, value in inputs.items():
                path.write_text(json.dumps(value), encoding="utf-8")
            before = {path: path.read_bytes() for path in inputs}
            with patch.object(summary, "EVALUATION_DIR", root), patch.object(summary, "ROOT", root), \
                    patch.object(summary, "validate_benchmark_questions"), redirect_stdout(io.StringIO()):
                (root / "data/processed").mkdir(parents=True)
                (root / "data/processed/law_articles.json").write_text("[]", encoding="utf-8")
                summary.main()
            result = json.loads((root / "final_benchmark_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(set(result["providers"]), set(summary.PROVIDERS))
            with (root / "final_benchmark_summary.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 3)
            self.assertEqual(list(rows[0]), list(summary.CSV_FIELDS))
            self.assertEqual(before, {path: path.read_bytes() for path in inputs})


if __name__ == "__main__":
    unittest.main()
