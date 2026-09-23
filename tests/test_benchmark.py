import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import call, patch

from src.evaluation import benchmark


def response(provider="groq", error=None, citations=None):
    return {
        "provider": provider, "model": "configured-test-model",
        "answer": "Supported answer" if error is None else None,
        "citations": ["55"] if citations is None else citations,
        "retrieved_articles": ["55", "55"],
        "retrieved_results": [{
            "article_number": "55", "rank": 1, "score": 0.9,
            "chunk_id": "article_55_chunk_1", "title": "Title", "text": "Source context",
        }],
        "prompt_tokens": 100, "completion_tokens": 20, "total_latency_ms": 250.0,
        "error": error, "raw_answer": "Raw answer", "raw_citations": ["55"],
    }


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / "groq_benchmark.json"
        mocked = patch("src.evaluation.benchmark.rag.ask", return_value=response())
        self.ask = mocked.start()
        self.addCleanup(mocked.stop)
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def run_benchmark(self, question_id=None):
        return benchmark.run_benchmark("groq", question_id, self.directory)

    def read_results(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_dataset_loads(self):
        questions = benchmark.load_questions()
        self.assertEqual(len(questions), 18)
        self.assertEqual(questions[0]["id"], "hy_001")
        self.assertIsNone(questions[-1]["expected_answer"])
        self.ask.assert_not_called()

    def test_all_questions_sequential_and_saved_after_each_call(self):
        questions = benchmark.load_questions()
        calls = []

        def answer(question, provider):
            saved = self.read_results() if self.path.exists() else []
            self.assertEqual(len(saved), len(calls))
            self.assertEqual([record["question_id"] for record in saved], calls)
            calls.append(questions[len(calls)]["id"])
            return response(provider)

        self.ask.side_effect = answer
        summary = self.run_benchmark()
        self.assertEqual(self.ask.call_args_list, [
            call(question["question"], provider="groq") for question in questions
        ])
        self.assertEqual(len(self.read_results()), 18)
        self.assertEqual(summary["attempted"], 18)
        self.assertEqual(summary["successful"], 18)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["total_prompt_tokens"], 1800)

    def test_single_success_preserves_metadata_and_review_placeholders(self):
        summary = self.run_benchmark("hy_003")
        records = self.read_results()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["question_id"], "hy_003")
        self.assertEqual(record["status"], "success")
        for field, value in response().items():
            self.assertEqual(record[field], value)
        for field in ("ttft_ms", "answer_accuracy_score", "hallucination", "reviewer_rationale"):
            self.assertIsNone(record[field])
        self.assertIsNotNone(datetime.fromisoformat(record["timestamp"]).tzinfo)
        self.assertEqual(record["citation_f1"], 1.0)
        self.assertEqual(summary["average_total_latency_ms"], 250)
        self.ask.assert_called_once_with(record["question"], provider="groq")

    def test_failures_saved_with_metadata_and_null_metrics(self):
        for error in (
            "Groq request failed (status=429).", "API error", "timeout",
            "malformed structured output", "invalid structured response",
        ):
            with self.subTest(error=error):
                self.ask.return_value = response(error=error)
                summary = self.run_benchmark("hy_003")
                record = self.read_results()[0]
                self.assertEqual(record["status"], "failed")
                for field, value in response(error=error).items():
                    self.assertEqual(record[field], value)
                for field in ("citation_precision", "citation_recall", "citation_f1",
                              "answer_accuracy_score", "hallucination", "reviewer_rationale"):
                    self.assertIsNone(record[field])
                self.assertEqual(summary["failed"], 1)
                self.assertIsNone(summary["average_total_latency_ms"])
                self.assertEqual(summary["total_prompt_tokens"], 100)

    def test_resume_skips_successes_retries_failures_and_retains_other_records(self):
        self.run_benchmark("hy_001")
        first = self.read_results()[0]
        self.ask.return_value = response(error="timeout")
        self.run_benchmark("hy_003")
        self.ask.reset_mock()
        self.ask.return_value = response()
        self.run_benchmark("hy_001")
        self.ask.assert_not_called()
        self.run_benchmark("hy_003")
        self.ask.assert_called_once()
        records = self.read_results()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0], first)
        self.assertEqual(records[1]["status"], "success")

    def test_rate_limit_saved_before_stop_and_resume(self):
        self.ask.side_effect = [response(), response(error="Gemini request failed (status=429).")]
        summary = self.run_benchmark()
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(len(self.read_results()), 2)
        self.ask.side_effect = None
        self.ask.reset_mock()
        summary = self.run_benchmark()
        self.assertEqual(summary["attempted"], 17)
        self.assertEqual(self.ask.call_count, 17)
        self.assertTrue(all(record["status"] == "success" for record in self.read_results()))

    def test_rate_limit_formats(self):
        for error in ("RateLimitError", "status=429", "RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED"):
            self.assertTrue(benchmark.is_rate_limit(error))
        self.assertFalse(benchmark.is_rate_limit("status=500"))

    def test_unexpected_exception_retains_completed_results(self):
        self.ask.side_effect = [response(), RuntimeError("unexpected failure")]
        with self.assertRaises(RuntimeError):
            self.run_benchmark()
        self.assertEqual(len(self.read_results()), 1)

    def test_failed_replacement_retains_previous_file(self):
        self.run_benchmark("hy_001")
        saved = self.path.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("Cannot replace file")):
            with self.assertRaises(OSError):
                self.run_benchmark("hy_003")
        self.assertEqual(self.path.read_bytes(), saved)

    def test_other_returned_errors_continue_and_summary_excludes_failed_scores(self):
        self.ask.side_effect = [response(error="timeout")] + [response() for _ in range(17)]
        summary = self.run_benchmark()
        records = self.read_results()
        self.assertEqual(len(records), 18)
        self.assertEqual(summary["successful"], 17)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["total_prompt_tokens"], 1800)
        self.assertAlmostEqual(
            summary["average_citation_f1"],
            sum(record["citation_f1"] for record in records[1:]) / 17,
        )

    def test_citation_metrics_follow_rubric(self):
        cases = [
            ([], [], (1, 1, 1)), ([], ["55"], (0, 0, 0)),
            (["55"], [], (0, 0, 0)), (["49"], ["55"], (0, 0, 0)),
            (["49", "49", "50", "55"], ["49", "50", "50"], (2 / 3, 1, 0.8)),
            (["49"], ["49", "50"], (1, 0.5, 2 / 3)),
        ]
        for returned, expected, scores in cases:
            with self.subTest(returned=returned, expected=expected):
                metrics = benchmark.citation_metrics(returned, expected)
                for actual, score in zip(metrics.values(), scores):
                    self.assertAlmostEqual(actual, score)

    def test_unanswerable_scores_and_optional_ttft(self):
        for identifier, citations, score in (("adv_001", [], 1), ("adv_002", ["55"], 0)):
            self.ask.return_value = {**response(citations=citations), "ttft_ms": 12.5}
            self.run_benchmark(identifier)
            record = self.read_results()[-1]
            self.assertEqual(record["citation_precision"], score)
            self.assertEqual(record["citation_recall"], score)
            self.assertEqual(record["citation_f1"], score)
            self.assertEqual(record["ttft_ms"], 12.5)
            self.assertIsNone(record["answer_accuracy_score"])
            self.assertIsNone(record["hallucination"])

    def test_missing_measurements_are_not_zero(self):
        self.ask.return_value = {**response(), "prompt_tokens": None,
                                 "completion_tokens": None, "total_latency_ms": None}
        summary = self.run_benchmark("hy_003")
        self.assertIsNone(summary["total_prompt_tokens"])
        self.assertIsNone(summary["total_completion_tokens"])
        self.assertIsNone(summary["average_total_latency_ms"])
        self.assertEqual(summary["prompt_tokens_reported_calls"], 0)

    def test_invalid_selection_and_corrupt_cache_do_not_call_rag(self):
        with self.assertRaises(ValueError):
            self.run_benchmark("unknown")
        with self.assertRaises(ValueError):
            benchmark.run_benchmark("unknown", results_dir=self.directory)
        self.path.write_text("invalid JSON", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_benchmark()
        self.ask.assert_not_called()
        self.assertEqual(self.path.read_text(encoding="utf-8"), "invalid JSON")

    def test_changed_cache_dataset_or_provider_rejected(self):
        self.run_benchmark("hy_003")
        original = self.read_results()
        self.ask.reset_mock()
        for field, value in (("provider", "mistral"), ("question", "Changed question")):
            altered = [{**original[0], field: value}]
            self.path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.run_benchmark()
        self.ask.assert_not_called()

    def test_cli_requires_provider_and_forwards_question_id(self):
        with patch("sys.argv", ["benchmark"]), redirect_stdout(io.StringIO()), \
                patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
            benchmark.main()
        self.assertEqual(error.exception.code, 2)
        self.ask.assert_not_called()
        for provider in benchmark.PROVIDERS:
            with patch("sys.argv", ["benchmark", "--provider", provider, "--question-id", "hy_003"]), \
                    patch.object(benchmark, "run_benchmark", return_value={"failed": 0}) as run:
                benchmark.main()
                run.assert_called_once_with(provider=provider, question_id="hy_003", delay_seconds=0.0)
        with patch("sys.argv", ["benchmark", "--provider", "groq", "--delay-seconds", "2.5"]), \
                patch.object(benchmark, "run_benchmark", return_value={"failed": 0}) as run:
            benchmark.main()
            run.assert_called_once_with(provider="groq", question_id=None, delay_seconds=2.5)


if __name__ == "__main__":
    unittest.main()
