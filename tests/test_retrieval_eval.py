import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.evaluation.retrieval_eval import evaluate_questions, main, validate_questions


def question(identifier="test_1", kind="answerable", expected=None):
    return {
        "id": identifier, "language": "en", "type": kind,
        "question": "Synthetic test question", "expected_articles": ["43"] if expected is None else expected,
    }


def retrieval_results(articles):
    return [
        {"rank": index + 1, "score": 0.9 - index * 0.1, "article_number": article,
         "chunk_id": f"article_{article}_chunk_{index + 1}", "title": "Test title", "text": "Test text"}
        for index, article in enumerate(articles)
    ]


class RetrievalEvaluationTests(unittest.TestCase):
    @patch("src.evaluation.retrieval_eval.retrieve", return_value=[])
    def test_shared_production_preprocessing(self, retrieve):
        from src.query_preprocessing import retrieval_query

        record = {**question(), "question": "Who is operator?"}
        with patch("src.evaluation.retrieval_eval.retrieval_query", wraps=retrieval_query) as preprocess:
            evaluate_questions([record])
        preprocess.assert_called_once_with(record["question"])
        retrieve.assert_called_once_with("Who is operator in electronic communications?", top_k=3)

    @patch("src.evaluation.retrieval_eval.retrieve")
    def test_unique_article_ranking_and_reciprocal_rank(self, retrieve):
        retrieve.return_value = retrieval_results(["45", "45", "43", "2", "7"])
        report = evaluate_questions([question()])
        result = report["results"][0]
        self.assertEqual(result["unique_articles"], ["45", "43", "2", "7"])
        self.assertEqual(result["expected_article_ranks"], {"43": 2})
        self.assertEqual(result["metrics"], {
            "recall_at_1": 0.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "reciprocal_rank": 0.5,
        })
        self.assertEqual(report["summary"]["first_relevant_below_rank_1"], ["test_1"])
        retrieve.assert_called_once_with("Synthetic test question", top_k=3)

    @patch("src.evaluation.retrieval_eval.retrieve")
    def test_synthesis_recall_and_coverage(self, retrieve):
        retrieve.return_value = retrieval_results(["50", "54", "49", "2", "55"])
        report = evaluate_questions([question(kind="synthesis", expected=["50", "49", "55", "60"])])
        result = report["results"][0]
        self.assertEqual(result["expected_article_ranks"], {"50": 1, "49": 3, "55": 5, "60": None})
        self.assertEqual(result["metrics"], {
            "recall_at_1": 0.25, "recall_at_3": 0.5, "recall_at_5": 0.75,
            "reciprocal_rank": 1.0, "expected_articles_found_at_5": 3, "coverage_at_5": 0.75,
        })
        self.assertEqual(report["summary"]["average_synthesis_coverage_at_5"], 0.75)

    @patch("src.evaluation.retrieval_eval.retrieve")
    def test_unanswerable_excluded_and_macro_averages(self, retrieve):
        raw = retrieval_results(["45", "43"])
        retrieve.return_value = raw
        report = evaluate_questions([
            question("hit"), question("miss", expected=["60"]),
            question("unanswerable", "unanswerable", []),
        ])
        summary = report["summary"]
        self.assertEqual(summary["question_count"], 3)
        self.assertEqual(summary["answerable_synthesis_count"], 2)
        self.assertEqual(summary["unanswerable_count"], 1)
        self.assertEqual(summary["recall_at_3"], 0.5)
        self.assertEqual(summary["recall_at_5"], 0.5)
        self.assertEqual(summary["mrr"], 0.25)
        self.assertEqual(summary["no_expected_article_at_5"], ["miss"])
        self.assertEqual(summary["first_relevant_below_rank_1"], ["hit"])
        unanswerable = report["results"][2]
        self.assertIsNone(unanswerable["metrics"])
        self.assertEqual(unanswerable["top_result_score"], 0.9)
        self.assertEqual(unanswerable["retrieved_results"], raw)

    @patch("src.evaluation.retrieval_eval.retrieve", return_value=[])
    def test_only_unanswerable_and_empty_retrieval(self, retrieve):
        report = evaluate_questions([question(kind="unanswerable", expected=[])])
        for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "average_synthesis_coverage_at_5"):
            self.assertIsNone(report["summary"][key])
        self.assertIsNone(report["results"][0]["top_result_score"])
        report = evaluate_questions([question()])
        self.assertEqual(report["summary"]["mrr"], 0)
        self.assertEqual(report["summary"]["no_expected_article_at_5"], ["test_1"])

    @patch("src.evaluation.retrieval_eval.retrieve")
    def test_invalid_schema_before_retrieval(self, retrieve):
        invalid = [None, {}, [], [None], [question(), question()]]
        for field, value in [
            ("id", ""), ("language", "fr"), ("type", "unknown"), ("question", "  "),
            ("expected_articles", "43"), ("expected_articles", [43]),
            ("expected_articles", ["43", "43"]), ("expected_articles", []),
        ]:
            invalid.append([{**question(), field: value}])
        invalid.extend([[question(kind="synthesis", expected=[])], [question(kind="unanswerable")]])
        missing = question()
        del missing["expected_articles"]
        invalid.append([missing])
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                evaluate_questions(records)
        retrieve.assert_not_called()

    @patch("src.evaluation.retrieval_eval.retrieve")
    def test_cli_writes_report(self, retrieve):
        retrieve.return_value = retrieval_results(["43"])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "questions.json"
            output = Path(directory) / "results.json"
            source.write_text(json.dumps([question()]), encoding="utf-8")
            with patch("sys.argv", ["retrieval_eval", str(source)]), patch(
                "src.evaluation.retrieval_eval.OUTPUT_PATH", output
            ), redirect_stdout(io.StringIO()) as stdout:
                main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["mrr"], 1.0)
            self.assertEqual(report["results"][0]["retrieved_results"], retrieve.return_value)
            self.assertIn("Recall@5: 1.0000", stdout.getvalue())

    def test_example_contains_only_two_marked_records(self):
        path = Path(__file__).resolve().parents[1] / "data/evaluation/retrieval_questions.example.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        validate_questions(records)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["question"].startswith("EXAMPLE ONLY:") for record in records))


if __name__ == "__main__":
    unittest.main()
