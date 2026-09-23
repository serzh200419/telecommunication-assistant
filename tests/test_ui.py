import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.ui import SUMMARY_PATH, format_metric, load_benchmark_tables


class BenchmarkDisplayTests(unittest.TestCase):
    def test_formatting(self):
        for value, kind, expected in (
            (0.888888, "percent", "88.89%"), (1749.83, "seconds", "1.75 s"),
            (48382, "tokens", "48,382"), (0, "money", "$0.00"),
            (0.0040148, "money", "$0.0040148"), (None, "seconds", "N/A"),
        ):
            self.assertEqual(format_metric(value, kind), expected)

    def test_summary_tables_use_saved_values(self):
        comparison, breakdown = load_benchmark_tables()
        data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(comparison), 3)
        self.assertEqual(len(comparison[0]), 16)
        self.assertEqual(len(breakdown), 4)
        self.assertEqual(comparison[0]["Answer accuracy"],
                         f"{data['providers']['gemini']['normalized_answer_accuracy']:.2%}")

    def test_missing_and_malformed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            with self.assertRaisesRegex(ValueError, "missing"):
                load_benchmark_tables(path)
            for content in ("invalid", "null", "[]", "{}", '{"providers": []}'):
                path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "malformed"):
                    load_benchmark_tables(path)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app_path = Path(__file__).resolve().parents[1] / "app.py"

    def test_startup_and_blank_question_make_no_calls(self):
        with patch("src.rag.ask") as ask, patch("src.evaluation.benchmark.run_benchmark") as run:
            app = AppTest.from_file(str(self.app_path)).run()
            self.assertFalse(app.exception)
            self.assertEqual([tab.label for tab in app.tabs], ["Ask the Law", "Benchmark"])
            self.assertEqual(len(app.dataframe), 2)
            app.button[0].click().run()
            self.assertTrue(app.warning)
            ask.assert_not_called()
            app.run()
            run.assert_not_called()

    def test_benchmark_runs_only_on_explicit_submit(self):
        original_summary = SUMMARY_PATH.read_bytes()
        with patch("src.evaluation.benchmark.run_benchmark", return_value={
            "attempted": 18, "successful": 18, "failed": 0,
        }) as run, patch("src.rag.ask") as ask:
            app = AppTest.from_file(str(self.app_path)).run()
            app.selectbox[1].select("mistral")
            app.number_input[0].set_value(2.5)
            app.run()
            run.assert_not_called()
            app.button[1].click().run()
            run.assert_called_once_with(provider="mistral", delay_seconds=2.5)
            self.assertFalse(app.exception)
            self.assertTrue(any("require human review" in message.value for message in app.success))
            self.assertEqual(len(app.dataframe), 2)
            app.run()
            run.assert_called_once()
            ask.assert_not_called()
        self.assertEqual(SUMMARY_PATH.read_bytes(), original_summary)

    def test_benchmark_failure_noop_and_unexpected_error(self):
        with patch("src.evaluation.benchmark.run_benchmark") as run:
            app = AppTest.from_file(str(self.app_path)).run()
            run.return_value = {"attempted": 1, "successful": 0, "failed": 1}
            app.button[1].click().run()
            self.assertTrue(any("1 failed calls" in message.value for message in app.error))
            run.return_value = {"attempted": 0, "successful": 0, "failed": 0}
            app.button[1].click().run()
            self.assertTrue(any("no API calls" in message.value for message in app.success))
            run.side_effect = RuntimeError("private credential")
            app.button[1].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("stopped unexpectedly" in message.value for message in app.error))
            self.assertFalse(any("private credential" in message.value for message in app.error))
            self.assertEqual(len(app.dataframe), 2)

    def test_question_and_provider_pass_through_and_answer_persists(self):
        result = {"provider": "groq", "model": "test-model", "answer": "Պատասխան",
                  "citations": ["45"], "error": None, "retrieved_results": []}
        with patch.dict(os.environ, {"GROQ_API_KEY": "test", "GROQ_MODEL": "test-model"}), \
                patch("src.rag.ask", return_value=result) as ask:
            app = AppTest.from_file(str(self.app_path)).run()
            app.text_area[0].input("Ո՞վ է օպերատորը:")
            app.selectbox[0].select("groq")
            app.button[0].click().run()
            self.assertFalse(app.exception)
            ask.assert_called_once_with("Ո՞վ է օպերատորը:", provider="groq")
            self.assertTrue(any(text.value == "Պատասխան" for text in app.text))
            self.assertTrue(any("Article 45" in text.value for text in app.markdown))
            app.run()
            ask.assert_called_once()

    def test_abstention_and_errors(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test", "GEMINI_MODEL": "test-model"}), \
                patch("src.rag.ask") as ask:
            app = AppTest.from_file(str(self.app_path)).run()
            ask.return_value = {"provider": "gemini", "model": "test-model",
                                "answer": "Insufficient information", "citations": [],
                                "error": None, "retrieved_results": []}
            app.text_area[0].input("What is the tax rate?")
            app.button[0].click().run()
            self.assertTrue(any("No citations returned" in message.value for message in app.info))
            ask.return_value = {**ask.return_value, "error": "Rate-limit failure"}
            app.button[0].click().run()
            self.assertEqual(app.error[0].value, "Rate-limit failure")
            ask.side_effect = RuntimeError("private credential")
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertNotIn("private credential", app.error[0].value)

    def test_missing_configuration(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GEMINI_MODEL": ""}), patch("src.rag.ask") as ask:
            app = AppTest.from_file(str(self.app_path)).run()
            app.text_area[0].input("Who is an operator?")
            app.button[0].click().run()
            self.assertTrue(any("not configured" in error.value for error in app.error))
            ask.assert_not_called()

    def test_meta_answer_without_configuration_is_not_abstention(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GEMINI_MODEL": ""}), \
                patch("src.rag.retrieve") as retrieve:
            app = AppTest.from_file(str(self.app_path)).run()
            app.text_area[0].input("Who are you?")
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertFalse(app.info)
            self.assertTrue(any("internal legal assistant" in text.value for text in app.text))
            retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
