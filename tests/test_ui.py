import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.ui import METRICS, PROVIDERS, GROUPS, format_metric, load_benchmark_tables


def setUpModule():
    global SUMMARY_PATH, summary_directory, loader_patch
    summary_directory = tempfile.TemporaryDirectory()
    SUMMARY_PATH = Path(summary_directory.name) / "summary.json"
    data = {
        "providers": {provider: {"provider": provider, "model": "test-model",
                      **{field: 0 for field, _, _ in METRICS}} for provider in PROVIDERS},
        "by_question_type": {group: {provider: {"normalized_answer_accuracy": 1}
                             for provider in PROVIDERS} for group in GROUPS},
    }
    SUMMARY_PATH.write_text(json.dumps(data), encoding="utf-8")
    loader_patch = patch("src.ui.load_benchmark_tables", side_effect=lambda: load_benchmark_tables(SUMMARY_PATH))
    loader_patch.start()


def tearDownModule():
    loader_patch.stop()
    summary_directory.cleanup()


class BenchmarkDisplayTests(unittest.TestCase):
    def test_formatting(self):
        for value, kind, expected in (
            (0.888888, "percent", "88.89%"), (1749.83, "seconds", "1.75 s"),
            (48382, "tokens", "48,382"), (0, "money", "$0.00"),
            (0.0040148, "money", "$0.0040148"), (None, "seconds", "N/A"),
        ):
            self.assertEqual(format_metric(value, kind), expected)

    def test_summary_tables_use_saved_values(self):
        comparison, breakdown = load_benchmark_tables(SUMMARY_PATH)
        data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(comparison), 4)
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

    def test_openai_summary_and_missing_provider_detection(self):
        data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        data["providers"]["openai"] = {**data["providers"]["gemini"], "provider": "openai", "model": "test-model"}
        for group in data["by_question_type"].values():
            group["openai"] = dict(group["gemini"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            comparison, breakdown = load_benchmark_tables(path)
        self.assertEqual(len(comparison), 4)
        self.assertEqual(comparison[-1]["Provider"], "OpenAI")
        self.assertEqual(set(breakdown[0]), {"Question type", "Gemini", "Groq", "Mistral", "OpenAI"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            del data["providers"]["openai"]
            path.write_text(json.dumps(data), encoding="utf-8")
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
            captions = " ".join(item.value for item in app.caption)
            self.assertIn("Gemini, Groq, and Mistral used free-tier", captions)
            self.assertIn("OpenAI was included as an additional paid comparison", captions)
            self.assertIn("human-reviewed", captions)
            self.assertIn("OpenAI (paid)", app.selectbox[0].options)
            app.button[0].click().run()
            self.assertTrue(app.warning)
            ask.assert_not_called()
            app.run()
            run.assert_not_called()

    def test_benchmark_is_read_only(self):
        original_summary = SUMMARY_PATH.read_bytes()
        with patch("src.evaluation.benchmark.run_benchmark") as run, patch("src.rag.ask") as ask:
            app = AppTest.from_file(str(self.app_path)).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.dataframe[0].value["Provider"].tolist(),
                             ["Gemini", "Groq", "Mistral", "OpenAI"])
            self.assertEqual(app.dataframe[1].value.columns.tolist(),
                             ["Question type", "Gemini", "Groq", "Mistral", "OpenAI"])
            self.assertEqual([item.label for item in app.selectbox], ["Provider"])
            self.assertEqual([item.label for item in app.button], ["Ask"])
            self.assertFalse(app.number_input)
            self.assertNotIn("Run benchmark", [item.value for item in app.subheader])
            self.assertTrue(any("saved benchmark summary" in item.value for item in app.markdown))
            app.run()
            run.assert_not_called()
            ask.assert_not_called()
        self.assertEqual(SUMMARY_PATH.read_bytes(), original_summary)

    def test_app_does_not_import_benchmark_runner(self):
        import ast

        tree = ast.parse(self.app_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "src.evaluation.benchmark")
                if node.module == "src.evaluation":
                    self.assertNotIn("benchmark", [alias.name for alias in node.names])
            elif isinstance(node, ast.Import):
                self.assertNotIn("src.evaluation.benchmark", [alias.name for alias in node.names])

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
