import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.providers.base import ProviderResponse
from src.providers.gemini import GeminiProvider, answer_schema
from src.rag import SYSTEM_INSTRUCTION, ask, assemble_context, main


class RagTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            {"rank": 2, "article_number": "45", "title": "Title", "chunk_id": "article_45_chunk_2", "text": "Second\nparagraph.", "score": 0.7},
            {"rank": 1, "article_number": "45", "title": "Title", "chunk_id": "article_45_chunk_1", "text": " Առաջին տեքստ։\n", "score": 0.8},
        ]
        self.provider = Mock()
        self.provider.generate.return_value = ProviderResponse("test", "test-model", "Supported answer", ["45"])

    def test_context_preserves_fields_and_rank_order(self):
        context = assemble_context(self.results)
        expected = (
            "[Article 45]\nTitle: Title\nChunk: article_45_chunk_1\nText:\n Առաջին տեքստ։\n\n\n"
            "[Article 45]\nTitle: Title\nChunk: article_45_chunk_2\nText:\nSecond\nparagraph."
        )
        self.assertEqual(context, expected)

    @patch("src.rag.retrieve")
    def test_question_passthrough_and_valid_citations(self, retrieve):
        retrieve.return_value = self.results
        for question in [" Ո՞ր դեպքերում կարող է ծառայությունը կասեցվել: ", " What rights do end users have? "]:
            with self.subTest(question=question):
                result = ask(question, self.provider)
                retrieve.assert_called_with(question, top_k=5)
                self.provider.generate.assert_called_with(question, assemble_context(self.results), SYSTEM_INSTRUCTION, ["45"])
                self.assertEqual(result["answer"], "Supported answer")
                self.assertEqual(result["question"], question)
                self.assertEqual(result["citations"], ["45"])
                self.assertIsNone(result["error"])
                self.assertEqual([item["rank"] for item in result["retrieved_results"]], [1, 2])

    @patch("src.rag.retrieve")
    def test_invented_citation_rejected(self, retrieve):
        retrieve.return_value = self.results
        self.provider.generate.return_value.citations = ["45", "99"]
        result = ask("Question", self.provider)
        self.assertEqual(result["error"], "Invalid provider response: citations outside retrieved context: 99")
        self.assertEqual(result["raw_answer"], "Supported answer")
        self.assertEqual(result["raw_citations"], ["45", "99"])
        self.assertIsNone(result["answer"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(self.provider.generate.return_value.citations, ["45", "99"])

    @patch("src.rag.retrieve")
    def test_unanswerable_can_have_no_citations(self, retrieve):
        retrieve.return_value = self.results
        self.provider.generate.return_value = ProviderResponse("test", "test-model", "Insufficient information", [])
        result = ask("Question outside the law", self.provider)
        self.assertIsNone(result["error"])
        self.assertEqual(result["citations"], [])

    @patch("src.rag.retrieve")
    def test_provider_error_retains_retrieval_metadata(self, retrieve):
        retrieve.return_value = self.results
        self.provider.generate.return_value = ProviderResponse("test", "test-model", error="Request failed")
        result = ask("Question", self.provider)
        self.assertEqual(result["error"], "Request failed")
        self.assertEqual(result["retrieved_articles"], ["45", "45"])

    @patch("src.rag.retrieve")
    def test_empty_questions_rejected_before_retrieval(self, retrieve):
        for question in [None, "", " \n"]:
            with self.assertRaises(ValueError):
                ask(question, self.provider)
        retrieve.assert_not_called()

    @patch("src.rag.ask")
    def test_cli_prints_invalid_raw_response(self, mocked_ask):
        from dataclasses import asdict

        mocked_ask.return_value = {
            **asdict(ProviderResponse("test", "test-model", error="Invalid citations: 99",
                                     raw_answer="Original answer", raw_citations=["99"])),
            "question": "Question", "retrieved_articles": ["45"],
        }
        with patch("sys.argv", ["rag", "Question"]), redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit):
                main()
        self.assertIn("Raw answer: Original answer", output.getvalue())
        self.assertIn("Raw citations: ['99']", output.getvalue())


class GeminiTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict("os.environ", {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model"})
        environment.start()
        self.addCleanup(environment.stop)
        dotenv = patch("src.providers.gemini.load_dotenv")
        dotenv.start()
        self.addCleanup(dotenv.stop)
        client = patch("src.providers.gemini.genai.Client")
        self.client_factory = client.start()
        self.addCleanup(client.stop)
        self.create = self.client_factory.return_value.__enter__.return_value.interactions.create
        self.response = SimpleNamespace(
            output_text=json.dumps({"answer": "Answer", "citations": ["45"]}),
            status="completed", usage=SimpleNamespace(total_input_tokens=100, total_output_tokens=20),
        )
        self.create.return_value = self.response
        self.provider = GeminiProvider()

    def test_structured_request_usage_and_latency(self):
        with patch("src.providers.gemini.perf_counter", side_effect=[10.0, 10.125]):
            result = self.provider.generate("Հայերեն հարց", "Context", SYSTEM_INSTRUCTION, ["45", "45", "2"])
        self.assertEqual(result.answer, "Answer")
        self.assertEqual(result.citations, ["45"])
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (100, 20))
        self.assertEqual(result.total_latency_ms, 125)
        request = self.create.call_args.kwargs
        self.assertEqual(json.loads(request["input"]), {
            "question": "Հայերեն հարց", "legal_context": "Context", "allowed_citations": ["45", "2"],
        })
        self.assertEqual(request["generation_config"], {"thinking_level": "low"})
        self.assertEqual(request["response_format"]["schema"]["properties"]["citations"]["items"]["enum"], ["45", "2"])
        self.assertIn("without article references or citation markers", request["system_instruction"])
        self.assertEqual(request["response_format"]["mime_type"], "application/json")
        self.assertEqual(request["response_format"]["schema"]["required"], ["answer", "citations"])
        self.assertEqual(request["system_instruction"], SYSTEM_INSTRUCTION)
        self.assertFalse(request["store"])

    def test_malformed_outputs(self):
        for output in [None, "not json", "[]", '{"answer":"Answer"}',
                       '{"answer":"", "citations":[]}', '{"answer":"Answer", "citations":[45]}',
                       '{"answer":"Answer", "citations":[], "extra":true}']:
            with self.subTest(output=output):
                self.response.output_text = output
                result = self.provider.generate("Question", "Context", SYSTEM_INSTRUCTION, ["45"])
                self.assertIn("malformed structured output", result.error)
                self.assertIsNone(result.answer)
                self.assertEqual(result.prompt_tokens, 100)

    def test_missing_usage_remains_none(self):
        self.response.usage = None
        result = self.provider.generate("Question", "Context", SYSTEM_INSTRUCTION, ["45"])
        self.assertIsNone(result.prompt_tokens)
        self.assertIsNone(result.completion_tokens)

    def test_api_error_is_clean_and_timed(self):
        self.create.side_effect = RuntimeError("Request contains test-key")
        with patch("src.providers.gemini.perf_counter", side_effect=[1.0, 1.25]):
            result = self.provider.generate("Question", "Context", SYSTEM_INSTRUCTION, ["45"])
        self.assertEqual(result.error, "Gemini request failed (RuntimeError).")
        self.assertEqual(result.total_latency_ms, 250)
        self.assertIsNone(result.answer)

    def test_incomplete_response_rejected(self):
        self.response.status = "failed"
        result = self.provider.generate("Question", "Context", SYSTEM_INSTRUCTION, ["45"])
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.answer)

    def test_missing_configuration_does_not_call_api(self):
        self.provider.api_key = ""
        result = self.provider.generate("Question", "Context", SYSTEM_INSTRUCTION, ["45"])
        self.assertIn("GEMINI_API_KEY", result.error)
        self.client_factory.assert_not_called()

    def test_empty_citations_remain_valid(self):
        self.response.output_text = json.dumps({"answer": "Insufficient information", "citations": []})
        result = self.provider.generate("Question", "Context", SYSTEM_INSTRUCTION, ["45"])
        self.assertIsNone(result.error)
        self.assertEqual(result.citations, [])
        schema = self.create.call_args.kwargs["response_format"]["schema"]
        self.assertNotIn("minItems", schema["properties"]["citations"])

    def test_schema_handles_no_allowed_articles(self):
        self.assertEqual(answer_schema([])["properties"]["citations"]["maxItems"], 0)


if __name__ == "__main__":
    unittest.main()
