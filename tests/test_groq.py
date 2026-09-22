import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from groq import RateLimitError

from src.providers.base import ProviderResponse
from src.providers.groq import GroqProvider, response_format
from src.rag import SYSTEM_INSTRUCTION, ask, assemble_context


class GroqTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict("os.environ", {"GROQ_API_KEY": "secret-test-key", "GROQ_MODEL": "test-model"})
        environment.start()
        self.addCleanup(environment.stop)
        dotenv = patch("src.providers.groq.load_dotenv")
        dotenv.start()
        self.addCleanup(dotenv.stop)
        client = patch("src.providers.groq.Groq")
        self.client = client.start()
        self.addCleanup(client.stop)
        self.create = self.client.return_value.__enter__.return_value.chat.completions.create
        self.response = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"answer":"Answer", "citations":["45"]}'))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )
        self.create.return_value = self.response
        self.provider = GroqProvider()

    def generate(self):
        return self.provider.generate("Question", "Context", SYSTEM_INSTRUCTION, ["45"])

    def test_languages_context_usage_and_latency(self):
        for question, answer in [(" Who is an operator? ", "An operator is an authorized person."),
                                 ("Ո՞վ է օպերատորը:", "Օպերատորը լիազորված անձ է։")]:
            with self.subTest(question=question):
                context = "[Article 45]\nTitle: Վերնագիր\nText:\n Բնագիր։\n"
                self.response.choices[0].message.content = json.dumps({"answer": answer, "citations": ["45"]})
                with patch("src.providers.groq.perf_counter", side_effect=[1.0, 1.125]):
                    result = self.provider.generate(question, context, SYSTEM_INSTRUCTION, ["45", "45"])
                self.assertIsNone(result.error)
                self.assertEqual(result.answer, answer)
                self.assertEqual(result.raw_answer, answer)
                self.assertEqual(result.citations, ["45"])
                self.assertEqual((result.prompt_tokens, result.completion_tokens), (100, 20))
                self.assertEqual(result.total_latency_ms, 125)
                arguments = self.create.call_args.kwargs
                self.assertEqual(arguments["messages"][0], {"role": "system", "content": SYSTEM_INSTRUCTION})
                self.assertEqual(json.loads(arguments["messages"][1]["content"]), {
                    "question": question, "legal_context": context, "allowed_citations": ["45"],
                })
                self.assertEqual(arguments["response_format"], {"type": "json_object"})
                self.assertEqual(arguments["model"], "test-model")
                self.client.assert_called_with(api_key="secret-test-key", max_retries=0)

    def test_strict_output_for_supported_model(self):
        self.provider.model = "openai/gpt-oss-20b"
        self.generate()
        schema = self.create.call_args.kwargs["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertEqual(schema["schema"]["properties"]["citations"]["items"]["enum"], ["45"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        empty = response_format(self.provider.model, [])["json_schema"]["schema"]
        self.assertEqual(empty["properties"]["citations"]["maxItems"], 0)

    def test_invalid_citation_preserves_raw_response(self):
        self.response.choices[0].message.content = '{"answer":"Original", "citations":["45", "99"]}'
        result = self.generate()
        self.assertIn("99", result.error)
        self.assertIsNone(result.answer)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.raw_answer, "Original")
        self.assertEqual(result.raw_citations, ["45", "99"])

    def test_unanswerable_empty_citations(self):
        self.response.choices[0].message.content = '{"answer":"Insufficient information", "citations":[]}'
        result = self.generate()
        self.assertIsNone(result.error)
        self.assertEqual(result.citations, [])

    def test_malformed_output(self):
        for content in [None, "not JSON", "[]", '{"answer":"Answer"}', '{"answer":"Answer","citations":[45]}',
                        '{"answer":"Answer","citations":[],"extra":true}']:
            with self.subTest(content=content):
                self.response.choices[0].message.content = content
                result = self.generate()
                self.assertIn("malformed", result.error)
                self.assertIsNone(result.answer)

    def test_missing_usage_and_incomplete_output(self):
        self.response.usage = None
        result = self.generate()
        self.assertIsNone(result.prompt_tokens)
        self.assertIsNone(result.completion_tokens)
        self.response.choices[0].finish_reason = "length"
        self.assertIsNotNone(self.generate().error)

    def test_rate_limit_error_does_not_leak_secrets(self):
        response = httpx.Response(429, headers={"retry-after": "12", "authorization": "private-auth"},
                                  request=httpx.Request("POST", "https://example.test", content="private-body"))
        self.create.side_effect = RateLimitError("secret-test-key private-auth private-body", response=response, body={})
        result = self.generate()
        self.assertEqual(result.error, "Groq request failed (RateLimitError; status=429; retry_after=12).")
        for secret in ["secret-test-key", "private-auth", "private-body"]:
            self.assertNotIn(secret, result.error)
        self.assertIsNotNone(result.total_latency_ms)

    def test_missing_configuration(self):
        self.provider.api_key = ""
        self.assertIn("GROQ_API_KEY", self.generate().error)
        self.client.assert_not_called()


class ProviderSelectionTests(unittest.TestCase):
    @patch("src.rag.retrieve")
    def test_selection_preserves_retrieval_and_grounding(self, retrieve):
        results = [{"rank": 1, "article_number": "2", "title": "Title", "text": "Text", "chunk_id": "article_2_chunk_1"}]
        retrieve.return_value = results
        for name, target in [("gemini", "src.providers.gemini.GeminiProvider"), ("groq", "src.providers.groq.GroqProvider")]:
            with self.subTest(provider=name), patch(target) as factory:
                factory.return_value.generate.return_value = ProviderResponse(name, "test-model", "Answer", ["2"])
                result = ask("Question", provider=name)
                self.assertEqual(result["provider"], name)
                factory.return_value.generate.assert_called_once_with("Question", assemble_context(results), SYSTEM_INSTRUCTION, ["2"])
                retrieve.assert_called_with("Question", top_k=5)

    @patch("src.rag.retrieve")
    def test_unknown_provider_is_rejected_before_retrieval(self, retrieve):
        with self.assertRaises(ValueError):
            ask("Question", provider="unknown")
        retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
