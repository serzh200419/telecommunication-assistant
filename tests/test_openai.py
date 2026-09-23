import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import OpenAI, RateLimitError

from src.evaluation import benchmark
from src.evaluation.evaluation_summary import PRICING
from src.providers.base import ProviderResponse
from src.providers.openai import OpenAIProvider, response_format
from src.query_preprocessing import retrieval_query
from src.rag import SYSTEM_INSTRUCTION, ask, assemble_context


def delta(text):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def terminal(status="completed", usage=True):
    return SimpleNamespace(type=f"response.{status}", response=SimpleNamespace(
        status=status, error=None,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20) if usage else None,
    ))


class OpenAIProviderTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"OPENAI_API_KEY": "private-test-key", "OPENAI_MODEL": "gpt-5.6-sol"})
        environment.start()
        self.addCleanup(environment.stop)
        dotenv = patch("src.providers.openai.load_dotenv")
        dotenv.start()
        self.addCleanup(dotenv.stop)
        client = patch("src.providers.openai.OpenAI")
        self.client = client.start()
        self.addCleanup(client.stop)
        self.create = self.client.return_value.__enter__.return_value.responses.create
        self.stream = self.create.return_value.__enter__.return_value
        self.stream.__iter__.side_effect = lambda: iter([
            delta('{"answer":"Answer",'), delta('"citations":["2"]}'), terminal(),
        ])
        self.provider = OpenAIProvider()

    def generate(self):
        return self.provider.generate("Who is operator?", "Context", SYSTEM_INSTRUCTION, ["2", "2"])

    def test_missing_key_or_model(self):
        for field in ("api_key", "model"):
            with self.subTest(field=field):
                provider = OpenAIProvider()
                setattr(provider, field, "")
                result = provider.generate("Question", "Context", SYSTEM_INSTRUCTION, ["2"])
                self.assertIn("OPENAI_API_KEY and OPENAI_MODEL", result.error)
                self.assertIsNone(result.ttft_ms)
        self.client.assert_not_called()

    def test_structured_request_and_usage(self):
        result = self.generate()
        self.assertIsNone(result.error)
        self.assertEqual((result.provider, result.model), ("openai", "gpt-5.6-sol"))
        self.assertEqual(result.answer, "Answer")
        self.assertEqual(result.citations, ["2"])
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (100, 20))
        request = self.create.call_args.kwargs
        self.assertEqual(request["instructions"], SYSTEM_INSTRUCTION)
        self.assertEqual(json.loads(request["input"]), {
            "question": "Who is operator?", "legal_context": "Context", "allowed_citations": ["2"],
        })
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertTrue(request["stream"])
        self.assertFalse(request["store"])
        self.assertNotIn("tools", request)
        schema = request["text"]["format"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertEqual(schema["schema"]["properties"]["citations"]["items"]["enum"], ["2"])
        self.assertEqual(response_format([])["schema"]["properties"]["citations"]["maxItems"], 0)
        self.client.assert_called_once_with(api_key="private-test-key", max_retries=0)

    def test_first_nonempty_text_and_stream_completion(self):
        with patch("src.providers.openai.perf_counter", return_value=10.0) as clock:
            def events():
                clock.return_value = 10.1
                yield SimpleNamespace(type="response.created")
                yield SimpleNamespace(type="response.reasoning_summary_text.delta", delta="reasoning")
                yield delta("")
                clock.return_value = 10.25
                yield delta('{"answer":"Answer",')
                clock.return_value = 10.5
                yield delta('"citations":["2"]}')
                yield terminal()
                clock.return_value = 11.0
            self.stream.__iter__.side_effect = events
            result = self.generate()
        self.assertIsNone(result.error)
        self.assertEqual(result.ttft_ms, 250)
        self.assertEqual(result.total_latency_ms, 1000)

    def test_invalid_citation_preserves_raw_response(self):
        self.stream.__iter__.side_effect = lambda: iter([
            delta('{"answer":"Original","citations":["99"]}'), terminal(),
        ])
        result = self.generate()
        self.assertIn("outside retrieved context", result.error)
        self.assertEqual(result.raw_answer, "Original")
        self.assertEqual(result.raw_citations, ["99"])
        self.assertIsNone(result.answer)
        self.assertEqual(result.citations, [])

    def test_malformed_empty_refusal_and_incomplete(self):
        for content in ("not json", "[]", '{"answer":"Answer"}',
                        '{"answer":"Answer","citations":[2]}',
                        '{"answer":"Answer","citations":[],"extra":1}', ""):
            with self.subTest(content=content):
                self.stream.__iter__.side_effect = lambda: iter([delta(content), terminal()])
                result = self.generate()
                self.assertIn("malformed", result.error)
                self.assertEqual(result.prompt_tokens, 100)
                if not content:
                    self.assertIsNone(result.ttft_ms)
        self.stream.__iter__.side_effect = lambda: iter([terminal("incomplete")])
        self.assertIn("did not complete", self.generate().error)
        self.stream.__iter__.side_effect = lambda: iter([SimpleNamespace(type="response.refusal.delta", delta="Refusal"), terminal()])
        self.assertIn("malformed", self.generate().error)
        self.stream.__iter__.side_effect = lambda: iter([delta('{"answer":"Answer","citations":[]}')])
        self.assertIn("did not complete", self.generate().error)

    def test_no_usage_and_empty_citations(self):
        self.stream.__iter__.side_effect = lambda: iter([
            delta('{"answer":"Insufficient information","citations":[]}'), terminal(usage=False),
        ])
        result = self.generate()
        self.assertIsNone(result.error)
        self.assertEqual(result.citations, [])
        self.assertIsNone(result.prompt_tokens)
        self.assertIsNone(result.completion_tokens)

    def test_safe_rate_limit_and_stream_errors(self):
        response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))
        self.create.side_effect = RateLimitError("private-test-key", response=response, body={})
        result = self.generate()
        self.assertIn("status=429", result.error)
        self.assertNotIn("private-test-key", result.error)
        self.assertIsNotNone(result.total_latency_ms)
        self.assertTrue(benchmark.is_rate_limit(result.error))
        self.create.side_effect = None
        self.stream.__iter__.side_effect = lambda: iter([
            SimpleNamespace(type="error", code="insufficient_quota", message="private-test-key"),
        ])
        self.assertTrue(benchmark.is_rate_limit(self.generate().error))

    def test_midstream_failure_preserves_timings(self):
        def events():
            yield delta('{"answer":')
            raise RuntimeError("private-test-key")
        self.stream.__iter__.side_effect = events
        result = self.generate()
        self.assertIsNotNone(result.ttft_ms)
        self.assertIsNotNone(result.total_latency_ms)
        self.assertIsNone(result.answer)
        self.assertEqual(result.error, "OpenAI request failed (RuntimeError).")


class OpenAIIntegrationTests(unittest.TestCase):
    @patch("src.providers.openai.OpenAIProvider")
    @patch("src.rag.retrieve")
    def test_selection_and_meta_bypass(self, retrieve, factory):
        chunks = [{"rank": 1, "article_number": "2", "title": "Title", "text": "Text", "chunk_id": "2_1"}]
        retrieve.return_value = chunks
        factory.return_value.generate.return_value = ProviderResponse("openai", "gpt-5.6-sol", "Answer", ["2"])
        result = ask("Who is operator?", provider="openai")
        self.assertEqual(result["provider"], "openai")
        retrieve.assert_called_once_with(retrieval_query("Who is operator?"), top_k=3)
        factory.return_value.generate.assert_called_once_with("Who is operator?", assemble_context(chunks), SYSTEM_INSTRUCTION, ["2"])
        factory.reset_mock()
        retrieve.reset_mock()
        self.assertEqual(ask("Who are you?", provider="openai")["provider"], "local")
        factory.assert_not_called()
        retrieve.assert_not_called()

    def test_benchmark_openai_resume_and_paid_pricing(self):
        with tempfile.TemporaryDirectory() as directory, patch("src.evaluation.benchmark.rag.ask", return_value={
            "provider": "openai", "model": "gpt-5.6-sol", "answer": "Answer", "citations": ["2"], "error": None,
        }) as mocked, redirect_stdout(io.StringIO()):
            benchmark.run_benchmark("openai", "hy_001", Path(directory), delay_seconds=0)
            path = Path(directory) / "openai_benchmark.json"
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))[0]["provider"], "openai")
            benchmark.run_benchmark("openai", "hy_001", Path(directory))
            mocked.assert_called_once()
        self.assertEqual(PRICING["providers"]["openai"]["actual_benchmark_tier"], "paid")
        prices = PRICING["providers"]["openai"]
        self.assertEqual(prices["input_price_per_million_usd"], 4.00)
        self.assertEqual(prices["output_price_per_million_usd"], 20.00)
        self.assertEqual(prices["pricing_date"], "2026-09-24")

    def test_sdk_rate_limit_has_no_retries(self):
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(429, json={"error": {"message": "private-test-key", "type": "rate_limit_error"}})
        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client, \
                patch.dict(os.environ, {"OPENAI_API_KEY": "private-test-key", "OPENAI_MODEL": "gpt-5.6-sol"}), \
                patch("src.providers.openai.load_dotenv"), \
                patch("src.providers.openai.OpenAI", side_effect=lambda **kwargs: OpenAI(**kwargs, http_client=http_client)):
            result = OpenAIProvider().generate("Question", "Context", SYSTEM_INSTRUCTION, ["2"])
        self.assertEqual(len(requests), 1)
        self.assertIn("status=429", result.error)
        self.assertNotIn("private-test-key", result.error)
        self.assertTrue(requests[0]["stream"])

    def test_installed_sdk_parses_streamed_json_and_usage(self):
        events = [
            {"type": "response.output_text.delta", "delta": "", "sequence_number": 1,
             "item_id": "message", "output_index": 0, "content_index": 0, "logprobs": []},
            {"type": "response.output_text.delta", "delta": '{"answer":"Answer",', "sequence_number": 2,
             "item_id": "message", "output_index": 0, "content_index": 0, "logprobs": []},
            {"type": "response.output_text.delta", "delta": '"citations":["2"]}', "sequence_number": 3,
             "item_id": "message", "output_index": 0, "content_index": 0, "logprobs": []},
            {"type": "response.completed", "sequence_number": 4, "response": {
                "id": "test", "object": "response", "created_at": 1, "status": "completed",
                "model": "gpt-5.6-sol", "output": [], "error": None,
                "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 5}},
            }},
        ]
        def respond(request):
            payload = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload)
        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client, \
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-5.6-sol"}), \
                patch("src.providers.openai.load_dotenv"), \
                patch("src.providers.openai.OpenAI", side_effect=lambda **kwargs: OpenAI(**kwargs, http_client=http_client)):
            result = OpenAIProvider().generate("Question", "Context", SYSTEM_INSTRUCTION, ["2"])
        self.assertIsNone(result.error)
        self.assertEqual(result.answer, "Answer")
        self.assertEqual(result.citations, ["2"])
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (100, 20))
        self.assertIsNotNone(result.ttft_ms)
        self.assertGreaterEqual(result.total_latency_ms, result.ttft_ms)


if __name__ == "__main__":
    unittest.main()
