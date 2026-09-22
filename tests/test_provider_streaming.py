import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from google import genai
from groq import Groq
from mistralai.client import Mistral

from src.providers.gemini import GeminiProvider
from src.providers.groq import GroqProvider
from src.providers.mistral import MistralProvider
from src.rag import SYSTEM_INSTRUCTION


PROVIDERS = {
    "gemini": (GeminiProvider, "genai.Client"),
    "groq": (GroqProvider, "Groq"),
    "mistral": (MistralProvider, "Mistral"),
}


def text_event(provider, content, finish=None, usage=None):
    if provider == "gemini":
        if finish:
            return SimpleNamespace(event_type="interaction.completed", interaction=SimpleNamespace(
                status="completed", usage=usage,
            ))
        return SimpleNamespace(event_type="step.delta", delta=SimpleNamespace(type="text", text=content))
    chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)],
        usage=usage, x_groq=None,
    )
    return SimpleNamespace(data=chunk) if provider == "mistral" else chunk


class StreamingTests(unittest.TestCase):
    def generate(self, name, parts, failure=False, complete=True):
        factory, target = PROVIDERS[name]
        with ExitStack() as stack:
            stack.enter_context(patch.dict("os.environ", {
                name.upper() + "_API_KEY": "test-key", name.upper() + "_MODEL": "test-model",
            }))
            stack.enter_context(patch(f"src.providers.{name}.load_dotenv"))
            client = stack.enter_context(patch(f"src.providers.{name}.{target}"))
            clock = stack.enter_context(patch(f"src.providers.{name}.perf_counter", return_value=10.0))
            sdk = client.return_value.__enter__.return_value
            create = (sdk.interactions.create if name == "gemini" else
                      sdk.chat.completions.create if name == "groq" else sdk.chat.stream)
            stream = create.return_value if name == "gemini" else create.return_value.__enter__.return_value

            def events():
                for index, part in enumerate(parts):
                    clock.return_value = 10.0 + (index + 1) / 10
                    yield text_event(name, part)
                clock.return_value = 11.0
                if failure:
                    raise RuntimeError("test-key must not appear in error")
                if complete:
                    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20,
                                            total_input_tokens=100, total_output_tokens=20)
                    yield text_event(name, None, finish="stop", usage=usage)
                clock.return_value = 11.5

            stream.__iter__.side_effect = events
            result = factory().generate("Question", "Context", SYSTEM_INSTRUCTION, ["45"])
            create.assert_called_once()
            return result

    def test_first_nonempty_content_and_complete_stream_timing(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                result = self.generate(provider, [None, "", '{"answer":', '"Answer",', '"citations":["45"]}'])
                self.assertIsNone(result.error)
                self.assertAlmostEqual(result.ttft_ms, 300)
                self.assertEqual(result.total_latency_ms, 1500)
                self.assertEqual(result.answer, "Answer")
                self.assertEqual(result.citations, ["45"])
                self.assertEqual((result.prompt_tokens, result.completion_tokens), (100, 20))

    def test_no_text_has_no_ttft_and_is_not_valid_output(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                result = self.generate(provider, [None, ""])
                self.assertIsNone(result.ttft_ms)
                self.assertIn("malformed", result.error)
                self.assertEqual(result.total_latency_ms, 1500)

    def test_fragmented_malformed_json(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                result = self.generate(provider, ['{"answer":', "invalid}"])
                self.assertIn("malformed", result.error)
                self.assertAlmostEqual(result.ttft_ms, 100)
                self.assertEqual(result.completion_tokens, 20)

    def test_failure_before_and_after_first_content(self):
        for provider in PROVIDERS:
            for parts in ([None, ""], ['{"answer":']):
                with self.subTest(provider=provider, parts=parts):
                    result = self.generate(provider, parts, failure=True)
                    self.assertIn("request failed", result.error)
                    self.assertNotIn("test-key", result.error)
                    self.assertIsNone(result.answer)
                    self.assertEqual(result.total_latency_ms, 1000)
                    if parts[0] is None:
                        self.assertIsNone(result.ttft_ms)
                    else:
                        self.assertAlmostEqual(result.ttft_ms, 100)

    def test_truncated_stream_is_failure_even_with_complete_json(self):
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                result = self.generate(provider, ['{"answer":"Answer","citations":[]}'], complete=False)
                self.assertIn("did not complete", result.error)
                self.assertIsNone(result.prompt_tokens)


def wire_events(provider):
    text = json.dumps({"answer": "Answer", "citations": ["45"]})
    if provider == "gemini":
        return [
            {"event_type": "interaction.created", "interaction": {"id": "test", "status": "in_progress"}},
            {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_summary",
             "content": {"type": "text", "text": "Reasoning is not output"}}},
            *[{"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": part}}
              for part in ("", text[:12], text[12:])],
            {"event_type": "interaction.completed", "interaction": {
                "id": "test", "status": "completed",
                "usage": {"total_input_tokens": 100, "total_output_tokens": 20},
            }},
        ]
    chunks = []
    for part, finish in (("", None), (text[:12], None), (text[12:], None), (None, "stop")):
        chunks.append({
            "id": "test", "object": "chat.completion.chunk", "created": 1, "model": "test-model",
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": finish}],
        })
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    chunks.append({**chunks[-1], "choices": [],
                   **({"x_groq": {"usage": usage}} if provider == "groq" else {"usage": usage})})
    return chunks


class StreamingTransportTests(unittest.TestCase):
    def test_installed_sdks_parse_streams_and_preserve_usage(self):
        for name, (factory, target) in PROVIDERS.items():
            with self.subTest(provider=name), ExitStack() as stack:
                requests = []

                def respond(request):
                    requests.append(json.loads(request.content))
                    data = "".join("data: " + json.dumps(event) + "\n\n" for event in wire_events(name))
                    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                          content=data + "data: [DONE]\n\n")

                http_client = stack.enter_context(httpx.Client(transport=httpx.MockTransport(respond)))
                constructor = {"gemini": genai.Client, "groq": Groq, "mistral": Mistral}[name]
                if name == "gemini":
                    def client(**kwargs):
                        options = kwargs.pop("http_options")
                        return constructor(**kwargs, http_options={**options, "httpx_client": http_client})
                elif name == "groq":
                    def client(**kwargs):
                        return Groq(**kwargs, http_client=http_client)
                else:
                    def client(**kwargs):
                        return Mistral(**kwargs, client=http_client)
                stack.enter_context(patch.dict("os.environ", {
                    name.upper() + "_API_KEY": "test-key", name.upper() + "_MODEL": "test-model",
                }))
                stack.enter_context(patch(f"src.providers.{name}.load_dotenv"))
                stack.enter_context(patch(f"src.providers.{name}.{target}", side_effect=client))
                result = factory().generate("Question", "Context", SYSTEM_INSTRUCTION, ["45"])
                self.assertIsNone(result.error)
                self.assertEqual(result.answer, "Answer")
                self.assertEqual((result.prompt_tokens, result.completion_tokens), (100, 20))
                self.assertIsNotNone(result.ttft_ms)
                self.assertGreaterEqual(result.total_latency_ms, result.ttft_ms)
                self.assertEqual(len(requests), 1)
                self.assertTrue(requests[0]["stream"])


if __name__ == "__main__":
    unittest.main()
