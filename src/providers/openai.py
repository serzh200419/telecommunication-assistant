import json
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from openai import OpenAI

from src.providers.base import ProviderResponse, validate_answer


def response_format(allowed):
    citations = {"type": "array", "items": {"type": "string"}}
    if allowed:
        citations["items"]["enum"] = list(dict.fromkeys(allowed))
    else:
        citations["maxItems"] = 0
    return {
        "type": "json_schema", "name": "legal_answer", "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}, "citations": citations},
            "required": ["answer", "citations"], "additionalProperties": False,
        },
    }


def format_api_error(error):
    fields = [type(error).__name__]
    status = getattr(error, "status_code", None)
    if type(status) is int and 100 <= status <= 599:
        fields.append(f"status={status}")
    return "OpenAI request failed (" + "; ".join(fields) + ")."


def stream_error(code):
    if code in ("rate_limit_exceeded", "insufficient_quota", "quota_exceeded", "429"):
        return "OpenAI request failed (status=429)."
    return "OpenAI request failed (stream error)."


class OpenAIProvider:
    def __init__(self):
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "").strip()

    def generate(self, question, context, system_instruction, allowed_articles):
        result = ProviderResponse(provider="openai", model=self.model or None)
        if not self.api_key or not self.model:
            result.error = "Set OPENAI_API_KEY and OPENAI_MODEL in the environment or .env file."
            return result
        allowed = list(dict.fromkeys(allowed_articles))
        request_input = json.dumps({
            "question": question, "legal_context": context, "allowed_citations": allowed,
        }, ensure_ascii=False)
        output_format = response_format(allowed)
        start = None
        parts = []
        completed = False
        try:
            with OpenAI(api_key=self.api_key, max_retries=0) as client:
                start = perf_counter()
                with client.responses.create(
                    model=self.model, instructions=system_instruction, input=request_input,
                    text={"format": output_format}, reasoning={"effort": "low"},
                    stream=True, store=False,
                ) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            if isinstance(event.delta, str) and event.delta:
                                if result.ttft_ms is None:
                                    result.ttft_ms = (perf_counter() - start) * 1000
                                parts.append(event.delta)
                        elif event.type in ("response.completed", "response.incomplete", "response.failed"):
                            response = event.response
                            if response.usage is not None:
                                result.prompt_tokens = response.usage.input_tokens
                                result.completion_tokens = response.usage.output_tokens
                            completed = event.type == "response.completed" and response.status == "completed"
                            if event.type == "response.failed":
                                result.error = stream_error(getattr(response.error, "code", None))
                        elif event.type == "error":
                            result.error = stream_error(event.code)
                            break
                    result.total_latency_ms = (perf_counter() - start) * 1000
        except Exception as error:
            if start is not None and result.total_latency_ms is None:
                result.total_latency_ms = (perf_counter() - start) * 1000
            result.error = format_api_error(error)
            return result
        if result.error is not None:
            return result
        if not completed:
            result.error = "OpenAI did not complete the response."
            return result
        try:
            answer = json.loads("".join(parts))
            if isinstance(answer, dict):
                if isinstance(answer.get("answer"), str):
                    result.raw_answer = answer["answer"]
                if isinstance(answer.get("citations"), list) and all(isinstance(item, str) for item in answer["citations"]):
                    result.raw_citations = answer["citations"]
            validate_answer(answer)
        except (ValueError, TypeError):
            result.error = "OpenAI returned malformed structured output; expected answer and citations."
            return result
        invalid = list(dict.fromkeys(citation for citation in answer["citations"] if citation not in allowed))
        if invalid:
            result.error = "Invalid provider response: citations outside retrieved context: " + ", ".join(invalid)
            return result
        result.answer = answer["answer"]
        result.citations = answer["citations"]
        return result
