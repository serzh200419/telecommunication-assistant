import json
import os
import re
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from groq import Groq

from src.providers.base import ProviderResponse, validate_answer


def format_api_error(error: Exception) -> str:
    fields = [type(error).__name__]
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
        fields.append(f"status={status}")
    response = getattr(error, "response", None)
    if response is not None:
        value = response.headers.get("retry-after")
        if isinstance(value, str):
            if re.fullmatch(r"\d+(?:\.\d+)?", value):
                fields.append(f"retry_after={value}")
            else:
                try:
                    fields.append("retry_after=" + format_datetime(parsedate_to_datetime(value)))
                except (ValueError, TypeError, OverflowError):
                    pass
    return "Groq request failed (" + "; ".join(fields) + ")."


class GroqProvider:
    def __init__(self):
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model = os.getenv("GROQ_MODEL", "").strip()

    def generate(self, question: str, context: str, system_instruction: str, allowed_articles: list[str]) -> ProviderResponse:
        result = ProviderResponse(provider="groq", model=self.model or None)
        if not self.api_key or not self.model:
            result.error = "Set GROQ_API_KEY and GROQ_MODEL in the environment or .env file."
            return result
        allowed = list(dict.fromkeys(allowed_articles))
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": json.dumps({
                "question": question, "legal_context": context, "allowed_citations": allowed,
            }, ensure_ascii=False)},
        ]
        start = None
        parts = []
        finish_reason = None
        try:
            with Groq(api_key=self.api_key, max_retries=0) as client:
                start = perf_counter()
                # Groq does not support json_schema with streaming.
                with client.chat.completions.create(
                    model=self.model, messages=messages, response_format={"type": "json_object"},
                    stream=True,
                ) as stream:
                    for chunk in stream:
                        for choice in chunk.choices:
                            content = choice.delta.content
                            if isinstance(content, str) and content:
                                if result.ttft_ms is None:
                                    result.ttft_ms = (perf_counter() - start) * 1000
                                parts.append(content)
                            if choice.finish_reason is not None:
                                finish_reason = choice.finish_reason
                        usage = chunk.usage
                        if usage is None and chunk.x_groq is not None:
                            usage = chunk.x_groq.usage
                        if usage is not None:
                            result.prompt_tokens = usage.prompt_tokens
                            result.completion_tokens = usage.completion_tokens
                    result.total_latency_ms = (perf_counter() - start) * 1000
        except Exception as error:
            if start is not None and result.total_latency_ms is None:
                result.total_latency_ms = (perf_counter() - start) * 1000
            result.error = format_api_error(error)
            return result
        if finish_reason != "stop":
            result.error = "Groq did not complete the response."
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
            result.error = "Groq returned malformed structured output; expected answer and citations."
            return result
        invalid = list(dict.fromkeys(citation for citation in answer["citations"] if citation not in allowed))
        if invalid:
            result.error = "Invalid provider response: citations outside retrieved context: " + ", ".join(invalid)
            return result
        result.answer = answer["answer"]
        result.citations = answer["citations"]
        return result
