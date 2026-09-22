import json
import os
import re
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from mistralai.client import Mistral

from src.providers.base import ProviderResponse, validate_answer


def response_format(allowed: list[str]) -> dict:
    citations = {"type": "array", "items": {"type": "string"}}
    if allowed:
        citations["items"]["enum"] = list(dict.fromkeys(allowed))
    else:
        citations["maxItems"] = 0
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "legal_answer", "strict": True,
            "schema_definition": {
                "type": "object",
                "properties": {"answer": {"type": "string"}, "citations": citations},
                "required": ["answer", "citations"], "additionalProperties": False,
            },
        },
    }


def format_api_error(error: Exception) -> str:
    fields = [type(error).__name__]
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
        fields.append(f"status={status}")
    response = getattr(error, "raw_response", None)
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
    return "Mistral request failed (" + "; ".join(fields) + ")."


class MistralProvider:
    def __init__(self):
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        self.api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        self.model = os.getenv("MISTRAL_MODEL", "").strip()

    def generate(self, question: str, context: str, system_instruction: str, allowed_articles: list[str]) -> ProviderResponse:
        result = ProviderResponse(provider="mistral", model=self.model or None)
        if not self.api_key or not self.model:
            result.error = "Set MISTRAL_API_KEY and MISTRAL_MODEL in the environment or .env file."
            return result
        allowed = list(dict.fromkeys(allowed_articles))
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": json.dumps({
                "question": question, "legal_context": context, "allowed_citations": allowed,
            }, ensure_ascii=False)},
        ]
        output_format = response_format(allowed)
        start = None
        try:
            with Mistral(api_key=self.api_key, retry_config=None) as client:
                start = perf_counter()
                response = client.chat.complete(
                    model=self.model, messages=messages, response_format=output_format, temperature=0,
                )
                result.total_latency_ms = (perf_counter() - start) * 1000
        except Exception as error:
            if start is not None and result.total_latency_ms is None:
                result.total_latency_ms = (perf_counter() - start) * 1000
            result.error = format_api_error(error)
            return result
        if response.usage is not None:
            result.prompt_tokens = response.usage.prompt_tokens
            result.completion_tokens = response.usage.completion_tokens
        if not response.choices or response.choices[0].finish_reason != "stop":
            result.error = "Mistral did not complete the response."
            return result
        try:
            answer = json.loads(response.choices[0].message.content or "")
            if isinstance(answer, dict):
                if isinstance(answer.get("answer"), str):
                    result.raw_answer = answer["answer"]
                if isinstance(answer.get("citations"), list) and all(isinstance(item, str) for item in answer["citations"]):
                    result.raw_citations = answer["citations"]
            validate_answer(answer)
        except (ValueError, TypeError):
            result.error = "Mistral returned malformed structured output; expected answer and citations."
            return result
        invalid = list(dict.fromkeys(citation for citation in answer["citations"] if citation not in allowed))
        if invalid:
            result.error = "Invalid provider response: citations outside retrieved context: " + ", ".join(invalid)
            return result
        result.answer = answer["answer"]
        result.citations = answer["citations"]
        return result
