import json
import os
import re
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from google import genai

from src.providers.base import ProviderResponse, validate_answer


def safe_retry_after(value) -> str | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    value = str(value).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?s?", value):
        return value
    try:
        return format_datetime(parsedate_to_datetime(value))
    except (ValueError, TypeError, OverflowError):
        return None


def format_api_error(error: Exception) -> str:
    error_type = type(error).__name__
    response = getattr(error, "response", None)
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)
    if status is None:
        status = getattr(response, "status_code", None)
    fields = [error_type]
    if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
        fields.append(f"status={status}")
    if error_type == "RateLimitError" or status == 429:
        body = getattr(error, "body", None)
        if body is None:
            body = getattr(error, "response_json", None)
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except ValueError:
                body = None
        payload = body.get("error", body) if isinstance(body, dict) else {}
        if not isinstance(payload, dict):
            payload = {}
        reasons = [payload.get("status"), payload.get("reason"), getattr(error, "status", None)]
        retry_after = safe_retry_after(getattr(error, "retry_after", None))
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = retry_after or safe_retry_after(headers.get("retry-after"))
        details = payload.get("details", [])
        for detail in details if isinstance(details, list) else []:
            if isinstance(detail, dict):
                reasons.append(detail.get("reason"))
                if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                    retry_after = retry_after or safe_retry_after(detail.get("retryDelay"))
        # Only known codes are safe to display; free-form messages can echo credentials.
        quota_codes = {
            "RESOURCE_EXHAUSTED", "RATE_LIMIT_EXCEEDED", "QUOTA_EXCEEDED",
            "QUOTA_EXHAUSTED", "DAILY_LIMIT_EXCEEDED", "INSUFFICIENT_QUOTA",
        }
        reasons = list(dict.fromkeys(reason for reason in reasons if isinstance(reason, str) and reason in quota_codes))
        if reasons:
            fields.append("reason=" + ",".join(reasons))
        if retry_after is not None:
            fields.append(f"retry_after={retry_after}")
    return "Gemini request failed (" + "; ".join(fields) + ")."


def answer_schema(allowed_articles: list[str]) -> dict:
    allowed = list(dict.fromkeys(allowed_articles))
    citations = {"type": "array", "items": {"type": "string"}}
    if allowed:
        citations["items"]["enum"] = allowed
    else:
        citations["maxItems"] = 0
    return {
        "type": "object",
        "properties": {"answer": {"type": "string"}, "citations": citations},
        "required": ["answer", "citations"],
        "additionalProperties": False,
    }


class GeminiProvider:
    def __init__(self):
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", "").strip()

    def generate(self, question: str, context: str, system_instruction: str, allowed_articles: list[str]) -> ProviderResponse:
        result = ProviderResponse(provider="gemini", model=self.model or None)
        if not self.api_key or not self.model:
            result.error = "Set GEMINI_API_KEY and GEMINI_MODEL in the environment or .env file."
            return result

        schema = answer_schema(allowed_articles)
        request_input = json.dumps({
            "question": question, "legal_context": context,
            "allowed_citations": list(dict.fromkeys(allowed_articles)),
        }, ensure_ascii=False)
        start = None
        parts = []
        completed = False
        try:
            # Interactions counts attempts as retries in this SDK; explicitly exclude 429.
            with genai.Client(api_key=self.api_key, http_options={
                "retry_options": {"attempts": 1, "http_status_codes": [500, 502, 503, 504]},
            }) as client:
                start = perf_counter()
                stream = client.interactions.create(
                    model=self.model,
                    system_instruction=system_instruction,
                    input=request_input,
                    response_format={"type": "text", "mime_type": "application/json", "schema": schema},
                    generation_config={"thinking_level": "low"},
                    store=False,
                    stream=True,
                )
                for event in stream:
                    if event.event_type == "step.delta" and event.delta.type == "text":
                        content = event.delta.text
                        if isinstance(content, str) and content:
                            if result.ttft_ms is None:
                                result.ttft_ms = (perf_counter() - start) * 1000
                            parts.append(content)
                    elif event.event_type == "interaction.completed":
                        completed = event.interaction.status == "completed"
                        usage = event.interaction.usage
                        if usage is not None:
                            result.prompt_tokens = usage.total_input_tokens
                            result.completion_tokens = usage.total_output_tokens
                    elif event.event_type == "error":
                        code = (getattr(event.error, "code", None) or "").lower()
                        if code in ("resource_exhausted", "rate_limit_exceeded", "quota_exceeded", "429"):
                            result.error = "Gemini request failed (stream rate limit)."
                        else:
                            result.error = "Gemini request failed (stream error)."
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
            result.error = "Gemini did not complete the response."
            return result
        try:
            answer = json.loads("".join(parts))
            validate_answer(answer)
        except (ValueError, TypeError):
            result.error = "Gemini returned malformed structured output; expected answer and citations."
            return result
        result.answer = answer["answer"]
        result.citations = answer["citations"]
        return result
