from dataclasses import dataclass, field


@dataclass
class ProviderResponse:
    provider: str
    model: str | None
    answer: str | None = None
    citations: list[str] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_latency_ms: float | None = None
    error: str | None = None
    raw_answer: str | None = None
    raw_citations: list[str] = field(default_factory=list)
    ttft_ms: float | None = None


def validate_answer(answer: dict) -> None:
    if not isinstance(answer, dict) or set(answer) != {"answer", "citations"}:
        raise ValueError("Structured output must contain exactly answer and citations.")
    if not isinstance(answer["answer"], str) or not answer["answer"].strip():
        raise ValueError("Structured answer must be a nonempty string.")
    citations = answer["citations"]
    if not isinstance(citations, list) or any(not isinstance(item, str) or not item.strip() for item in citations):
        raise ValueError("Structured citations must be a list of article number strings.")
