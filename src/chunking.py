import argparse
import json
import re
from collections import Counter
from pathlib import Path


PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data/processed"
DEFAULT_MAX_TOKENS = 700
CHARACTERS_PER_TOKEN = 4


def approximate_tokens(text: str) -> int:
    """Estimate ceil(Unicode characters / 4), not a provider's actual token count.

    This sizing heuristic includes whitespace and can underestimate Armenian tokens.
    """
    return (len(text) + CHARACTERS_PER_TOKEN - 1) // CHARACTERS_PER_TOKEN


def structural_units(text: str) -> list[str]:
    paragraphs = text.splitlines(keepends=True)
    numbered_parts = re.compile(r"^\s*\d+\.\s")
    numbered_clauses = re.compile(r"^\s*\d+\)\s")
    if any(numbered_parts.match(paragraph) for paragraph in paragraphs):
        boundary = numbered_parts
    elif any(numbered_clauses.match(paragraph) for paragraph in paragraphs):
        boundary = numbered_clauses
    else:
        return paragraphs

    units = []
    current = ""
    for paragraph in paragraphs:
        if boundary.match(paragraph) and current:
            units.append(current)
            current = ""
        current += paragraph
    if current:
        units.append(current)
    return units


def split_oversized_paragraph(text: str, max_characters: int) -> list[str]:
    pieces = []
    while len(text) > max_characters:
        whitespace = list(re.finditer(r"\s+", text[:max_characters]))
        end = whitespace[-1].end() if whitespace else max_characters
        pieces.append(text[:end])
        text = text[end:]
    if text:
        pieces.append(text)
    return pieces


def split_article_text(text: str, max_tokens: int) -> list[str]:
    if max_tokens < 1:
        raise ValueError("Maximum chunk size must be positive.")
    if not text.strip():
        raise ValueError("Article text must not be empty.")
    max_characters = max_tokens * CHARACTERS_PER_TOKEN
    pieces = []
    for unit in structural_units(text):
        if len(unit) <= max_characters:
            pieces.append(unit)
        else:
            for paragraph in unit.splitlines(keepends=True):
                pieces.extend(split_oversized_paragraph(paragraph, max_characters))

    chunks = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > max_characters:
            chunks.append(current)
            current = ""
        current += piece
    if current:
        chunks.append(current)
    return chunks


def create_chunks(articles: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS) -> list[dict]:
    if max_tokens < 1:
        raise ValueError("Maximum chunk size must be positive.")
    if not articles:
        raise ValueError("No articles were provided.")
    chunks = []
    seen_numbers = set()
    for article in articles:
        number = article["article_number"]
        if number in seen_numbers:
            raise ValueError(f"Duplicate article number: {number}")
        seen_numbers.add(number)
        texts = split_article_text(article["text"], max_tokens)
        # Keeping the original separators makes exact reconstruction possible.
        if "".join(texts) != article["text"]:
            raise ValueError(f"Text preservation failed for article {number}.")
        for index, text in enumerate(texts):
            chunks.append({
                "chunk_id": f"article_{number}_chunk_{index + 1}",
                "article_number": number,
                "title": article["title"],
                "text": text,
                "chunk_index": index,
            })
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Create article-based law retrieval chunks.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    articles = json.loads((PROCESSED_DIR / "law_articles.json").read_text(encoding="utf-8"))
    chunks = create_chunks(articles, args.max_tokens)
    output_path = PROCESSED_DIR / "law_chunks.json"
    output_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sizes = [approximate_tokens(chunk["text"]) for chunk in chunks]
    counts = Counter(chunk["article_number"] for chunk in chunks)
    multiple_chunks = ", ".join(f"{number} ({count})" for number, count in counts.items() if count > 1)
    print(f"Total articles: {len(articles)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Approximate tokens: min={min(sizes)}, average={sum(sizes) / len(sizes):.1f}, max={max(sizes)}")
    print(f"Articles with multiple chunks (chunk count): {multiple_chunks or 'none'}")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
