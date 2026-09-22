import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


SOURCE_URL = "https://www.arlis.am/hy/acts/1869"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data/processed/law_articles.json"
ARTICLE_HEADING = re.compile(r"Հոդված\s+(\d+(?:\.\d+)*)\.")
CHAPTER_HEADING = re.compile(r"Գ\s*Լ\s*Ո\s*Ւ\s*Խ\s+\d+")
SECTION_HEADING = re.compile(r"Բ\s*Ա\s*Ժ\s*Ի\s*Ն\s+\d+")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def fetch_law() -> str:
    response = requests.get(SOURCE_URL, timeout=30)
    response.raise_for_status()
    return response.content.decode("utf-8-sig")


def extract_articles(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#act_body .act-block_main .act-block__section")
    if content is None:
        raise ValueError("The ARLIS law content container was not found.")

    for element in content.select("script, style, nav, header, footer"):
        element.decompose()

    titles = {}
    for table in content.find_all("table"):
        cells = table.find_all("td")
        label = normalize_whitespace(cells[0].get_text()) if cells else ""
        match = ARTICLE_HEADING.fullmatch(label)
        if match:
            number = match.group(1)
            if number in titles:
                raise ValueError(f"Duplicate article number: {number}")
            titles[number] = normalize_whitespace(cells[1].get_text()) if len(cells) > 1 else ""
            table.replace_with(f"\nՀոդված {number}.\n")
        elif "Հայաստանի Հանրապետության" in table.get_text() and "Նախագահ" in table.get_text():
            table.decompose()
        else:
            raise ValueError("Unexpected table in the law content; review the page structure.")

    # Unclosed paragraphs in the source can nest later articles inside earlier ones.
    for paragraph in content.find_all("p"):
        paragraph.insert_before("\n")
    for line_break in content.find_all("br"):
        line_break.replace_with("\n")

    articles = []
    current_article = None
    paragraphs = []
    for line in content.get_text().splitlines():
        line = normalize_whitespace(line)
        if not line:
            continue
        heading = ARTICLE_HEADING.fullmatch(line)
        if heading or CHAPTER_HEADING.match(line) or SECTION_HEADING.fullmatch(line):
            if current_article is not None:
                current_article["text"] = "\n".join(paragraphs)
            current_article = None
            paragraphs = []
            if heading:
                number = heading.group(1)
                current_article = {"article_number": number, "title": titles[number], "text": ""}
                articles.append(current_article)
        elif current_article is not None:
            paragraphs.append(line)

    if current_article is not None:
        current_article["text"] = "\n".join(paragraphs)
    if not articles or len(articles) != len(titles):
        raise ValueError("Could not extract all article headings from the law content.")
    if any(not article["text"] for article in articles):
        raise ValueError("An extracted article has no body text; review the page structure.")
    return articles


def main() -> None:
    articles = extract_articles(fetch_law())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Total articles: {len(articles)}")
    print(f"First article: {articles[0]['article_number']} - {articles[0]['title']}")
    print(f"Last article: {articles[-1]['article_number']} - {articles[-1]['title']}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
