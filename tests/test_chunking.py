import unittest
from unittest.mock import patch

from src.chunking import approximate_tokens, create_chunks, split_article_text


class ChunkingTests(unittest.TestCase):
    def test_short_article_remains_one_chunk(self):
        article = {"article_number": "45", "title": "Title", "text": "First paragraph.\nSecond paragraph."}
        self.assertEqual(create_chunks([article]), [{
            "chunk_id": "article_45_chunk_1", "article_number": "45",
            "title": "Title", "text": article["text"], "chunk_index": 0,
        }])

    def test_numbered_parts_keep_subclauses_together(self):
        first = "1. Main part\n1) First clause\n2) Second clause\n"
        second = "2. Next part\n1) Another clause\n"
        self.assertEqual(split_article_text(first + second, 12), [first, second])

    def test_definition_paragraphs_are_boundaries(self):
        definitions = ["Անձ` սահմանում։\n", "Ցանց` սահմանում։\n", "Կապ` սահմանում։"]
        article = {"article_number": "2", "title": "Հասկացություններ", "text": "".join(definitions)}
        chunks = create_chunks([article], max_tokens=5)
        self.assertEqual([chunk["text"] for chunk in chunks], definitions)
        self.assertEqual([chunk["chunk_index"] for chunk in chunks], [0, 1, 2])
        self.assertTrue(all(chunk["title"] == article["title"] for chunk in chunks))

    def test_oversized_clause_splits_at_paragraphs(self):
        paragraphs = ["1. First part\n", "1) Subclause\n", "2) Subclause\n"]
        self.assertEqual(split_article_text("".join(paragraphs), 4), paragraphs)

    def test_fallback_preserves_exact_text_and_size(self):
        for source in ["Հայերեն բառեր " * 20, "ա" * 95, "First\r\n\r\nSecond\t paragraph.  "]:
            with self.subTest(source=source):
                chunks = split_article_text(source, 6)
                self.assertEqual("".join(chunks), source)
                self.assertTrue(all(0 < approximate_tokens(chunk) <= 6 for chunk in chunks))

    def test_articles_are_never_mixed(self):
        articles = [
            {"article_number": "1", "title": "First", "text": "First article. " * 10},
            {"article_number": "2", "title": "Second", "text": "Second article. " * 10},
        ]
        chunks = create_chunks(articles, max_tokens=8)
        for article in articles:
            matching = [chunk for chunk in chunks if chunk["article_number"] == article["article_number"]]
            self.assertEqual("".join(chunk["text"] for chunk in matching), article["text"])
            self.assertTrue(all(chunk["title"] == article["title"] for chunk in matching))

    def test_text_loss_is_rejected(self):
        with patch("src.chunking.split_article_text", return_value=["Missing text"]):
            with self.assertRaisesRegex(ValueError, "preservation"):
                create_chunks([{"article_number": "1", "title": "Title", "text": "Original text"}])

    def test_invalid_size_is_rejected(self):
        with self.assertRaises(ValueError):
            split_article_text("Text", 0)


if __name__ == "__main__":
    unittest.main()
