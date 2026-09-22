import unittest

from src.ingestion import extract_articles


def law_page(body: str) -> str:
    return (
        '<nav>Navigation</nav><div id="act_body"><div class="act-block_main">'
        f'<div class="act-block__section">{body}</div></div></div><footer>Footer</footer>'
    )


class IngestionTests(unittest.TestCase):
    def test_nested_paragraphs_and_non_article_content(self):
        html = law_page(
            '<p>Law preamble</p><p><table><tr><td>Հոդված 1.</td>'
            '<td>Առաջին վերնագիր</td></tr></table>'
            '<p>Հայերեն <strong>տեքստ</strong>։<br>Երկրորդ տող։'
            '<p>Գ Լ ՈՒ Խ 2<br>Chapter title</p>'
            '<p><table><tr><td>Հոդված 2.</td><td></td></tr></table>'
            '<p>Վերջին հոդված։'
            '<table><tr><td>Հայաստանի Հանրապետության<br>Նախագահ</td>'
            '<td>Signature and date</td></tr></table>'
        )
        self.assertEqual(extract_articles(html), [
            {"article_number": "1", "title": "Առաջին վերնագիր", "text": "Հայերեն տեքստ։\nԵրկրորդ տող։"},
            {"article_number": "2", "title": "", "text": "Վերջին հոդված։"},
        ])

    def test_section_headings_and_titles_are_excluded(self):
        for section_heading, section_title in [
            ("Բ Ա Ժ Ի Ն 2", "ԼԻՑԵՆԶԻԱՆԵՐԸ"),
            ("Բ Ա Ժ Ի Ն 3", "ՌԱԴԻՈՀԱՃԱԽԱԿԱՆՈՒԹՅՈՒՆՆԵՐԻ ՏԻՐՈՒՅԹԻ ԿԱՌԱՎԱՐՈՒՄԸ"),
        ]:
            with self.subTest(section=section_heading):
                html = law_page(
                    '<table><tr><td>Հոդված 1.</td><td>Առաջին</td></tr></table>'
                    '<p>Հոդվածի տեքստ։</p><p>ՊԱՀՊԱՆՎՈՂ ՏԵՔՍՏ</p>'
                    f'<p>{section_heading}</p><p>{section_title}</p>'
                    '<table><tr><td>Հոդված 2.</td><td>Երկրորդ</td></tr></table>'
                    '<p>Հաջորդ հոդված։</p>'
                )
                self.assertEqual(extract_articles(html), [
                    {"article_number": "1", "title": "Առաջին", "text": "Հոդվածի տեքստ։\nՊԱՀՊԱՆՎՈՂ ՏԵՔՍՏ"},
                    {"article_number": "2", "title": "Երկրորդ", "text": "Հաջորդ հոդված։"},
                ])

    def test_missing_container(self):
        with self.assertRaisesRegex(ValueError, "container"):
            extract_articles("<html>Unavailable</html>")

    def test_missing_articles(self):
        with self.assertRaisesRegex(ValueError, "headings"):
            extract_articles(law_page("<p>No law text</p>"))

    def test_duplicate_article_numbers(self):
        heading = '<table><tr><td>Հոդված 1.</td><td>Title</td></tr></table><p>Body</p>'
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            extract_articles(law_page(heading + heading))


if __name__ == "__main__":
    unittest.main()
