import unittest
from unittest.mock import Mock, patch

from src.providers.base import ProviderResponse
from src.query_preprocessing import meta_answer, retrieval_query
from src.rag import ask


class QueryPreprocessingTests(unittest.TestCase):
    @patch("src.rag.retrieve")
    @patch("src.providers.gemini.GeminiProvider")
    def test_meta_bypasses_retrieval_and_provider(self, factory, retrieve):
        for question, expected in (("Who are you?", "internal legal assistant"),
                                   ("Ո՞վ ես դու։", "ներքին իրավական օգնական")):
            result = ask(question)
            self.assertIn(expected, result["answer"])
            self.assertEqual(result["provider"], "local")
            self.assertEqual(result["citations"], [])
            self.assertEqual(result["retrieved_results"], [])
            for field in ("prompt_tokens", "completion_tokens", "ttft_ms", "total_latency_ms"):
                self.assertIsNone(result[field])
        factory.assert_not_called()
        retrieve.assert_not_called()

    def test_phrase_normalization_and_scope(self):
        for question in ("  WHO ARE YOU?! ", "What are you?", "What can you do?",
                         "What can I ask you?", "Which law do you use?", "What law do you answer from?",
                         "Ի՞նչ կարող ես անել։", "Ի՞նչ հարցեր կարող եմ տալ։",
                         "Ո՞ր օրենքի հիման վրա ես պատասխանում։"):
            self.assertIsNotNone(meta_answer(question), question)
        for question in ("What is the income tax rate?", "Who is operator?",
                         "Who are you? What is the income tax rate?", "What can operators do?"):
            self.assertIsNone(meta_answer(question), question)

    @patch("src.rag.retrieve", return_value=[])
    def test_conditional_retrieval_and_original_generation_question(self, retrieve):
        provider = Mock()
        provider.generate.return_value = ProviderResponse("test", "test", "Insufficient information", [])
        for question, expected in (
            ("Who is operator?", "Who is operator in electronic communications?"),
            ("Ո՞վ է օպերատորը։", "Ո՞վ է օպերատորը էլեկտրոնային հաղորդակցության ոլորտում։"),
            (" When may a provider suspend services for unpaid bills? ", "When may a provider suspend services for unpaid bills?"),
            ("What corporate income tax rate applies?", "What corporate income tax rate applies?"),
        ):
            result = ask(question, provider=provider)
            retrieve.assert_called_with(expected, top_k=3)
            self.assertEqual(provider.generate.call_args.args[0], question)
            self.assertEqual(result["question"], question)
            self.assertEqual(result["answer"], "Insufficient information")

    def test_short_definition_queries(self):
        for question in SHORT_QUERIES:
            with self.subTest(question=question):
                expected = question.rstrip("?։") + (
                    " in electronic communications?" if question.isascii()
                    else " էլեկտրոնային հաղորդակցության ոլորտում։"
                )
                self.assertEqual(retrieval_query(question), expected)
        self.assertEqual(retrieval_query("  WHAT   DOES operator MEAN?! "),
                         "WHAT DOES operator MEAN in electronic communications?")
        self.assertEqual(retrieval_query("Define a subscriber."), "Define a subscriber in electronic communications?")

    def test_specific_questions_are_not_augmented(self):
        questions = [
            "When is a radio frequency use authorization required to operate an electronic communications network or provide a service?",
            "Ո՞ր դեպքերում օպերատորը կարող է սահմանափակել ծառայությունը։",
            "What corporate income tax rate applies?", "What is the income tax rate?",
            "When is a radio frequency use authorization required?",
            "What fines may the Regulator impose?", "What rights do end users have?",
            "Can an operator suspend service for non-payment?",
            "Who is operator in telecommunications?", "What is an operator in mathematics?",
            "Who are you?",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(retrieval_query(question), question)


SHORT_QUERIES = [
    "Who is operator?", "What is interconnection?", "Who is subscriber?",
    "What is a service provider?", "Ո՞վ է օպերատորը։",
    "Ի՞նչ է փոխկապակցումը։", "Ո՞վ է բաժանորդը։",
]


if __name__ == "__main__":
    unittest.main()
