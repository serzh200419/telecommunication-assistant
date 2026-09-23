import unittest
from unittest.mock import Mock, patch

from src.providers.base import ProviderResponse
from src.query_preprocessing import meta_answer
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
    def test_prefixed_retrieval_and_original_generation_question(self, retrieve):
        provider = Mock()
        provider.generate.return_value = ProviderResponse("test", "test", "Insufficient information", [])
        for question, prefix in (
            ("Who is operator?", "Law of the Republic of Armenia on Electronic Communications. "),
            ("Ո՞վ է օպերատորը։", "Հայաստանի Հանրապետության էլեկտրոնային հաղորդակցության մասին օրենք։ "),
            (" When may a provider suspend services for unpaid bills? ", "Law of the Republic of Armenia on Electronic Communications. "),
            ("What is the income tax rate?", "Law of the Republic of Armenia on Electronic Communications. "),
        ):
            result = ask(question, provider=provider)
            retrieve.assert_called_with(prefix + question, top_k=3)
            self.assertEqual(provider.generate.call_args.args[0], question)
            self.assertEqual(result["question"], question)
            self.assertEqual(result["answer"], "Insufficient information")


if __name__ == "__main__":
    unittest.main()
