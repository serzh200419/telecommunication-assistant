import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from src.embeddings import build_index, embedding_text, encode_texts
from src.retrieval import load_index, retrieve


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.paths = (root / "chunks.json", root / "index.faiss", root / "metadata.json")
        self.chunks = [
            {"chunk_id": "article_45_chunk_1", "article_number": "45", "title": "Կասեցում", "text": "Առաջին տեքստ։", "chunk_index": 0},
            {"chunk_id": "article_2_chunk_1", "article_number": "2", "title": "Հասկացություններ", "text": "Երկրորդ տեքստ։", "chunk_index": 0},
        ]
        self.paths[0].write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
        self.model = Mock()
        self.model.max_seq_length = 8192
        self.model.tokenizer.side_effect = lambda texts, **kwargs: {"input_ids": [[1, 2, 3] for text in texts]}
        self.model.encode.return_value = np.array([[1, 0], [0, 1]], dtype=np.float32)
        with patch("src.embeddings.get_model", return_value=self.model), redirect_stdout(io.StringIO()):
            build_index(*self.paths)

    def test_embedding_text_preserves_fields(self):
        self.assertEqual(embedding_text(self.chunks[0]), "Հոդված 45. Կասեցում\n\nԱռաջին տեքստ։")

    def test_vector_order_matches_metadata(self):
        index, chunks = load_index(*self.paths)
        self.assertEqual(chunks, self.chunks)
        np.testing.assert_array_equal(index.reconstruct(0), [1, 0])
        np.testing.assert_array_equal(index.reconstruct(1), [0, 1])
        texts = self.model.encode.call_args.args[0]
        self.assertEqual(texts, [embedding_text(chunk) for chunk in self.chunks])
        self.assertTrue(self.model.encode.call_args.kwargs["normalize_embeddings"])

    def test_ranked_results_preserve_metadata(self):
        with patch("src.retrieval.load_index", return_value=load_index(*self.paths)):
            with patch("src.retrieval.encode_texts", return_value=np.array([[0, 1]], dtype=np.float32)):
                results = retrieve("What is an operator?", top_k=5)
        self.assertEqual(len(results), 2)
        for rank, chunk in enumerate(reversed(self.chunks), start=1):
            self.assertEqual(results[rank - 1], {
                "rank": rank, "score": float(2 - rank),
                **{key: chunk[key] for key in ["chunk_id", "article_number", "title", "text"]},
            })

    def test_top_k_validation(self):
        for value in [0, -1, 1.5, True, "5"]:
            with self.subTest(value=value), patch("src.retrieval.load_index") as loader:
                with self.assertRaisesRegex(ValueError, "top_k"):
                    retrieve("Question", value)
                loader.assert_not_called()

    def test_empty_question_validation(self):
        for question in ["", " \n\t", None]:
            with self.subTest(question=question), patch("src.retrieval.load_index") as loader:
                with self.assertRaisesRegex(ValueError, "Question"):
                    retrieve(question)
                loader.assert_not_called()

    def test_metadata_length_mismatch_is_rejected(self):
        metadata = json.loads(self.paths[2].read_text(encoding="utf-8"))
        metadata["chunk_ids"].pop()
        self.paths[2].write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lengths"):
            load_index(*self.paths)

    def test_changed_chunk_source_is_rejected(self):
        self.chunks[0]["text"] = "Changed text"
        self.paths[0].write_text(json.dumps(self.chunks), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source changed"):
            load_index(*self.paths)

    def test_oversized_input_is_rejected_before_encoding(self):
        self.model.max_seq_length = 2
        self.model.encode.reset_mock()
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            encode_texts(["Oversized input"], self.model)
        self.model.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
