import hashlib
import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np


MODEL_NAME = "BAAI/bge-m3"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHUNKS_PATH = DATA_DIR / "processed/law_chunks.json"
INDEX_PATH = DATA_DIR / "processed/law_chunks.faiss"
METADATA_PATH = DATA_DIR / "processed/law_index_metadata.json"


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, cache_folder=str(DATA_DIR / "raw/models"))


def embedding_text(chunk: dict) -> str:
    return f"Հոդված {chunk['article_number']}. {chunk['title']}\n\n{chunk['text']}"


def token_lengths(texts: list[str], model) -> list[int]:
    encoded = model.tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
    return [len(tokens) for tokens in encoded["input_ids"]]


def encode_texts(texts: list[str], model=None) -> np.ndarray:
    if model is None:
        model = get_model()
    lengths = token_lengths(texts, model)
    if any(length > model.max_seq_length for length in lengths):
        raise ValueError(f"Input exceeds the model limit of {model.max_seq_length} tokens; refusing to truncate.")
    vectors = model.encode(
        texts, batch_size=4, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("The model returned invalid dense embeddings.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("The model returned embeddings that are not normalized.")
    return vectors


def build_index(
    chunks_path: Path = CHUNKS_PATH,
    index_path: Path = INDEX_PATH,
    metadata_path: Path = METADATA_PATH,
) -> None:
    source = chunks_path.read_bytes()
    chunks = json.loads(source)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if not chunk_ids or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Chunks must be nonempty and have unique chunk IDs.")
    texts = [embedding_text(chunk) for chunk in chunks]
    model = get_model()
    lengths = token_lengths(texts, model)
    vectors = encode_texts(texts, model)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    metadata = {
        "model_name": MODEL_NAME,
        "dimension": index.d,
        "chunk_ids": chunk_ids,
        "chunks_sha256": hashlib.sha256(source).hexdigest(),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Model: {MODEL_NAME}")
    print(f"Indexed chunks: {len(chunks)}")
    print(f"Embedding dimension: {index.d}")
    print(f"FAISS index size: {index.ntotal} vectors ({index_path.stat().st_size} bytes)")
    print(f"Maximum source tokens: {max(lengths)} / {model.max_seq_length}; truncated chunks: 0")
    print(f"Index: {index_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Chunk source: {chunks_path}")


if __name__ == "__main__":
    build_index()
