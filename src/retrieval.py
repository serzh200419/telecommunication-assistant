import argparse
import hashlib
import json
from pathlib import Path

import faiss

from src.embeddings import CHUNKS_PATH, INDEX_PATH, METADATA_PATH, MODEL_NAME, encode_texts


def load_index(
    chunks_path: Path = CHUNKS_PATH,
    index_path: Path = INDEX_PATH,
    metadata_path: Path = METADATA_PATH,
):
    try:
        source = chunks_path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        index_bytes = index_path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError("Retrieval files are missing. Run python -m src.embeddings first.") from error
    if metadata["model_name"] != MODEL_NAME:
        raise ValueError("Index model differs from the query model. Rebuild the index.")
    if hashlib.sha256(source).hexdigest() != metadata["chunks_sha256"]:
        raise ValueError("Chunk source changed. Rebuild the index.")
    if hashlib.sha256(index_bytes).hexdigest() != metadata["index_sha256"]:
        raise ValueError("Index and metadata do not match. Rebuild the index.")
    index = faiss.read_index(str(index_path))
    chunks = json.loads(source)
    chunk_ids = metadata["chunk_ids"]
    if index.ntotal == 0 or index.ntotal != len(chunk_ids) or index.ntotal != len(chunks):
        raise ValueError("Index and chunk metadata lengths differ or are empty.")
    if index.d != metadata["dimension"] or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Index dimension or similarity metric differs from the expected configuration.")
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(by_id) != len(chunks) or len(set(chunk_ids)) != len(chunk_ids) or set(chunk_ids) != set(by_id):
        raise ValueError("Index chunk IDs do not match the chunk source.")
    return index, [by_id[chunk_id] for chunk_id in chunk_ids]


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a nonempty string.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    index, chunks = load_index()
    query = encode_texts([question.strip()])
    if query.shape[1] != index.d:
        raise ValueError("Query embedding dimension differs from the index.")
    scores, positions = index.search(query, min(top_k, index.ntotal))
    results = []
    for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
        chunk = chunks[int(position)]
        results.append({
            "rank": rank,
            "score": float(score),
            "chunk_id": chunk["chunk_id"],
            "article_number": chunk["article_number"],
            "title": chunk["title"],
            "text": chunk["text"],
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the law using local dense embeddings.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    for result in retrieve(args.question, args.top_k):
        print(f"{result['rank']}. Score: {result['score']:.4f} | Article {result['article_number']}")
        print(f"Title: {result['title']}")
        print(f"Chunk: {result['chunk_id']}")
        preview = " ".join(result["text"].split())
        print(preview[:240] + ("..." if len(preview) > 240 else ""))
        print()


if __name__ == "__main__":
    main()
