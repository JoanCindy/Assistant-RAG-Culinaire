from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from src.chunking import TextChunk
from src.openai_client import make_openai_client


DEFAULT_INDEX_DIR = Path("storage/faiss_index")


@dataclass(frozen=True)
class SearchResult:
    chunk: TextChunk
    score: float


def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 64,
) -> np.ndarray:
    if not texts:
        raise ValueError("texts cannot be empty")

    client = make_openai_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)

    embeddings = np.array(vectors, dtype="float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def save_index(
    chunks: Iterable[TextChunk],
    embeddings: np.ndarray,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> None:
    chunk_list = list(chunks)
    if len(chunk_list) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_path / "index.faiss"))

    with (index_path / "chunks.json").open("w", encoding="utf-8") as f:
        json.dump([chunk.to_dict() for chunk in chunk_list], f, ensure_ascii=False, indent=2)


def load_index(index_dir: str | Path = DEFAULT_INDEX_DIR) -> tuple[faiss.Index, list[TextChunk]]:
    index_path = Path(index_dir)
    faiss_file = index_path / "index.faiss"
    chunks_file = index_path / "chunks.json"
    if not faiss_file.exists() or not chunks_file.exists():
        raise FileNotFoundError(
            f"Index not found in {index_path.resolve()}. Run `python build_index.py` first."
        )

    index = faiss.read_index(str(faiss_file))
    with chunks_file.open("r", encoding="utf-8") as f:
        chunks = [TextChunk(**item) for item in json.load(f)]
    return index, chunks


def search(
    query: str,
    index: faiss.Index,
    chunks: list[TextChunk],
    model: str = "text-embedding-3-small",
    top_k: int = 5,
) -> list[SearchResult]:
    query_vector = embed_texts([query], model=model)
    scores, indices = index.search(query_vector, top_k)
    results: list[SearchResult] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append(SearchResult(chunk=chunks[int(idx)], score=float(score)))
    return results
