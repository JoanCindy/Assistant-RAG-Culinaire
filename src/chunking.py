from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from typing import Iterable

from src.ingestion import PageDocument

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    source_file: str
    volume: str
    page: int

    def to_dict(self) -> dict:
        return asdict(self)


_TOKENIZER = None
_TOKENIZER_FAILED = False


def _encoding():
    global _TOKENIZER, _TOKENIZER_FAILED
    if os.getenv("RAG_USE_TIKTOKEN", "0") != "1":
        return None
    if _TOKENIZER_FAILED:
        return None
    if _TOKENIZER is not None:
        return _TOKENIZER
    if tiktoken is None:
        return None
    try:
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        return _TOKENIZER
    except Exception:
        _TOKENIZER_FAILED = True
        return None


def count_tokens(text: str) -> int:
    enc = _encoding()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))


def split_text_by_tokens(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    enc = _encoding()
    if enc is None:
        approx_chars = chunk_size * 4
        overlap_chars = overlap * 4
        chunks = []
        step = max(1, approx_chars - overlap_chars)
        for start in range(0, len(text), step):
            end = min(len(text), start + approx_chars)
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
        return [chunk for chunk in chunks if chunk]

    tokens = enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk = enc.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks(
    pages: Iterable[PageDocument],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page_doc in pages:
        page_chunks = split_text_by_tokens(page_doc.text, chunk_size=chunk_size, overlap=overlap)
        for local_index, text in enumerate(page_chunks, start=1):
            chunk_id = f"{page_doc.source_file}:p{page_doc.page}:c{local_index}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_file=page_doc.source_file,
                    volume=page_doc.volume,
                    page=page_doc.page,
                )
            )
    return chunks
