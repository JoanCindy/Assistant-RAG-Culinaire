from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from src.chunking import build_chunks
from src.ingestion import extract_pdf_pages
from src.vector_store import DEFAULT_INDEX_DIR, embed_texts, save_index


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build the FAISS index for the recipe RAG project.")
    parser.add_argument("--data-dir", default="Data", help="Directory containing PDF files.")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Output FAISS index directory.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in tokens.")
    parser.add_argument("--overlap", type=int, default=150, help="Chunk overlap in tokens.")
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="OpenAI embedding model.",
    )
    args = parser.parse_args()

    pages = extract_pdf_pages(args.data_dir)
    chunks = build_chunks(pages, chunk_size=args.chunk_size, overlap=args.overlap)
    if not chunks:
        raise RuntimeError("No chunks were created from the PDF corpus.")

    print(f"Extracted {len(pages)} pages and created {len(chunks)} chunks.")
    print(f"Creating embeddings with {args.embedding_model}...")
    embeddings = embed_texts([chunk.text for chunk in chunks], model=args.embedding_model)
    save_index(chunks, embeddings, args.index_dir)
    print(f"Index saved to {args.index_dir}")


if __name__ == "__main__":
    main()
