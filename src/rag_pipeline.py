from __future__ import annotations

from dataclasses import dataclass
import os

from src.openai_client import make_openai_client
from src.vector_store import SearchResult, load_index, search


DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[SearchResult]


def format_context(results: list[SearchResult]) -> str:
    context_parts = []
    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        context_parts.append(
            "\n".join(
                [
                    f"[Source {i}] {chunk.volume}, page {chunk.page}, fichier {chunk.source_file}",
                    chunk.text,
                ]
            )
        )
    return "\n\n---\n\n".join(context_parts)


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context = format_context(results)
    return f"""Tu es un assistant culinaire base sur un corpus de recettes.
Reponds en francais, de facon claire et utile.

Regles importantes:
- Utilise uniquement le contexte fourni.
- Si le contexte ne suffit pas, dis-le explicitement.
- Cite les sources utilisees avec volume et page.
- Ne fabrique pas d'ingredients, temps de cuisson ou etapes absents du contexte.

Contexte:
{context}

Question:
{question}
"""


def answer_question(
    question: str,
    index_dir: str = "storage/faiss_index",
    top_k: int = 5,
    chat_model: str = DEFAULT_CHAT_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> RagAnswer:
    index, chunks = load_index(index_dir)
    results = search(question, index, chunks, model=embedding_model, top_k=top_k)
    if not results:
        return RagAnswer(
            answer="Je n'ai trouve aucun passage pertinent dans le corpus.",
            sources=[],
        )

    client = make_openai_client()
    response = client.responses.create(
        model=chat_model,
        input=build_prompt(question, results),
    )
    return RagAnswer(answer=response.output_text, sources=results)
