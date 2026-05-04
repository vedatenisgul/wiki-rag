"""
Per-entity retrieval and context layout for comparison questions (no extra ML deps).
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config  # noqa: E402

from wiki_rag import embedder  # noqa: E402
from wiki_rag import reranker  # noqa: E402
from wiki_rag import vectorstore  # noqa: E402


def retrieve_for_comparison(
    query: str,
    entities: list[str],
    entity_types: list[str],
) -> dict:
    """
    Retrieve chunks separately for each entity (title-scoped vector search when possible).

    ``entity_types`` must align index-wise with ``entities``.
    """
    result: dict = {
        "query": query,
        "is_comparison": True,
        "entities": {},
    }

    for i, entity in enumerate(entities):
        etype = entity_types[i] if i < len(entity_types) else "person"
        entity_query = f"{entity} {query}"
        q_emb = embedder.get_embedding(entity_query)

        chunks = vectorstore.query_collection(
            q_emb,
            n_results=config.COMPARISON_VECTOR_POOL,
            entity_type_filter=etype,
            title_filter=entity,
        )
        if not chunks:
            chunks = vectorstore.query_collection(
                q_emb,
                n_results=config.TOP_K_CANDIDATES,
                entity_type_filter=etype,
                title_filter=None,
            )

        # Per-entity scoring only (does not use the standard retrieve() rerank path).
        if config.RERANK_ENABLED:
            chunks = reranker.rerank(
                entity_query,
                chunks,
                top_k=config.COMPARISON_CHUNKS_PER_ENTITY,
            )
            chunks = reranker.filter_low_quality(
                chunks,
                threshold=config.RERANK_THRESHOLD,
            )
        else:
            chunks = chunks[: config.COMPARISON_CHUNKS_PER_ENTITY]

        from wiki_rag.retriever import format_context

        ctx = format_context(chunks)
        result["entities"][entity] = {
            "chunks": chunks,
            "entity_type": etype,
            "context": ctx,
        }

    return result


def build_comparison_context(retrieval_result: dict) -> str:
    """Format labeled sections so the LLM can contrast subjects cleanly."""
    blocks: list[str] = []
    entities_payload = retrieval_result.get("entities") or {}
    # Preserve stable order: first keys as returned (dict preserves insertion in 3.7+)
    for name, payload in entities_payload.items():
        body = (payload or {}).get("context") or ""
        blocks.append(f"=== About {name} ===\n{body}\n")
    return "\n".join(blocks).strip()


def get_comparison_sources(retrieval_result: dict) -> list[str]:
    """Unique source titles in traversal order of ``entities``."""
    out: list[str] = []
    seen: set[str] = set()
    entities_payload = retrieval_result.get("entities") or {}
    for title in entities_payload.keys():
        if title not in seen:
            seen.add(title)
            out.append(title)
    return out
