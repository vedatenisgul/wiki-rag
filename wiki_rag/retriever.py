"""
Embed queries, route entity type, query Chroma, and format context for the LLM (PRD §6).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config  # noqa: E402

from wiki_rag import embedder  # noqa: E402
from wiki_rag import reranker  # noqa: E402
from wiki_rag import router  # noqa: E402
from wiki_rag import vectorstore  # noqa: E402

_PRONOUN_RE = re.compile(
    r"\b(he|she|it|they|his|her|its)\b",
    re.IGNORECASE,
)


def _entity_from_last_assistant(chat_history: list[dict]) -> str | None:
    """Find a catalog person/place name mentioned in the latest assistant turn."""
    for msg in reversed(chat_history):
        if msg.get("role") != "assistant":
            continue
        text = (msg.get("content") or "").lower()
        if not text:
            continue
        catalog = list(config.FAMOUS_PEOPLE) + list(config.FAMOUS_PLACES)
        for name in sorted(catalog, key=len, reverse=True):
            if name.lower() in text:
                return name
        return None
    return None


def rewrite_query_for_retrieval(query: str, chat_history: list[dict]) -> str:
    """
    If the query uses pronouns and history includes an assistant message naming
    a catalog entity, replace pronouns with that name for embedding/search.
    """
    if not chat_history or not _PRONOUN_RE.search(query):
        return query
    entity = _entity_from_last_assistant(chat_history)
    if not entity:
        return query

    def _sub(_m: re.Match[str]) -> str:
        return entity

    return _PRONOUN_RE.sub(_sub, query)


def _chunk_key(c: dict) -> tuple[str, int]:
    return (str(c["title"]), int(c["chunk_index"]))


def _merge_boosted_and_vector(
    boosted: list[dict],
    vector_chunks: list[dict],
    n_results: int,
) -> list[dict]:
    """
    Deduplicate by (title, chunk_index); keep all boosted rows first, then vector hits.

    When any boosted chunks exist, allow up to ``min(RETRIEVAL_MAX_CHUNKS,
    n_results + len(boosted))`` total so the named entity stays in context.
    """
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for c in boosted:
        k = _chunk_key(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    for c in vector_chunks:
        k = _chunk_key(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    if not boosted:
        return out[:n_results]
    cap = min(config.RETRIEVAL_MAX_CHUNKS, n_results + len(boosted))
    return out[:cap]


def retrieve(query: str, n_results: int = config.TOP_K_RESULTS) -> dict:
    """
    Classify the query, embed it, retrieve filtered chunks, and list source titles.
    """
    entity_filter = router.get_entity_filter(query)
    query_embedding = embedder.get_embedding(query)
    vector_chunks = vectorstore.query_collection(
        query_embedding,
        config.TOP_K_CANDIDATES,
        entity_filter,
    )
    if config.RERANK_ENABLED:
        vector_chunks = reranker.rerank(
            query, vector_chunks, top_k=n_results
        )
        vector_chunks = reranker.filter_low_quality(
            vector_chunks,
            threshold=config.RERANK_THRESHOLD,
        )
    else:
        vector_chunks = vector_chunks[:n_results]
    mentioned = router.extract_mentioned_entities(query)
    prioritize: list[str] = list(mentioned.get("people") or []) + list(
        mentioned.get("places") or []
    )
    boosted = vectorstore.get_chunks_for_titles(
        prioritize,
        entity_filter,
        limit_per_title=config.ENTITY_BOOST_CHUNKS_PER_TITLE,
    )
    chunks = _merge_boosted_and_vector(boosted, vector_chunks, n_results)
    query_type = router.classify_query(query)

    sources: list[str] = []
    seen: set[str] = set()
    for c in chunks:
        title = c["title"]
        if title not in seen:
            seen.add(title)
            sources.append(title)

    return {
        "query": query,
        "query_type": query_type,
        "entity_filter": entity_filter,
        "chunks": chunks,
        "sources": sources,
    }


def _word_count(s: str) -> int:
    return len(s.split()) if s else 0


def format_context(chunks: list[dict]) -> str:
    """
    Build a single string of cited chunks for the LLM, capped at
    ``config.MAX_CONTEXT_WORDS`` by dropping later chunks then trimming the body
    of the final chunk if needed.
    """
    if not chunks:
        return ""

    def blocks_for(ch_list: list[dict]) -> list[str]:
        return [
            "--- Source: {title} ({entity_type}) ---\n{text}\n".format(
                title=c["title"],
                entity_type=c["entity_type"],
                text=c["text"],
            )
            for c in ch_list
        ]

    def joined(ch_list: list[dict]) -> str:
        return "\n".join(blocks_for(ch_list))

    work = list(chunks)
    while work and _word_count(joined(work)) > config.MAX_CONTEXT_WORDS:
        if len(work) > 1:
            work.pop()
            continue
        c0 = work[0]
        header = "--- Source: {title} ({entity_type}) ---\n".format(
            title=c0["title"],
            entity_type=c0["entity_type"],
        )
        body_words = c0["text"].split()
        budget = config.MAX_CONTEXT_WORDS - _word_count(header)
        if budget <= 0:
            return header
        return f"{header}{' '.join(body_words[:budget])}\n"

    return joined(work)


def retrieve_and_format(
    query: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """Run :func:`retrieve` and attach a word-bounded :func:`format_context` string."""
    hist = chat_history if chat_history is not None else []
    rq = rewrite_query_for_retrieval(query, hist)

    if config.COMPARISON_ENABLED and router.is_comparison_query(query):
        comparison_info = router.extract_comparison_entities(query)
        if len(comparison_info["entities"]) >= 2:
            from wiki_rag import comparator

            comp_result = comparator.retrieve_for_comparison(
                rq,
                comparison_info["entities"],
                comparison_info["entity_types"],
            )
            context = comparator.build_comparison_context(comp_result)
            sources = comparator.get_comparison_sources(comp_result)
            return {
                "query": query,
                "query_type": comparison_info["comparison_type"],
                "is_comparison": True,
                "entities": list(comparison_info["entities"]),
                "chunks": [],
                "sources": sources,
                "context": context,
                "entity_filter": None,
            }

    out = retrieve(rq)
    out["query"] = query
    out["context"] = format_context(out["chunks"])
    out["is_comparison"] = False
    out["entities"] = []
    return out


if __name__ == "__main__":
    result = retrieve_and_format("What did Marie Curie discover?")
    print(f"Type: {result['query_type']}")
    print(f"Sources: {result['sources']}")
    print(f"Context preview: {result['context'][:300]}...")
