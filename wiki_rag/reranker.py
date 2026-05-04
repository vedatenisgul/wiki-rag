"""
Hybrid re-ranking on top of Chroma vector hits (keyword + entity + distance).
Stdlib only; no ML dependencies.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config  # noqa: E402

STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "was",
        "were",
        "are",
        "what",
        "who",
        "where",
        "when",
        "why",
        "how",
        "did",
        "do",
        "does",
        "in",
        "of",
        "to",
        "and",
        "or",
        "for",
        "with",
        "about",
        "that",
        "this",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def _tokenize_for_keywords(text: str) -> set[str]:
    out: set[str] = set()
    for m in _WORD_RE.finditer(text or ""):
        w = m.group(0).lower()
        if w and w not in STOPWORDS:
            out.add(w)
    return out


def _chunk_search_text(chunk: dict) -> str:
    title = chunk.get("title") or ""
    body = chunk.get("text") or ""
    return f"{title} {body}"


def _entity_score(query: str, chunk: dict) -> float:
    query_l = (query or "").lower()
    combined_l = _chunk_search_text(chunk).lower()
    catalog = list(config.FAMOUS_PEOPLE) + list(config.FAMOUS_PLACES)

    for name in catalog:
        nl = name.lower()
        if nl in query_l and nl in combined_l:
            return 1.0

    best = 0.0
    for name in catalog:
        parts = name.split()
        if len(parts) < 2:
            continue
        nl = name.lower()
        if nl in query_l and nl in combined_l:
            continue
        last = parts[-1].lower()
        if len(last) < 2:
            continue
        if last in query_l and last in combined_l:
            best = max(best, 0.6)

    return best


def _vector_score(chunk: dict) -> float:
    dist = float(chunk.get("distance", 1.0))
    sim = 1.0 - dist
    return max(0.0, min(1.0, sim))


def _signal_scores(query: str, chunk: dict) -> tuple[float, float, float]:
    qwords = _tokenize_for_keywords(query)
    cwords = _tokenize_for_keywords(chunk.get("text") or "")
    if qwords:
        keyword_score = len(qwords & cwords) / len(qwords)
    else:
        keyword_score = 0.0
    keyword_score = max(0.0, min(1.0, keyword_score))

    entity_score = _entity_score(query, chunk)
    entity_score = max(0.0, min(1.0, entity_score))

    vector_score = _vector_score(chunk)

    return keyword_score, entity_score, vector_score


def score_chunk(query: str, chunk: dict) -> float:
    k, e, v = _signal_scores(query, chunk)
    final = (0.4 * k) + (0.4 * e) + (0.2 * v)
    return max(0.0, min(1.0, final))


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return []

    scored: list[dict] = []
    for c in chunks:
        nc = dict(c)
        nc["rerank_score"] = score_chunk(query, nc)
        scored.append(nc)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    top = scored[:top_k]

    if top:
        k, e, v = _signal_scores(query, top[0])
        fr = top[0]["rerank_score"]
        print(
            "[rerank] top result breakdown: "
            f"rerank_score={fr:.4f} "
            f"(keyword={k:.4f}*0.4 entity={e:.4f}*0.4 vector={v:.4f}*0.2) "
            f"title={top[0].get('title', '')!r}"
        )

    return top


def filter_low_quality(
    chunks: list[dict],
    threshold: float = 0.15,
) -> list[dict]:
    if not chunks:
        return chunks

    kept = [c for c in chunks if float(c.get("rerank_score", 0.0)) >= threshold]
    if not kept:
        return list(chunks)
    return kept
