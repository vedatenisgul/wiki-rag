"""
Fixed-size overlapping word chunks with per-chunk metadata for vector ingest.

Uses ``config.CHUNK_SIZE`` and ``config.CHUNK_OVERLAP`` (PRD §5.2).
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config  # noqa: E402

from wiki_rag import ingest  # noqa: E402


def chunk_text(text: str, title: str, entity_type: str) -> list[dict]:
    """
    Split ``text`` into overlapping word windows and return chunk dicts.

    Windows use ``config.CHUNK_SIZE`` words and advance by
    ``config.CHUNK_SIZE - config.CHUNK_OVERLAP`` words. Chunks with fewer than
    Chunks with fewer than ``config.MIN_CHUNK_WORDS`` words are omitted.
    """
    words = text.split()
    size = config.CHUNK_SIZE
    step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
    if not words or step <= 0:
        return []

    chunks: list[dict] = []
    chunk_index = 0
    start = 0
    slug = title.replace(" ", "_")

    while start < len(words):
        window = words[start : start + size]
        n = len(window)
        if n >= config.MIN_CHUNK_WORDS:
            chunks.append(
                {
                    "chunk_id": f"{slug}_{chunk_index}",
                    "title": title,
                    "entity_type": entity_type,
                    "chunk_index": chunk_index,
                    "text": " ".join(window),
                    "word_count": n,
                }
            )
            chunk_index += 1
        start += step

    return chunks


def chunk_document(title: str, entity_type: str) -> list[dict]:
    """Load raw text for ``title`` / ``entity_type`` and chunk it, or ``[]`` if missing."""
    raw = ingest.load_raw_text(title, entity_type)
    if raw is None:
        return []
    return chunk_text(raw, title, entity_type)


def chunk_all_documents() -> list[dict]:
    """Chunk every catalogued person and place; print progress per title."""
    out: list[dict] = []
    for title in config.FAMOUS_PEOPLE:
        doc_chunks = chunk_document(title, "person")
        print(f"Chunking: {title} → {len(doc_chunks)} chunks")
        out.extend(doc_chunks)
    for title in config.FAMOUS_PLACES:
        doc_chunks = chunk_document(title, "place")
        print(f"Chunking: {title} → {len(doc_chunks)} chunks")
        out.extend(doc_chunks)
    return out


if __name__ == "__main__":
    chunks = chunk_all_documents()
    print(f"Total chunks: {len(chunks)}")
    print(f"Sample: {chunks[0]}")
