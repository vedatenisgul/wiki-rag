# Design choice: Single collection "wiki_rag" with entity_type metadata.
# Chosen over two separate collections because:
# 1. Simpler codebase - one client, one collection to manage
# 2. Supports cross-type queries (person AND place) without merging results
# 3. ChromaDB metadata filtering is efficient at our scale (~1200 chunks)
# 4. Easier to add new entity types later without code changes

"""
ChromaDB persistent client and ``wiki_rag`` collection (PRD §5.4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config  # noqa: E402

ADD_BATCH_SIZE = 50
_EXISTING_IDS_LOOKAHEAD = 500

_COLLECTION_COSINE_METADATA = {"hnsw:space": "cosine"}


def get_client() -> chromadb.PersistentClient:
    """
    Persistent Chroma client rooted at ``config.CHROMA_PERSIST_DIR``.

    Creates the persist directory if it does not exist.
    """
    Path(config.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def get_collection() -> chromadb.Collection:
    """
    Return the ``wiki_rag`` collection, creating it with cosine space if needed.
    """
    client = get_client()
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata=_COLLECTION_COSINE_METADATA,
    )


def _existing_ids_for(collection: chromadb.Collection, ids: list[str]) -> set[str]:
    """Return the subset of ``ids`` that are already stored."""
    have: set[str] = set()
    for i in range(0, len(ids), _EXISTING_IDS_LOOKAHEAD):
        batch = ids[i : i + _EXISTING_IDS_LOOKAHEAD]
        got = collection.get(ids=batch, include=[])
        have.update(got["ids"])
    return have


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """
    Insert chunks that are not already present (keyed by ``chunk_id``).

    New rows are added in batches of ``ADD_BATCH_SIZE``.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
        )
    if not chunks:
        return 0

    collection = get_collection()
    all_ids = [c["chunk_id"] for c in chunks]
    already = _existing_ids_for(collection, all_ids)

    pending: list[tuple[dict, list[float]]] = [
        (c, e) for c, e in zip(chunks, embeddings) if c["chunk_id"] not in already
    ]
    n_new = len(pending)
    print(f"Adding {n_new} new chunks to vector store...")

    for i in range(0, n_new, ADD_BATCH_SIZE):
        batch = pending[i : i + ADD_BATCH_SIZE]
        ids = [c["chunk_id"] for c, _ in batch]
        embs = [e for _, e in batch]
        documents = [c["text"] for c, _ in batch]
        metadatas = [
            {
                "title": c["title"],
                "entity_type": c["entity_type"],
                "chunk_index": c["chunk_index"],
            }
            for c, _ in batch
        ]
        collection.add(
            ids=ids,
            embeddings=embs,
            documents=documents,
            metadatas=metadatas,
        )

    return n_new


def query_collection(
    query_embedding: list[float],
    n_results: int | None = None,
    entity_type_filter: str | None = None,
    title_filter: str | None = None,
) -> list[dict]:
    """
    Similarity search with optional ``entity_type`` metadata filter.

    ``entity_type_filter`` must be ``\"person\"``, ``\"place\"``, or ``None``
    (search all types).

    Optional ``title_filter`` restricts hits to a single Wikipedia title (metadata).
    """
    if n_results is None:
        n_results = config.TOP_K_RESULTS
    if entity_type_filter is not None and entity_type_filter not in (
        "person",
        "place",
    ):
        raise ValueError(
            'entity_type_filter must be "person", "place", or None; '
            f"got {entity_type_filter!r}"
        )

    conds: list[dict] = []
    if title_filter:
        conds.append({"title": title_filter})
    if entity_type_filter == "person":
        conds.append({"entity_type": "person"})
    elif entity_type_filter == "place":
        conds.append({"entity_type": "place"})

    where = None
    if len(conds) == 1:
        where = conds[0]
    elif len(conds) > 1:
        where = {"$and": conds}

    collection = get_collection()
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids_row = raw.get("ids", [[]])[0]
    if not ids_row:
        return []

    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    out: list[dict] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        md = meta or {}
        out.append(
            {
                "text": doc if doc is not None else "",
                "title": md["title"],
                "entity_type": md["entity_type"],
                "chunk_index": int(md["chunk_index"]),
                "distance": float(dist),
            }
        )
    return out


def get_chunks_for_titles(
    titles: list[str],
    entity_type_filter: str | None = None,
    limit_per_title: int = 3,
) -> list[dict]:
    """
    Fetch chunks whose ``title`` metadata matches each catalog name (earliest
    ``chunk_index`` rows first). Used to ground retrieval when the query names
    an entity but vector similarity ranks other articles higher.
    """
    if not titles or limit_per_title <= 0:
        return []

    if entity_type_filter is not None and entity_type_filter not in (
        "person",
        "place",
    ):
        raise ValueError(
            'entity_type_filter must be "person", "place", or None; '
            f"got {entity_type_filter!r}"
        )

    collection = get_collection()
    out: list[dict] = []

    for title in titles:
        if not title or not str(title).strip():
            continue
        if entity_type_filter in ("person", "place"):
            where: dict = {
                "$and": [
                    {"title": title},
                    {"entity_type": entity_type_filter},
                ]
            }
        else:
            where = {"title": title}

        got = collection.get(
            where=where,
            include=["documents", "metadatas"],
        )
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        if not ids:
            continue

        rows: list[tuple[int, str, dict, str | None]] = []
        for cid, doc, meta in zip(ids, docs, metas):
            md = meta or {}
            try:
                idx = int(md["chunk_index"])
            except (KeyError, TypeError, ValueError):
                idx = 0
            rows.append((idx, cid, md, doc))

        rows.sort(key=lambda r: r[0])
        for idx, _cid, md, doc in rows[:limit_per_title]:
            out.append(
                {
                    "text": doc if doc is not None else "",
                    "title": md["title"],
                    "entity_type": md["entity_type"],
                    "chunk_index": idx,
                    "distance": 0.0,
                }
            )

    return out


def get_stats() -> dict:
    """Return ``total`` count and counts by ``entity_type``."""
    col = get_collection()
    total = col.count()
    try:
        people = col.count(where={"entity_type": "person"})
        places = col.count(where={"entity_type": "place"})
    except TypeError:
        people = len(
            col.get(where={"entity_type": "person"}, include=[])["ids"]
        )
        places = len(
            col.get(where={"entity_type": "place"}, include=[])["ids"]
        )
    return {"total": total, "people": people, "places": places}


def reset_collection() -> None:
    """Delete and recreate the ``wiki_rag`` collection (e.g. ``--reset``)."""
    client = get_client()
    try:
        client.delete_collection(config.CHROMA_COLLECTION_NAME)
    except Exception:
        pass
    get_collection()
