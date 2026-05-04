"""
Diagnostic script: verify Ollama, embeddings, ingest, chunking, Chroma, router, and retrieval.

Run from project root: ``python test_pipeline.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from wiki_rag import chunker  # noqa: E402
from wiki_rag import embedder  # noqa: E402
from wiki_rag import ingest  # noqa: E402
from wiki_rag import retriever  # noqa: E402
from wiki_rag import router  # noqa: E402
from wiki_rag import vectorstore  # noqa: E402

_REQUIRED_CHUNK_KEYS = frozenset(
    {"chunk_id", "title", "entity_type", "chunk_index", "text", "word_count"}
)


def test_ollama_connection() -> tuple[bool, str]:
    ok = embedder.check_ollama_running()
    if ok:
        return True, "Ollama HTTP endpoint reachable"
    return False, "Ollama not reachable (start with: ollama serve)"


def test_embedding() -> tuple[bool, str]:
    text = "Who was Albert Einstein?"
    try:
        vec = embedder.get_embedding(text)
    except RuntimeError as err:
        return False, f"Embedding call failed: {err}"
    if not vec or len(vec) == 0:
        return False, "Embedding vector is empty"
    dim = len(vec)
    return True, f"Embedding OK (dimension={dim})"


def test_ingestion_single() -> tuple[bool, str]:
    title = "Albert Einstein"
    try:
        html = ingest.fetch_wikipedia_html(title)
        paragraphs = ingest.parse_article_paragraphs(html)
    except Exception as err:  # noqa: BLE001
        return False, f"Ingest fetch/parse failed: {err}"

    body = "\n\n".join(paragraphs)
    if len(body) <= 500:
        return False, f"Article text too short ({len(body)} chars, expected > 500)"
    preview = body[:150].replace("\n", " ")
    return True, f"Fetched article ({len(body)} chars). Preview: {preview!r}..."


def test_chunking() -> tuple[bool, str]:
    raw = ingest.load_raw_text("Albert Einstein", "person")
    if raw is None:
        try:
            html = ingest.fetch_wikipedia_html("Albert Einstein")
            paragraphs = ingest.parse_article_paragraphs(html)
            if not paragraphs:
                return False, "No paragraphs parsed; cannot chunk"
            raw = "\n\n".join(paragraphs)
        except Exception as err:  # noqa: BLE001
            return False, f"No local file and fetch failed: {err}"

    chunks = chunker.chunk_text(raw, "Albert Einstein", "person")
    if len(chunks) == 0:
        return False, "Chunker returned zero chunks"

    for i, ch in enumerate(chunks):
        keys = set(ch.keys())
        if not _REQUIRED_CHUNK_KEYS.issubset(keys):
            missing = _REQUIRED_CHUNK_KEYS - keys
            return False, f"Chunk {i} missing keys: {sorted(missing)}"

    return True, f"Chunking OK ({len(chunks)} chunks, keys validated)"


def test_vectorstore() -> tuple[bool, str]:
    try:
        vectorstore.get_client()
        stats = vectorstore.get_stats()
    except Exception as err:  # noqa: BLE001
        return False, f"Vector store error: {err}"

    total = stats.get("total", 0)
    if total == 0:
        return (
            True,
            "WARN: total chunks = 0 (run `python ingest_pipeline.py` first). "
            f"Client OK; stats={stats!r}",
        )
    return True, f"Client OK; total={total} people={stats['people']} places={stats['places']}"


def test_router() -> tuple[bool, str]:
    cases: list[tuple[str, str]] = [
        ("Who was Albert Einstein?", "person"),
        ("Where is the Eiffel Tower?", "place"),
        ("Compare Einstein and Taj Mahal", "both"),
        ("Who is the president of Mars?", "both"),
        ("What did Marie Curie discover?", "person"),
    ]
    failures: list[str] = []
    for query, expected in cases:
        got = router.classify_query(query)
        if got != expected:
            failures.append(f"{query!r} → got {got!r}, expected {expected!r}")
    if failures:
        return False, "; ".join(failures)
    return True, "All 5 routing queries matched expected labels"


def test_full_retrieval() -> tuple[bool, str]:
    try:
        stats = vectorstore.get_stats()
    except Exception as err:  # noqa: BLE001
        return False, f"Could not read vector store: {err}"

    if stats.get("total", 0) == 0:
        return True, "SKIP: vector DB empty — run ingest_pipeline.py before testing retrieval"

    try:
        result = retriever.retrieve_and_format("What did Marie Curie discover?")
    except Exception as err:  # noqa: BLE001
        return False, f"retrieve_and_format failed: {err}"

    sources = result.get("sources") or []
    if not sources:
        return False, "sources list is empty (expected at least one title)"
    return True, f"Retrieval OK ({len(sources)} source title(s): {sources[:3]}...)"


def test_retrieval_includes_named_entity() -> tuple[bool, str]:
    """Named catalog entity in the query must appear in at least one chunk title."""
    try:
        stats = vectorstore.get_stats()
    except Exception as err:  # noqa: BLE001
        return False, str(err)

    if stats.get("total", 0) == 0:
        return True, "SKIP: vector DB empty"

    q = "Who was Albert Einstein?"
    try:
        r = retriever.retrieve(q)
    except Exception as err:  # noqa: BLE001
        return False, f"retrieve failed: {err}"

    titles = {c["title"] for c in r.get("chunks") or []}
    if "Albert Einstein" not in titles:
        return (
            False,
            f"'Albert Einstein' missing from chunk titles; got {sorted(titles)!r}",
        )
    return True, f"Chunks include Albert Einstein ({len(r['chunks'])} chunks)"


def test_retrieval_typo_query_still_grounds_named_article() -> tuple[bool, str]:
    """
    Short/garbled queries can yield poor embedding similarity; boosted metadata
    fetch must still pull the named article (user-reported: 'ho was ...').
    """
    try:
        stats = vectorstore.get_stats()
    except Exception as err:  # noqa: BLE001
        return False, str(err)

    if stats.get("total", 0) == 0:
        return True, "SKIP: vector DB empty"

    q = "ho was Albert Einstein?"
    try:
        r = retriever.retrieve(q)
    except Exception as err:  # noqa: BLE001
        return False, f"retrieve failed: {err}"

    titles = [c["title"] for c in r.get("chunks") or []]
    if "Albert Einstein" not in titles:
        return (
            False,
            f"Expected 'Albert Einstein' in retrieval after entity boost; "
            f"titles={titles!r}",
        )
    if titles[0] != "Albert Einstein":
        return (
            False,
            f"First chunk should be Einstein after boost; first title={titles[0]!r}",
        )
    return True, f"Typo query grounded: first source is Albert Einstein ({len(titles)} chunks)"


def test_vectorstore_title_fetch() -> tuple[bool, str]:
    """Metadata get by title returns rows in chunk_index order."""
    try:
        stats = vectorstore.get_stats()
    except Exception as err:  # noqa: BLE001
        return False, str(err)

    if stats.get("total", 0) == 0:
        return True, "SKIP: vector DB empty"

    rows = vectorstore.get_chunks_for_titles(
        ["Albert Einstein"],
        "person",
        limit_per_title=2,
    )
    if len(rows) < 1:
        return False, "get_chunks_for_titles returned no rows for Albert Einstein"
    if rows[0]["title"] != "Albert Einstein":
        return False, f"unexpected title: {rows[0]!r}"
    if int(rows[0]["chunk_index"]) != 0:
        return False, f"expected chunk_index 0 first, got {rows[0]['chunk_index']}"
    return True, f"get_chunks_for_titles OK ({len(rows)} row(s))"


def test_generator_answer_not_idk_for_einstein() -> tuple[bool, str]:
    """With grounded Einstein chunks, the model should not default to I don't know."""
    try:
        stats = vectorstore.get_stats()
    except Exception as err:  # noqa: BLE001
        return False, str(err)

    if stats.get("total", 0) == 0:
        return True, "SKIP: vector DB empty"

    from wiki_rag import generator  # noqa: PLC0415

    q = "ho was Albert Einstein?"
    try:
        r = retriever.retrieve_and_format(q)
    except Exception as err:  # noqa: BLE001
        return False, str(err)

    if "Albert Einstein" not in r["context"]:
        return False, "Einstein not present in formatted context"

    ans = generator.generate_answer(q, r["context"])
    if ans.strip().lower() in ("i don't know", "i don’t know"):
        return (
            False,
            "LLM returned I don't know despite Einstein in context — check Ollama/model",
        )
    if "einstein" not in ans.lower():
        return (
            False,
            f"Answer likely ungrounded (no 'Einstein'); preview={ans[:120]!r}",
        )
    return True, f"Answer mentions Einstein; len={len(ans)}"


def main() -> None:
    tests: list[tuple[str, object]] = [
        ("test_ollama_connection", test_ollama_connection),
        ("test_embedding", test_embedding),
        ("test_ingestion_single", test_ingestion_single),
        ("test_chunking", test_chunking),
        ("test_vectorstore", test_vectorstore),
        ("test_router", test_router),
        ("test_full_retrieval", test_full_retrieval),
        ("test_retrieval_includes_named_entity", test_retrieval_includes_named_entity),
        (
            "test_retrieval_typo_query_still_grounds_named_article",
            test_retrieval_typo_query_still_grounds_named_article,
        ),
        ("test_vectorstore_title_fetch", test_vectorstore_title_fetch),
        ("test_generator_answer_not_idk_for_einstein", test_generator_answer_not_idk_for_einstein),
    ]

    passed_n = 0
    any_failed = False
    total = len(tests)

    for name, fn in tests:
        try:
            ok, msg = fn()
        except Exception as err:  # noqa: BLE001
            ok, msg = False, f"Unhandled exception: {err}"

        if ok:
            passed_n += 1
            print(f"{name}: PASS ✓ — {msg}")
        else:
            any_failed = True
            print(f"{name}: FAIL ✗ — {msg}")

    print()
    print(f"{passed_n}/{total} tests passed")

    if any_failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
