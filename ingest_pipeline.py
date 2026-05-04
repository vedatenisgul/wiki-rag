"""
One-shot setup: Wikipedia ingest → chunk → embed → Chroma (PRD §6).

Run: ``python ingest_pipeline.py``  |  ``python ingest_pipeline.py --dry-run``
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config  # noqa: E402
from wiki_rag import cache  # noqa: E402
from wiki_rag import chunker  # noqa: E402
from wiki_rag import embedder  # noqa: E402
from wiki_rag import ingest  # noqa: E402
from wiki_rag import vectorstore  # noqa: E402


def _clear_local_data() -> None:
    """Remove Chroma persistence, raw text files, and ingestion SQLite (full reset)."""
    chroma = Path(config.CHROMA_PERSIST_DIR)
    if chroma.exists():
        shutil.rmtree(chroma)

    raw = config.RAW_DATA_DIR
    if raw.exists():
        shutil.rmtree(raw)

    sqlite_path = Path(config.SQLITE_DB_PATH)
    if sqlite_path.exists():
        sqlite_path.unlink()

    config.ensure_data_directories()
    vectorstore.reset_collection()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wikipedia RAG offline setup (ingest, chunk, embed, index).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete chroma_db, raw_data, and ingestion DB, then re-ingest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show catalog and exit after checks (no fetch/embed/store).",
    )
    args = parser.parse_args()

    if args.reset:
        print("This will delete all data. Confirm? (y/n)")
        if input().strip().lower() != "y":
            print("Aborted.")
            return
        _clear_local_data()
        print("✓ Local data cleared.\n")

    print("================================")
    print("  Wikipedia RAG Assistant Setup ")
    print("================================")
    print()

    cache.init_cache()

    if not embedder.check_ollama_running():
        print(
            "Error: Ollama is not reachable at",
            config.OLLAMA_BASE_URL,
            "- start it with: ollama serve",
        )
        raise SystemExit(1)
    print("✓ Ollama is running")
    print()

    if args.dry_run:
        print("Dry run — would ingest:")
        for t in config.FAMOUS_PEOPLE:
            print(f"  [person] {t}")
        for t in config.FAMOUS_PLACES:
            print(f"  [place]  {t}")
        print()
        print(f"Total: {len(config.FAMOUS_PEOPLE) + len(config.FAMOUS_PLACES)} pages")
        print("(No files written; exiting.)")
        return

    t_start = time.time()

    print("Step 1/4: Fetching Wikipedia pages...")
    config.ensure_data_directories()
    stats = ingest.ingest_catalog(
        config.INGEST_CATALOG,
        config.RAW_DATA_DIR,
        Path(config.SQLITE_DB_PATH),
    )
    print(
        f"✓ Fetched {stats['fetched']} pages "
        f"({stats['skipped']} skipped, {stats['failed']} failed)"
    )
    print()

    print("Step 2/4: Chunking documents...")
    chunks = chunker.chunk_all_documents()
    doc_count = len({c["title"] for c in chunks})
    print(f"✓ Created {len(chunks)} chunks from {doc_count} documents")
    print()

    print("Step 3/4: Generating embeddings (this takes ~5-10 min)...")
    texts = [c["text"] for c in chunks]
    embeddings = embedder.get_embeddings_batch(texts)
    print(f"✓ Generated {len(embeddings)} embeddings")
    print()

    print("Step 4/4: Storing in ChromaDB...")
    stored_new = vectorstore.add_chunks(chunks, embeddings)
    print(f"✓ Stored {stored_new} chunks in vector database")
    print()

    elapsed = time.time() - t_start
    print("================================")
    print(f"  Done in {elapsed:.1f}s ({elapsed / 60.0:.1f} min)")
    print("================================")
    print("Setup complete! Run: streamlit run app.py")


if __name__ == "__main__":
    main()
