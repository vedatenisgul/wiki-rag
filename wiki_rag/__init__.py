"""
Local Wikipedia RAG pipeline package.

Exposes ingestion, chunking, embedding, vector storage, routing, retrieval,
and generation building blocks for the Streamlit assistant described in
``product_prd.md``.
"""

from __future__ import annotations

__all__: list[str] = [
    "ingest",
    "chunker",
    "embedder",
    "vectorstore",
    "router",
    "retriever",
    "generator",
]
