# Wikipedia RAG Assistant (WikiRAG)

A fully local, ChatGPT-style assistant that answers questions about **famous people** and **famous places** using text from **40 curated Wikipedia articles** (20 people and 20 places). Everything runs on your machine: **Streamlit** for the chat UI, **Ollama** for embeddings and the LLM, **ChromaDB** for vector search, and **urllib** + **html.parser** for Wikipedia fetching—no paid APIs and no cloud dependency after setup. **Optional** behavior (response cache, heuristic re-ranking, comparison mode) is controlled in **`config.py`**.

## Features

- **Grounded answers** — The model is instructed to use only retrieved context and to say *I don't know* when the answer is not supported.
- **Response cache** — SQLite (`cache.db`) stores answers keyed by a normalized query hash so repeats and paraphrases (word order–invariant) skip retrieval and Ollama. Sidebar shows cache stats and **Clear cache**. Toggle with `CACHE_ENABLED` in `config.py`.
- **Hybrid re-ranking** — After Chroma returns candidates, a stdlib-only **reranker** blends keyword overlap, catalog entity overlap, and vector distance before context is sent to the LLM. Toggle with `RERANK_ENABLED` (when off, raw top‑k by distance is used).
- **Comparison questions** — Queries with comparison phrasing (*compare*, *versus*, *different from*, etc.) and **two or more** detected catalog entities use a dedicated **comparator** path: per-entity Chroma fetch (title-scoped when possible), optional per-entity rerank, structured context, and a comparison-focused prompt. UI shows **⚖️ comparison** and **Comparing: A \| B** when sources are on. Toggle with `COMPARISON_ENABLED`.
- **Multi-session chat** — Multiple conversations, sidebar history, **+ New Chat**, **Clear all chats**, and **per-chat delete** (✕ appears on row hover).
- **Conversation memory** — Follow-up questions in the same session reuse recent turns; pronouns in a new query can be rewritten for retrieval using the last assistant turn and the entity catalog.
- **Smart retrieval** — Rule-based **router** (person / place / both), metadata filtering on `entity_type`, **entity-boosted** fetch for catalog names in the query (helps typos and noisy phrasing), then merge with vector hits (see `config.py`: `ENTITY_BOOST_CHUNKS_PER_TITLE`, `RETRIEVAL_MAX_CHUNKS`, `TOP_K_CANDIDATES`, `TOP_K_RESULTS`, `RERANK_THRESHOLD`).
- **Light UI** — Gray sidebar, white main chat, optional **Show sources** with passage excerpts; sidebar shows Ollama status, document count, response cache summary, and a short **technical** caption.

## Prerequisites

- Python 3.11+ (3.12+ recommended)
- [Ollama](https://ollama.ai) installed and on your PATH
- Models pulled:

  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

## Installation

```bash
git clone https://github.com/vedatenisgul/wiki-rag.git
cd my-assistant
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Setup (run once)

Start Ollama:

```bash
ollama serve
```

Ingest Wikipedia pages and build the vector index (needs internet; on a fast machine this often finishes in roughly 10–20 minutes):

```bash
python ingest_pipeline.py
```

This writes under `raw_data/`, `chroma_db/`, and `ingestion.db`. The app may create **`cache.db`** in the project root when caching is enabled (safe to delete; it only stores cached answers).

## Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Verify the stack

With Ollama running and ingestion complete:

```bash
python test_pipeline.py
```

## Example questions

1. Who was Albert Einstein?
2. Where is the Eiffel Tower?
3. What did Marie Curie discover?
4. Compare Messi and Ronaldo.
5. How is the Eiffel Tower different from the Colosseum? (comparison path, **places**.)
6. After an answer about Albert Einstein: *When was he born?* (uses session context.)

**Out-of-catalog sanity check:** *Who is the president of Mars?* — expect a grounded *I don't know*–style response when the context does not support an answer (see `product_prd.md`).

## Request pipeline (order)

For each user message (when the index is ready):

1. **Cache lookup** — If `CACHE_ENABLED` and a hit exists, return the cached answer (no Chroma, no rerank, no Ollama).
2. **Comparison branch** — If `COMPARISON_ENABLED`, the query matches comparison triggers, and **≥2** catalog entities are detected → `comparator` builds context (per-entity Chroma; optional rerank per entity).
3. **Standard retrieval** — Else `retriever`: router → embed → Chroma (`TOP_K_CANDIDATES`) → **rerank + filter** if `RERANK_ENABLED` → entity boost merge → capped context.
4. **Generation** — `generator` (comparison prompt vs default prompt; chat history on non-comparison turns).
5. **Cache write** — If `CACHE_ENABLED`, save successful answers with the same **`hash_query`** logic (includes comparison answers).

See **`product_prd.md`** for the full diagram and component notes.

## Architecture (short)

| Piece | Role |
| --- | --- |
| `config.py` | Catalog, paths, limits; **`CACHE_ENABLED`**, **`RERANK_ENABLED`**, **`COMPARISON_ENABLED`** |
| `wiki_rag/cache.py` | SQLite response cache; normalize + MD5 key; stats; clear |
| `wiki_rag/router.py` | Query type; comparison triggers; **extract_comparison_entities** |
| `wiki_rag/reranker.py` | Stdlib hybrid scores on Chroma candidates (keyword / entity / distance) |
| `wiki_rag/comparator.py` | Per-entity retrieval + labeled comparison context |
| `wiki_rag/retriever.py` | Optional query rewrite; standard path vs comparison dispatch; format context |
| `wiki_rag/vectorstore.py` | Persistent collection; optional **`title`** filter for comparison queries |
| `wiki_rag/generator.py` | Ollama `llama3.2`; grounded prompt + optional **comparison** prompt |
| `app.py` | Streamlit UI, session state, orchestration (cache first) |
| `ingest_pipeline.py` | End-to-end ingest → chunk → embed → index (+ cache DB init) |

Details and data lists: **`product_prd.md`**.

## Known limitations

- Answers are limited to the **40 ingested** articles (people and places in `config.py`).
- **Ollama** must be running before using the app.
- **Cached answers** are keyed by **query text only** (normalized hash), not full chat history—identical wording in different threads shares an entry.
