# Product Requirements Document
## Local Wikipedia RAG Assistant (WikiRAG)

---

## 1. Overview

A fully local, ChatGPT-style assistant that answers questions about famous people and places using Wikipedia text. The system runs on the developer machine (documented for MacBook M2 Pro) with **no external paid APIs**: LLM and embeddings use **Ollama** on localhost; the index uses **ChromaDB**; Wikipedia is fetched over HTTPS with the Python standard library.

---

## 2. System Architecture

High-level request flow (happy path when the index is populated):

```
User query (Streamlit)
    → Response cache lookup (SQLite cache.db; skip if disabled or miss)
    → [miss] Optional comparison path (if COMPARISON_ENABLED + triggers + ≥2 catalog entities):
          comparator: per-entity embed → Chroma (title + entity_type when possible)
          → optional per-entity rerank (RERANK_ENABLED) → labeled comparison context
    → [else] Standard retrieval:
          Router (person / place / both + entity filter)
          → optional pronoun/query rewrite using session history
          → Embed query (nomic-embed-text via Ollama)
          → ChromaDB: metadata-filtered vector search (TOP_K_CANDIDATES)
          → Hybrid rerank + low-quality filter (if RERANK_ENABLED)
          → Entity-boosted chunks by title → merge, cap context
    → Generator (llama3.2): comparison prompt OR default prompt (context + history on standard path)
    → [success, not IDK] save to cache (same hash_query for all answer types)
    → Chat UI (answer, optional sources, router / comparison metadata)
```

---

## 3. Technology Stack

| Component | Choice | Reason |
| --- | --- | --- |
| Language | Python 3.11+ | Stable; broadly available |
| LLM | llama3.2 via Ollama | Local, `/api/generate` |
| Embeddings | nomic-embed-text via Ollama | Same stack, `/api/embeddings` |
| Vector DB | ChromaDB | Local persistence, metadata filters |
| Metadata / resume | SQLite | `ingestion.db` ingestion log |
| UI | Streamlit | Browser chat, `session_state` for multi-chat |
| HTTP (Wikipedia) | urllib (stdlib) | No extra HTTP client required |
| Response cache | SQLite (`cache.db`) | Optional; MD5 key over normalized query text |
| Re-ranking (v1) | Pure Python heuristics | Keyword + catalog entity + vector distance; no extra ML deps |

Using Ollama for both LLM and embeddings keeps deployment simple and allows GPU acceleration where Ollama supports it (e.g. Metal on Apple Silicon).

---

## 4. Data

### 4.1 Famous people (20)

Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare, Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo, Taylor Swift, Frida Kahlo, Isaac Newton, Charles Darwin, Cleopatra, Napoleon Bonaparte, Mahatma Gandhi, Nelson Mandela, Elon Musk, Steve Jobs, Aristotle, Michelangelo

### 4.2 Famous places (20)

Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu, Colosseum, Hagia Sophia, Statue of Liberty, Pyramids of Giza, Mount Everest, Stonehenge, Angkor Wat, Niagara Falls, Amazon Rainforest, Sahara Desert, Great Barrier Reef, Acropolis of Athens, Chichen Itza, Victoria Falls, Petra

Canonical lists and ingest order live in **`config.py`** (`FAMOUS_PEOPLE`, `FAMOUS_PLACES`, `INGEST_CATALOG`).

**Feature toggles (same file):** `CACHE_ENABLED`, `RERANK_ENABLED`, `COMPARISON_ENABLED`; retrieval sizing: `TOP_K_CANDIDATES`, `TOP_K_RESULTS`, `RERANK_THRESHOLD`, `COMPARISON_VECTOR_POOL`, `COMPARISON_CHUNKS_PER_ENTITY`.

---

## 5. Component Design

### 5.1 Ingestion

- Fetch Wikipedia HTML with **urllib**; parse with **html.parser**.
- Store article body text under `raw_data/{entity_type}/` (normalized filenames).
- Record progress in SQLite for **resume after interruption**.

### 5.2 Chunking

- Overlapping word chunks: size **300**, overlap **50** (`CHUNK_SIZE`, `CHUNK_OVERLAP`); minimum words per chunk enforced in pipeline.
- Metadata per chunk: `title`, `entity_type`, `chunk_index`, `chunk_id`, etc.

### 5.3 Embedding

- Model: **nomic-embed-text** via Ollama `POST /api/embeddings`.
- One embedding per chunk; vectors stored in Chroma.

### 5.4 Vector store

- **Single** collection (e.g. `wiki_rag`) with metadata: at least `title`, `entity_type`, `chunk_index`.
- **Query-time** `entity_type` filter when the router supplies a filter.
- **Optional `title` metadata filter** on vector query for comparison retrieval (restrict hits to one Wikipedia article when titles match the catalog).
- **Title-scoped fetch** for boost path: `get_chunks_for_titles()` (or equivalent) to pull chunks for detected catalog names before deduplicating with pure vector hits.

### 5.5 Query routing

- **Rule-based**: catalog name detection, keyword hints (person vs place), default **both** when ambiguous.
- Outputs **query type** (person / place / both) for UI and logic; drives entity filter where applicable.
- **Comparison triggers** (substring phrases such as *compare*, *versus*, *difference between*, *different from*, etc.): `is_comparison_query()`.
- **Comparison entity extraction**: `extract_comparison_entities()` returns matched catalog names (full or last-name style matching per router rules), parallel **entity types**, and a **comparison_type** label (`person_person`, `place_place`, `person_place`, `single_type`). Comparison retrieval runs only when **at least two** entities are detected and `COMPARISON_ENABLED` is true.

### 5.6 Response cache

- **Storage**: SQLite `cache.db` under project root (`CACHE_DB_PATH`).
- **Key**: `hash_query(user_message)` — lowercase, strip, strip selected punctuation, **sort words** (order-invariant), MD5 hex.
- **Value**: answer text, JSON list of source titles, `query_type` (includes comparison types for UI).
- **Behaviors**: increment hit count on read; skip caching for empty / *I don't know*–style answers; optional **Clear cache** in UI.
- **Note**: Cache is **query-text scoped**; it does not incorporate full chat history into the key (same words → same cache entry across threads).

### 5.7 Hybrid re-ranking (post–vector retrieval)

- Runs **after** Chroma returns a candidate set and **before** context is formatted for the LLM on the **standard** retrieval path (when `RERANK_ENABLED`).
- **Signals** (stdlib only): weighted mix of (1) stopword-stripped keyword overlap between query and chunk text, (2) catalog **person/place** name match between query and chunk (title + text for entity signal), (3) similarity from Chroma cosine distance (`1 - distance`, clamped).
- **Outputs**: chunks sorted by `rerank_score`, top-`TOP_K_RESULTS`, then optional **low-quality filter** with fallback if all scores fall below `RERANK_THRESHOLD`.
- **Comparison path** does not use the standard `retrieve()` merge pipeline; per-entity sub-queries may still apply the same rerank helper when `RERANK_ENABLED`.

### 5.8 Comparison retrieval

- **When**: comparison-like query + **≥2** catalog entities + `COMPARISON_ENABLED`.
- **How**: For each entity, embed `"<entity> <query>"`, query Chroma with `entity_type` and preferably `title` filter; fallback to type-only search if needed; rerank slice to `COMPARISON_CHUNKS_PER_ENTITY`; format per-entity context blocks (`=== About <Entity> ===`).
- **Sources**: Ordered list of compared article titles for UI badges.

### 5.9 Retrieval (standard path — implemented behavior)

- **Entity boost**: For titles mentioned in the query, fetch a bounded number of chunks per title from metadata, then **merge** with top‑k vector results, dedupe by `(title, chunk_index)`, cap total chunks (`RETRIEVAL_MAX_CHUNKS`).
- **Vector pool**: fetch `TOP_K_CANDIDATES` from Chroma before rerank on standard path; return `TOP_K_RESULTS` after rerank/filter.
- **Query rewrite (session)**: If the user query contains pronouns and prior messages exist, optionally substitute a catalog entity taken from the **last assistant** message before embedding (simple heuristic; catalog-driven).
- **Original query** remains available for display; retrieval may use rewritten text internally per implementation.

### 5.10 Generation

- Model: **llama3.2** via Ollama `POST /api/generate`, **non-streaming** for v1.
- **Standard prompt**: **context** and a short **conversation history** (last few turns formatted as Human / Assistant) so follow-ups and pronouns can be resolved in the answer step.
- **Comparison prompt**: Structured instructions (per-subject sections, Key differences / In common, admit missing info); **no** history block in the template (context carries the evidence).
- Strict grounding: answer only from context; if unsupported, respond with **I don't know** (or equivalent per prompt).

### 5.11 Chat UI (Streamlit)

- **Multi-chat**: `session_state` holds multiple chats (`chats`, `active_chat_id`, `chat_counter`); first user message can auto-title a thread.
- **Main area**: Message thread and composer; **no** persistent page header for the active chat title (minimal chrome).
- **Sidebar** (light gray theme): workspace branding, **+ New Chat**, **Recent** list (open thread; **delete** control with hover-reveal), **System** (Ollama online/offline, indexed document count, **response cache** count + clear), **Settings** (e.g. show sources), **Clear all chats**, footer caption (native Python / metadata filtering).
- **Processing**: In-progress state (e.g. static “thinking” in-thread; optional sidebar pipeline indicator during a turn).
- **Assistant metadata**: Router target badge (person / place / mixed, or comparison subtype labels); **⚖️ comparison** caption and **Comparing: A | B** line when applicable; optional source cards and passage excerpts when enabled; **cached response** indicator when served from SQLite.

---

## 6. Data flow

1. Run **`ingest_pipeline.py`** once (or after catalog changes): fetch 40 pages → `raw_data/` → chunk → embed → Chroma (+ SQLite log); pipeline may initialize **`cache.db`**.
2. Typical chunk count is on the order of **hundreds to low thousands** (depends on article length and chunker rules).
3. User opens **`streamlit run app.py`**.
4. User sends a message in an **active** chat; prior turns in **that** chat are passed as history to the retriever/generator **on the standard path** (bounded window); comparison prompts rely primarily on structured context.
5. **Cache hit** → return stored assistant text and metadata (no embedding / retrieval / generation calls).
6. **Cache miss** → comparison branch **or** standard retrieval (embed + Chroma + optional rerank + merge) → generator → optional **cache write** (same `hash_query` for comparison and non-comparison answers).
7. UI appends assistant message (`query_type`, `sources`, `chunks`, comparison flags, cache indicator as implemented).

---

## 7. File structure (reference)

```
my-assistant/
├── wiki_rag/
│   ├── __init__.py
│   ├── ingest.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vectorstore.py
│   ├── router.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── comparator.py
│   ├── cache.py
│   └── generator.py
├── app.py
├── config.py
├── ingest_pipeline.py
├── test_pipeline.py
├── requirements.txt
├── product_prd.md
├── README.md
├── raw_data/             # ingested Wikipedia text (gitignored if desired)
├── chroma_db/            # Chroma persistence (local)
├── ingestion.db          # SQLite ingestion state
└── cache.db              # SQLite response cache (created when caching runs)
```

---

## 8. Success criteria

- All 40 Wikipedia pages ingest without unrecoverable errors; resume works via SQLite.
- Grounded answers for on-catalog questions; **I don't know** (or equivalent) for unsupported questions when context does not contain the fact.
- On-catalog **comparison** questions resolve to structured context for each detected entity when `COMPARISON_ENABLED` and two or more catalog names match.
- App runs **fully local** for chat/RAG after indexing (Ollama + local files only).
- **Cache** (when enabled) serves identical normalized queries without repeating retrieval or generation; stats and clear work from the sidebar.
- Response time acceptable on target hardware (e.g. under ~15 seconds for typical Q&A on M2-class machine, subject to model and load; cache hits are near-instant).

---

## 9. Failure cases (expected UX)

| Situation | Expected behavior |
| --- | --- |
| Out-of-scope fact / not in context | Model says it does not know (per prompt rules) |
| Ollama not running | Clear error / warning in UI; sidebar shows offline |
| Empty or missing index | Prompt to run `ingest_pipeline.py` / diagnostics |
| Empty retrieval | Warning + message that nothing was retrieved |
| Comparison phrasing but fewer than **two** detected catalog entities | Falls back to standard retrieval (no dedicated comparison context layout) |

---

## 10. Out of scope (current v1)

- Streaming token-by-token responses
- Learned **cross-encoder** or neural re-rankers (the shipped **heuristic** re-ranker is in scope; it uses no extra models)
- User authentication / multi-user accounts
- Hosting as a public multi-tenant service

**In scope (beyond original v1 notes):** multi-turn **session** memory within a chat, **lightweight query rewrite** for pronouns, **entity-boosted** retrieval merge, **multi-chat** sessions in the UI, **SQLite response caching**, **stdlib hybrid re-ranking**, and **catalog-grounded comparison** Q&A with dedicated retrieval and prompts.
