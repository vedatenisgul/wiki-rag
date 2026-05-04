"""
Central configuration and entity catalog for the local Wikipedia RAG assistant.

Paths, models, chunking, and retrieval limits follow ``product_prd.md``;
Ollama endpoints match §5.3 (embeddings) and §5.6 (generate).
"""

from pathlib import Path

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3.2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Paths
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
SQLITE_DB_PATH = str(BASE_DIR / "ingestion.db")
CACHE_DB_PATH = str(BASE_DIR / "cache.db")
CACHE_ENABLED = True
RERANK_ENABLED = True
COMPARISON_ENABLED = True

# Vector store (PRD §5.4 single collection)
CHROMA_COLLECTION_NAME = "wiki_rag"

# Wikipedia (ingest HTTP; urllib only)
WIKIPEDIA_WIKI_BASE = "https://en.wikipedia.org/wiki/"

# Chunking
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MIN_CHUNK_WORDS = 50

# Retrieval
TOP_K_CANDIDATES = 10  # Chroma fetch size before re-rank
TOP_K_RESULTS = 5  # Chunks returned to the LLM after re-rank
RERANK_THRESHOLD = 0.15  # filter_low_quality minimum score (falls back if all low)
# Comparison queries: vector pool per entity before re-rank; chunks passed to LLM per subject
COMPARISON_VECTOR_POOL = 20
COMPARISON_CHUNKS_PER_ENTITY = 3
MAX_CONTEXT_WORDS = 2000
# When the router detects catalog names in the query, pull this many chunks per
# matched title (by metadata) *before* pure vector hits. Fixes typo / garbled
# queries whose embeddings don't rank the named article in the top-k.
ENTITY_BOOST_CHUNKS_PER_TITLE = 3
# Max chunks passed to the LLM after merge (boosted + vector deduped).
RETRIEVAL_MAX_CHUNKS = 12

# Ollama generation options (non-streaming)
OLLAMA_NUM_CTX = 4096
OLLAMA_TEMPERATURE = 0.1

# Data lists
FAMOUS_PEOPLE = [
  "Albert Einstein", "Marie Curie", "Leonardo da Vinci",
  "William Shakespeare", "Ada Lovelace", "Nikola Tesla",
  "Lionel Messi", "Cristiano Ronaldo", "Taylor Swift",
  "Frida Kahlo", "Isaac Newton", "Charles Darwin",
  "Cleopatra", "Napoleon Bonaparte", "Mahatma Gandhi",
  "Nelson Mandela", "Elon Musk", "Steve Jobs",
  "Aristotle", "Michelangelo"
]

FAMOUS_PLACES = [
  "Eiffel Tower", "Great Wall of China", "Taj Mahal",
  "Grand Canyon", "Machu Picchu", "Colosseum",
  "Hagia Sophia", "Statue of Liberty", "Pyramids of Giza",
  "Mount Everest", "Stonehenge", "Angkor Wat",
  "Niagara Falls", "Amazon Rainforest", "Sahara Desert",
  "Great Barrier Reef", "Acropolis of Athens", "Chichen Itza",
  "Victoria Falls", "Petra"
]

# All 40 Wikipedia titles for ingestion (person then place), per PRD §4.
INGEST_CATALOG = [(title, "person") for title in FAMOUS_PEOPLE] + [
    (title, "place") for title in FAMOUS_PLACES
]


def ollama_embeddings_url() -> str:
    """
    Full URL for ``POST /api/embeddings`` (nomic-embed-text).

    Returns:
        str: ``OLLAMA_BASE_URL`` with ``/api/embeddings`` appended.
    """
    return f"{OLLAMA_BASE_URL}/api/embeddings"


def ollama_generate_url() -> str:
    """
    Full URL for ``POST /api/generate`` (llama3.2).

    Returns:
        str: ``OLLAMA_BASE_URL`` with ``/api/generate`` appended.
    """
    return f"{OLLAMA_BASE_URL}/api/generate"


def ensure_data_directories() -> None:
    """
    Ensure raw output layout and Chroma persistence directory exist on disk.

    Creates ``raw_data/person``, ``raw_data/place``, and ``CHROMA_PERSIST_DIR``
    (PRD §5.1 raw layout; §5.4 local persistence).
    """
    (RAW_DATA_DIR / "person").mkdir(parents=True, exist_ok=True)
    (RAW_DATA_DIR / "place").mkdir(parents=True, exist_ok=True)
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
