"""
Embedding client for ``nomic-embed-text`` via Ollama HTTP API (stdlib only).

Uses :func:`urllib.request.urlopen` and :mod:`json` per PRD §5.3 — no sentence-transformers.
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config  # noqa: E402

_EMBED_POST_TIMEOUT_S = 120
_OLLAMA_PROBE_TIMEOUT_S = 5


def get_embedding(text: str) -> list[float]:
    """
    Request one embedding vector from Ollama ``/api/embeddings``.

    Raises:
        RuntimeError: If Ollama is unreachable, HTTP fails, or response is invalid.
    """
    url = config.ollama_embeddings_url()
    payload = json.dumps(
        {"model": config.OLLAMA_EMBED_MODEL, "prompt": text}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_EMBED_POST_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(
            f"Ollama embeddings failed (HTTP {e.code}): {e.reason}. "
            f"Response: {detail[:500]}. "
            "Ensure `ollama serve` is running and the model is available: "
            f"`ollama pull {config.OLLAMA_EMBED_MODEL}`"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL} ({e.reason}). "
            "Start the server with: ollama serve"
        ) from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        raise RuntimeError(
            f"Ollama returned invalid JSON from {url}: {err}"
        ) from err

    if "embedding" not in data:
        raise RuntimeError(
            f"Ollama response missing 'embedding' key: {data!r}"
        )
    return [float(x) for x in data["embedding"]]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed each string in order (one HTTP request per text; no batch API).

    Prints ``Embedding [i]/[total]...`` every 10 completed items.
    """
    total = len(texts)
    vectors: list[list[float]] = []
    for i, t in enumerate(texts):
        vectors.append(get_embedding(t))
        if total and (i + 1) % 10 == 0:
            print(f"Embedding {i + 1}/{total}...")
    return vectors


def check_ollama_running() -> bool:
    """
    Return True if a GET to the Ollama base URL returns HTTP 200.
    """
    try:
        req = urllib.request.Request(
            config.OLLAMA_BASE_URL,
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=_OLLAMA_PROBE_TIMEOUT_S) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


if __name__ == "__main__":
    if check_ollama_running():
        print("Ollama is running")
        vec = get_embedding("test sentence")
        print(f"Embedding dimension: {len(vec)}")
    else:
        print("Ollama is not running. Start with: ollama serve")
