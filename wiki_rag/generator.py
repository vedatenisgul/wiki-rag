"""
Context-grounded answer generation via Ollama ``/api/generate`` (urllib + json).

PRD §5.6: llama3.2, non-streaming, strict context-only answers.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config  # noqa: E402

_GENERATE_TIMEOUT_S = 60
_OLLAMA_PROBE_TIMEOUT_S = 5

_OLLAMA_UNAVAILABLE = (
    "Error: Ollama is not running.\n"
    "     Please start it with: ollama serve"
)

_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about 
famous people and places using only the provided context.

Rules:
- Answer based ONLY on the context below
- Use the conversation history to understand follow-up questions
- If someone says "he", "she", "it", "they" refer to the 
  conversation history to resolve who/what they mean
- If the answer is not in the context, say: I don't know
- Be concise and factual

Context:
{context}

Conversation History:
{history_string}

Current Question: {query}

Answer:"""


def build_comparison_prompt(
    query: str,
    context: str,
    entities: list[str],
) -> str:
    """Prompt template for side-by-side comparison answers (context-only)."""
    ent = ", ".join(entities)
    return f"""You are a helpful assistant comparing famous people or places.
Use ONLY the provided context to answer.

You are comparing: {ent}

Context:
{context}

Instructions:
- Structure your answer with clear sections for each subject
- Use headers like "Einstein:" and "Tesla:" for clarity
- End with a brief "Key Differences:" or "In Common:" section
- If information about one subject is missing, say so
- Do not invent facts

Question: {query}

Comparison:"""


def build_prompt(
    query: str,
    context: str,
    chat_history: list[dict] | None = None,
) -> str:
    """Assemble the grounded system + user prompt for the LLM."""
    hist = chat_history if chat_history is not None else []
    tail = hist[-4:] if hist else []
    lines: list[str] = []
    for m in tail:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"Human: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    history_string = "\n".join(lines) if lines else "(none)"
    return _PROMPT_TEMPLATE.format(
        context=context,
        history_string=history_string,
        query=query,
    )


def call_ollama(prompt: str) -> str:
    """
    Non-streaming generate call; returns the model ``response`` text.

    On connection failures (incl. :class:`urllib.error.URLError` and timeouts),
    returns a fixed user-facing error string.
    """
    url = config.ollama_generate_url()
    payload = {
        "model": config.OLLAMA_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.OLLAMA_TEMPERATURE,
            "num_ctx": config.OLLAMA_NUM_CTX,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_GENERATE_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return (
            f"Error: Ollama generate failed (HTTP {e.code}): {e.reason}. "
            f"{detail[:300]}"
        )
    except (urllib.error.URLError, TimeoutError, OSError):
        return _OLLAMA_UNAVAILABLE

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return _OLLAMA_UNAVAILABLE

    if "response" not in decoded:
        return _OLLAMA_UNAVAILABLE

    return str(decoded["response"])


def generate_answer(
    query: str,
    context: str,
    chat_history: list[dict] | None = None,
    is_comparison: bool = False,
    entities: list[str] | None = None,
) -> str:
    """Return a grounded answer, or ``I don't know`` when there is no context."""
    if context is None or not str(context).strip():
        return "I don't know"

    ent = list(entities or [])
    if is_comparison:
        prompt = build_comparison_prompt(query, context, ent)
    else:
        prompt = build_prompt(query, context, chat_history=chat_history)
    return call_ollama(prompt)


def check_ollama_running() -> bool:
    """True if ``GET`` to ``config.OLLAMA_BASE_URL`` returns HTTP 200."""
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
    test_context = (
        "Marie Curie was a physicist and chemist who "
        "conducted pioneering research on radioactivity."
    )
    answer = generate_answer(
        "What was Marie Curie known for?",
        test_context,
    )
    print(f"Answer: {answer}")
