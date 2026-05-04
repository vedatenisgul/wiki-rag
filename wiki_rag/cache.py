"""
SQLite-backed response cache for grounded answers (stdlib only).

Skips Ollama on repeated or word-order–invariant queries when enabled in config.
Keys use :func:`hash_query` for all answer types (including comparison replies).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import config


_IDK_MARKERS = ("i don't know", "i do not know")


def _db_path() -> Path:
    return Path(config.CACHE_DB_PATH)


def init_cache() -> None:
    """Create cache table and optional ``accessed_at`` column if missing."""
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS response_cache (
                query_hash TEXT PRIMARY KEY,
                original_query TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT NOT NULL,
                query_type TEXT,
                created_at TEXT NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                accessed_at TEXT
            )
            """
        )
        cur.execute("PRAGMA table_info(response_cache)")
        cols = {row[1] for row in cur.fetchall()}
        if "accessed_at" not in cols:
            cur.execute(
                "ALTER TABLE response_cache ADD COLUMN accessed_at TEXT"
            )
        conn.commit()
    finally:
        conn.close()


def _normalize_for_hash(query: str) -> str:
    s = query.lower().strip()
    s = re.sub(r"[\?\.\!\,]+", "", s)
    words = sorted(w for w in s.split() if w)
    return " ".join(words)


def hash_query(query: str) -> str:
    normalized = _normalize_for_hash(query)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _is_idk_answer(answer: str) -> bool:
    t = (answer or "").strip().lower()
    if not t:
        return True
    return any(t == m or t.startswith(m) for m in _IDK_MARKERS)


def get_cached_response(query: str) -> dict[str, Any] | None:
    if not config.CACHE_ENABLED:
        return None
    init_cache()
    h = hash_query(query)
    conn = sqlite3.connect(str(_db_path()))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT answer, sources, query_type, hit_count FROM response_cache "
            "WHERE query_hash = ?",
            (h,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        answer, sources_json, query_type, _hits = row
        now = datetime.now().isoformat(timespec="seconds")
        cur.execute(
            "UPDATE response_cache SET hit_count = hit_count + 1, accessed_at = ? "
            "WHERE query_hash = ?",
            (now, h),
        )
        conn.commit()
        try:
            sources = json.loads(sources_json) if sources_json else []
        except json.JSONDecodeError:
            sources = []
        if not isinstance(sources, list):
            sources = []
        return {
            "answer": answer,
            "sources": sources,
            "query_type": query_type or "",
        }
    finally:
        conn.close()


def save_to_cache(
    query: str,
    answer: str,
    sources: list,
    query_type: str,
) -> None:
    if not config.CACHE_ENABLED:
        return
    if _is_idk_answer(answer):
        return
    init_cache()
    h = hash_query(query)
    now = datetime.now().isoformat(timespec="seconds")
    payload = json.dumps(list(sources or []), ensure_ascii=False)
    conn = sqlite3.connect(str(_db_path()))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO response_cache (
                query_hash, original_query, answer, sources, query_type,
                created_at, hit_count, accessed_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(query_hash) DO UPDATE SET
                original_query = excluded.original_query,
                answer = excluded.answer,
                sources = excluded.sources,
                query_type = excluded.query_type,
                accessed_at = excluded.accessed_at
            """,
            (h, query, answer, payload, query_type, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_cache_stats() -> dict[str, Any]:
    init_cache()
    conn = sqlite3.connect(str(_db_path()))
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM response_cache")
        total_cached = int(cur.fetchone()[0])
        cur.execute(
            "SELECT COALESCE(SUM(hit_count), 0) FROM response_cache"
        )
        total_hits = int(cur.fetchone()[0])
        cur.execute(
            "SELECT original_query FROM response_cache "
            "ORDER BY hit_count DESC, original_query ASC LIMIT 1"
        )
        row = cur.fetchone()
        most_hit_query = row[0] if row and row[0] else ""
        return {
            "total_cached": total_cached,
            "total_hits": total_hits,
            "most_hit_query": most_hit_query,
        }
    finally:
        conn.close()


def clear_cache() -> None:
    init_cache()
    conn = sqlite3.connect(str(_db_path()))
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM response_cache")
        conn.commit()
    finally:
        conn.close()
