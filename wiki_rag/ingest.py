"""
Wikipedia HTML fetch, article body extraction, raw file storage, and
SQLite-backed ingestion logging with resume support.
"""

from __future__ import annotations

import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import config

USER_AGENT = "WikiRAGAssistant/1.0 (local educational project; Python urllib)"


def load_raw_text(title: str, entity_type: str) -> str | None:
    """
    Read UTF-8 text from ``raw_data/{entity_type}/{Title_Underscores}.txt``.

    Returns:
        File contents, or ``None`` if the file does not exist.
    """
    path = config.RAW_DATA_DIR / entity_type / f"{title.replace(' ', '_')}.txt"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def fetch_wikipedia_html(title: str) -> str:
    """
    Download the Wikipedia article HTML for a page title using only urllib.

    Raises:
        urllib.error.URLError: Network or HTTP failure.
        UnicodeDecodeError: If the page is not UTF-8 decodable.
    """
    slug = title.strip().replace(" ", "_")
    path_component = urllib.parse.quote(slug, safe="/()%_")
    url = f"{config.WIKIPEDIA_WIKI_BASE}{path_component}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


class _MwParagraphParser(HTMLParser):
    """Collect ``<p>`` text inside ``#mw-content-text`` (stdlib html.parser)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._in_mw = False
        self._mw_div_depth = 0
        self._pbuf: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        ad = {k: v or "" for k, v in attrs}
        if tag == "div" and ad.get("id") == "mw-content-text":
            self._in_mw = True
            self._mw_div_depth = 1
            return
        if not self._in_mw:
            return
        if tag == "div":
            self._mw_div_depth += 1
        elif tag == "p" and self._pbuf is None:
            self._pbuf = []

    def handle_endtag(self, tag: str) -> None:
        if not self._in_mw:
            return
        if tag == "p" and self._pbuf is not None:
            text = "".join(self._pbuf).strip()
            self._pbuf = None
            if text and len(text) > 1:
                self.paragraphs.append(text)
        elif tag == "div":
            self._mw_div_depth -= 1
            if self._mw_div_depth <= 0:
                self._in_mw = False

    def handle_data(self, data: str) -> None:
        if self._pbuf is not None:
            self._pbuf.append(data)


def parse_article_paragraphs(html: str) -> list[str]:
    """
    Parse Wikipedia HTML and extract article body paragraphs (PRD §5.1).
    """
    parser = _MwParagraphParser()
    parser.feed(html)
    parser.close()
    return parser.paragraphs


def save_raw_text(output_path: Path, paragraphs: list[str]) -> None:
    """
    Write extracted paragraphs to a single UTF-8 text file under ``raw_data``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(paragraphs)
    output_path.write_text(body, encoding="utf-8")


def init_ingestion_db(db_path: Path) -> None:
    """Create SQLite DB and ``ingestion_log`` table if missing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_log (
                title TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                status TEXT NOT NULL,
                detail TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (title, entity_type)
            )
            """
        )
        con.commit()
    finally:
        con.close()


def get_ingestion_status(
    db_path: Path, title: str, entity_type: str
) -> str | None:
    """Return last ``status`` for ``(title, entity_type)``, or ``None``."""
    if not db_path.is_file():
        return None
    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "SELECT status FROM ingestion_log WHERE title = ? AND entity_type = ?",
            (title, entity_type),
        ).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def record_ingestion_status(
    db_path: Path,
    title: str,
    entity_type: str,
    status: str,
    detail: str | None = None,
) -> None:
    """Insert or replace ingestion row for one entity."""
    init_ingestion_db(db_path)
    ts = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            INSERT INTO ingestion_log (title, entity_type, status, detail, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(title, entity_type) DO UPDATE SET
                status = excluded.status,
                detail = excluded.detail,
                updated_at = excluded.updated_at
            """,
            (title, entity_type, status, detail, ts),
        )
        con.commit()
    finally:
        con.close()


def ingest_one_entity(
    title: str,
    entity_type: str,
    raw_root: Path,
    db_path: Path,
) -> str:
    """
    Fetch, parse, save, and log one entity.

    Returns:
        ``"fetched"``, ``"skipped"`` (already complete), or ``"failed"``.
    """
    if get_ingestion_status(db_path, title, entity_type) == "complete":
        if (raw_root / entity_type / f"{title.replace(' ', '_')}.txt").is_file():
            return "skipped"

    try:
        html = fetch_wikipedia_html(title)
        paragraphs = parse_article_paragraphs(html)
        if not paragraphs:
            record_ingestion_status(
                db_path, title, entity_type, "failed", "no paragraphs extracted"
            )
            return "failed"
        out = raw_root / entity_type / f"{title.replace(' ', '_')}.txt"
        save_raw_text(out, paragraphs)
        record_ingestion_status(db_path, title, entity_type, "complete", None)
        return "fetched"
    except Exception as err:  # noqa: BLE001
        record_ingestion_status(db_path, title, entity_type, "failed", str(err))
        return "failed"


def ingest_catalog(
    entities: list[tuple[str, str]],
    raw_root: Path,
    db_path: Path,
) -> dict[str, int]:
    """
    Ingest many ``(title, entity_type)`` tuples; return ``fetched`` / ``skipped`` / ``failed`` counts.
    """
    init_ingestion_db(db_path)
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "person").mkdir(exist_ok=True)
    (raw_root / "place").mkdir(exist_ok=True)

    fetched = skipped = failed = 0
    for title, entity_type in entities:
        result = ingest_one_entity(title, entity_type, raw_root, db_path)
        if result == "fetched":
            fetched += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1
    return {"fetched": fetched, "skipped": skipped, "failed": failed}


def ingest_all(
    famous_people: list[str],
    famous_places: list[str],
) -> dict[str, int]:
    """
    Ingest all configured people and places into ``config.RAW_DATA_DIR``.

    Returns:
        Counts ``fetched``, ``skipped``, ``failed`` (pages attempted).
    """
    config.ensure_data_directories()
    entities = [(t, "person") for t in famous_people] + [
        (t, "place") for t in famous_places
    ]
    db_path = Path(config.SQLITE_DB_PATH)
    return ingest_catalog(entities, config.RAW_DATA_DIR, db_path)
