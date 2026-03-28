"""
SQLite database helpers.
Uses stdlib sqlite3 – no ORM.
Schema is auto-created on first import / startup.
"""

import logging
import sqlite3
from contextlib import contextmanager
from typing import Generator

from app.config import DB_PATH, ensure_directories

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent read performance
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_cursor() -> Generator[sqlite3.Cursor, None, None]:
    """Context manager that yields a cursor and commits/rolls-back automatically."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS template_cache (
    template_hash  TEXT PRIMARY KEY,
    template_text  TEXT NOT NULL,
    result         TEXT NOT NULL,          -- PASS | FLAG
    confidence     REAL,                   -- NULL if unavailable
    source         TEXT NOT NULL,          -- rule | model
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS queued_templates (
    template_hash  TEXT PRIMARY KEY,
    template_text  TEXT NOT NULL,
    queued_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS classification_audit (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    template_hash  TEXT NOT NULL,
    template_text  TEXT NOT NULL,
    result         TEXT NOT NULL,
    confidence     REAL,
    source         TEXT NOT NULL,
    processed_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db() -> None:
    """Initialize the database, creating tables if they do not exist."""
    ensure_directories()
    with db_cursor() as cur:
        cur.executescript(SCHEMA_SQL)
    logger.info("Database initialized at %s", DB_PATH)


# ---------------------------------------------------------------------------
# template_cache helpers
# ---------------------------------------------------------------------------

def cache_get(template_hash: str) -> dict | None:
    """Return cached record for *template_hash*, or None if not found."""
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM template_cache WHERE template_hash = ?",
            (template_hash,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def cache_set(
    template_hash: str,
    template_text: str,
    result: str,
    source: str,
    confidence: float | None = None,
) -> None:
    """Insert or replace a cache entry."""
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO template_cache
                (template_hash, template_text, result, confidence, source, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(template_hash) DO UPDATE SET
                result      = excluded.result,
                confidence  = excluded.confidence,
                source      = excluded.source,
                updated_at  = CURRENT_TIMESTAMP
            """,
            (template_hash, template_text, result, confidence, source),
        )
    logger.info(
        "Cache write  hash=%s  result=%s  source=%s  confidence=%s",
        template_hash,
        result,
        source,
        confidence,
    )


# ---------------------------------------------------------------------------
# queued_templates helpers
# ---------------------------------------------------------------------------

def queue_registry_add(template_hash: str, template_text: str) -> bool:
    """
    Register *template_hash* as queued.
    Returns True if it was newly inserted, False if it already existed.
    """
    with db_cursor() as cur:
        try:
            cur.execute(
                "INSERT INTO queued_templates (template_hash, template_text) VALUES (?, ?)",
                (template_hash, template_text),
            )
            return True
        except sqlite3.IntegrityError:
            # Already registered – duplicate skip
            return False


def queue_registry_remove(template_hash: str) -> None:
    """Remove *template_hash* from the queued registry after processing."""
    with db_cursor() as cur:
        cur.execute(
            "DELETE FROM queued_templates WHERE template_hash = ?",
            (template_hash,),
        )


def queue_registry_exists(template_hash: str) -> bool:
    """Return True if *template_hash* is currently in the queued registry."""
    with db_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM queued_templates WHERE template_hash = ?",
            (template_hash,),
        )
        return cur.fetchone() is not None


def queue_registry_count() -> int:
    """Return number of templates currently queued."""
    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM queued_templates")
        row = cur.fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# template_cache stats helpers
# ---------------------------------------------------------------------------

def cache_count() -> int:
    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM template_cache")
        row = cur.fetchone()
    return row[0] if row else 0


def cache_count_by_source() -> dict[str, int]:
    with db_cursor() as cur:
        cur.execute(
            "SELECT source, COUNT(*) as cnt FROM template_cache GROUP BY source"
        )
        rows = cur.fetchall()
    return {row["source"]: row["cnt"] for row in rows}


# ---------------------------------------------------------------------------
# audit log helper
# ---------------------------------------------------------------------------

def audit_log(
    template_hash: str,
    template_text: str,
    result: str,
    source: str,
    confidence: float | None = None,
) -> None:
    """Write an entry to the classification_audit table."""
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO classification_audit
                (template_hash, template_text, result, confidence, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (template_hash, template_text, result, confidence, source),
        )
