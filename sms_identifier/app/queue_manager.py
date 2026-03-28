"""
File-based JSONL queue manager.

Queue design:
- pending.jsonl  – append-only; workers read from this file
- processing.jsonl – atomic rename during processing to prevent re-processing
- On crash recovery: if processing.jsonl exists at startup, it is merged back
  into pending.jsonl so items are not lost.

Each line in the queue file is a JSON object:
{
    "template_hash": "<sha1>",
    "template_text": "<normalized>",
    "queued_at": "<ISO datetime>"
}
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.config import PENDING_QUEUE_FILE, PROCESSING_QUEUE_FILE, QUEUE_DIR

logger = logging.getLogger(__name__)


def _ensure_queue_dir() -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------

def enqueue(template_hash: str, template_text: str) -> None:
    """Append a new template entry to the pending queue file."""
    _ensure_queue_dir()
    record = {
        "template_hash": template_hash,
        "template_text": template_text,
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    # Open in append mode – atomic for single-writer; safe for single-container deploy
    with PENDING_QUEUE_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line)
    logger.info("Queue append  hash=%s", template_hash)


# ---------------------------------------------------------------------------
# Dequeue (batch read)
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    """Read all valid JSON lines from *path*. Skips malformed lines."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed queue line %d: %s", lineno, exc)
    return records


def recover_processing_file() -> None:
    """
    If a processing.jsonl file exists (leftover from a previous crash),
    merge its contents back into pending.jsonl so those items are retried.
    """
    if not PROCESSING_QUEUE_FILE.exists():
        return
    logger.warning(
        "Found leftover processing.jsonl – recovering items back to pending queue"
    )
    records = _read_jsonl(PROCESSING_QUEUE_FILE)
    if records:
        _ensure_queue_dir()
        with PENDING_QUEUE_FILE.open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Recovered %d items from processing.jsonl", len(records))
    PROCESSING_QUEUE_FILE.unlink(missing_ok=True)


def take_batch(batch_size: int) -> list[dict]:
    """
    Atomically move up to *batch_size* items from pending.jsonl to
    processing.jsonl, returning the records for the caller to process.

    Uses an atomic rename so the items are "locked" during processing.
    After the worker finishes, it calls commit_batch() or rollback_batch().
    """
    if not PENDING_QUEUE_FILE.exists() or PENDING_QUEUE_FILE.stat().st_size == 0:
        return []

    # Rename pending → processing (atomic on same filesystem)
    try:
        os.rename(PENDING_QUEUE_FILE, PROCESSING_QUEUE_FILE)
    except OSError as exc:
        logger.error("Failed to rename queue file: %s", exc)
        return []

    all_records = _read_jsonl(PROCESSING_QUEUE_FILE)

    # Deduplicate by template_hash (preserve first occurrence)
    seen: set[str] = set()
    unique_records = []
    leftover = []
    for rec in all_records:
        h = rec.get("template_hash", "")
        if h not in seen:
            seen.add(h)
            unique_records.append(rec)
        else:
            leftover.append(rec)  # genuine duplicate lines – will be discarded

    batch = unique_records[:batch_size]
    remaining = unique_records[batch_size:]

    # Write remaining (beyond batch) back to pending
    if remaining:
        _ensure_queue_dir()
        with PENDING_QUEUE_FILE.open("w", encoding="utf-8") as fh:
            for rec in remaining:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return batch


def commit_batch() -> None:
    """
    Called after successfully processing a batch.
    Removes the processing file.
    """
    PROCESSING_QUEUE_FILE.unlink(missing_ok=True)
    logger.debug("Processing file removed – batch committed")


def rollback_batch() -> None:
    """
    Called when batch processing fails.
    Merges processing.jsonl back into pending.jsonl for retry.
    """
    recover_processing_file()


# ---------------------------------------------------------------------------
# Queue stats
# ---------------------------------------------------------------------------

def queue_line_count() -> int:
    """Estimate queue depth by counting non-empty lines in pending.jsonl."""
    if not PENDING_QUEUE_FILE.exists():
        return 0
    count = 0
    with PENDING_QUEUE_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count
