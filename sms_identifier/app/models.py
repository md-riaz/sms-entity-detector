"""
Audit / results log writer.

Writes JSONL entries to data/logs/results.jsonl for human-readable audit trail.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from app.config import RESULTS_LOG_FILE, LOG_DIR

logger = logging.getLogger(__name__)


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def write_result_log(
    template_hash: str,
    template_text: str,
    result: str,
    source: str,
    confidence: Optional[float] = None,
) -> None:
    """Append a classification result entry to the JSONL results log."""
    _ensure_log_dir()
    record = {
        "template_hash": template_hash,
        "template_text": template_text,
        "result": result,
        "source": source,
        "confidence": confidence,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with RESULTS_LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Failed to write result log: %s", exc)
