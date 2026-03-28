"""
Service layer – orchestrates normalization, cache, queue, rules, and classifier.

The ingest_sms() and check_sms() functions are the primary business-logic
entry points used by the API routes.
"""

import logging
from typing import Optional

from app import db, normalization, rules, classifier, queue_manager
from app.models import write_result_log
from app.schemas import MessageResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core pipeline helpers
# ---------------------------------------------------------------------------

def _process_single_message(raw_text: str) -> MessageResult:
    """
    Normalize *raw_text*, check cache, and enqueue if unseen.
    Returns a MessageResult (never raises).
    """
    template_text = normalization.normalize(raw_text)
    template_hash = normalization.compute_hash(template_text)

    # 1. Cache hit → return immediately
    cached = db.cache_get(template_hash)
    if cached:
        logger.info("Cache hit  hash=%s  result=%s", template_hash, cached["result"])
        return MessageResult(
            original_text=raw_text,
            template_text=template_text,
            template_hash=template_hash,
            status="cached",
            result=cached["result"],
            confidence=cached["confidence"],
            source=cached["source"],
        )

    # 2. Cache miss – try to enqueue (duplicate-safe via DB registry)
    newly_registered = db.queue_registry_add(template_hash, template_text)
    if newly_registered:
        queue_manager.enqueue(template_hash, template_text)
        logger.info("Queue append  hash=%s", template_hash)
    else:
        logger.info("Duplicate queue skip  hash=%s", template_hash)

    return MessageResult(
        original_text=raw_text,
        template_text=template_text,
        template_hash=template_hash,
        status="queued" if newly_registered else "pending",
    )


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------

def ingest_sms(messages: list[str]) -> list[MessageResult]:
    """
    Process a list of raw SMS messages.
    Returns per-message results (cached or queued).
    """
    return [_process_single_message(m) for m in messages]


def check_sms(raw_text: str) -> MessageResult:
    """
    Check a single SMS message.
    Returns cached result if available; otherwise enqueues and returns PENDING.
    """
    return _process_single_message(raw_text)


# ---------------------------------------------------------------------------
# Classification helper (used by worker)
# ---------------------------------------------------------------------------

def classify_template(
    template_hash: str,
    template_text: str,
) -> tuple[str, Optional[float], str]:
    """
    Classify a single template through rules → model pipeline.

    Returns (result, confidence, source).
    Writes to cache and audit log.
    """
    # Rule layer first
    result, confidence = rules.apply_rules(template_text)

    if result is not None:
        source = "rule"
        logger.info(
            "Rule decision  hash=%s  result=%s  confidence=%s",
            template_hash,
            result,
            confidence,
        )
    else:
        # Fall through to ML model
        result, confidence = classifier.classify(template_text)
        source = "model"
        logger.info(
            "Model decision  hash=%s  result=%s  confidence=%s",
            template_hash,
            result,
            confidence,
        )

    # Persist to cache
    db.cache_set(template_hash, template_text, result, source, confidence)

    # Audit DB
    db.audit_log(template_hash, template_text, result, source, confidence)

    # Audit JSONL log
    write_result_log(template_hash, template_text, result, source, confidence)

    return result, confidence, source
