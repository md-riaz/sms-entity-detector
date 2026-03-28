"""
Worker process – continuously reads queued templates from the JSONL queue,
applies the rule engine and ML classifier, and stores results in cache.

Usage:
  python -m worker.run_worker          # production loop
  python -m worker.run_worker --once   # process one batch and exit
"""

import logging
import sys
import time
from typing import Optional

# Ensure the sms_identifier package root is on sys.path when run directly
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app import db, classifier, queue_manager
from app.config import WORKER_BATCH_SIZE, WORKER_SLEEP_SECONDS, ensure_directories
from app.models import write_result_log
from app import rules

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core batch processor (reusable by both worker loop and API run-once)
# ---------------------------------------------------------------------------

def process_batch() -> tuple[int, int, int, int]:
    """
    Read one batch from the queue, classify each template, store results.

    Returns:
        (batch_size_config, num_processed, rule_decisions, model_decisions)
    """
    batch = queue_manager.take_batch(WORKER_BATCH_SIZE)
    if not batch:
        return WORKER_BATCH_SIZE, 0, 0, 0

    logger.info("Worker batch size=%d", len(batch))

    # Separate items needing model inference from those resolved by rules
    rule_items: list[tuple[dict, str, Optional[float]]] = []
    model_items: list[dict] = []

    for item in batch:
        template_hash = item.get("template_hash", "")
        template_text = item.get("template_text", "")

        if not template_hash or not template_text:
            logger.warning("Skipping malformed queue item: %s", item)
            continue

        # Check cache first (item might have been processed via /check endpoint)
        cached = db.cache_get(template_hash)
        if cached:
            logger.info("Worker cache hit (skip)  hash=%s", template_hash)
            db.queue_registry_remove(template_hash)
            continue

        # Rule layer
        result, confidence = rules.apply_rules(template_text)
        if result is not None:
            rule_items.append((item, result, confidence))
        else:
            model_items.append(item)

    # Process rule-resolved items
    rule_count = 0
    for item, result, confidence in rule_items:
        template_hash = item["template_hash"]
        template_text = item["template_text"]
        db.cache_set(template_hash, template_text, result, "rule", confidence)
        db.audit_log(template_hash, template_text, result, "rule", confidence)
        write_result_log(template_hash, template_text, result, "rule", confidence)
        db.queue_registry_remove(template_hash)
        logger.info(
            "Rule decision  hash=%s  result=%s  confidence=%s",
            template_hash, result, confidence,
        )
        rule_count += 1

    # Process model-inference items
    model_count = 0
    if model_items:
        logger.info("Model fallback count=%d", len(model_items))
        texts = [item["template_text"] for item in model_items]
        try:
            model_results = classifier.classify_batch(texts)
        except Exception as exc:
            logger.error("Batch classification failed: %s – rolling back batch", exc)
            queue_manager.rollback_batch()
            return WORKER_BATCH_SIZE, 0, rule_count, 0

        for item, (result, confidence) in zip(model_items, model_results):
            template_hash = item["template_hash"]
            template_text = item["template_text"]
            db.cache_set(template_hash, template_text, result, "model", confidence)
            db.audit_log(template_hash, template_text, result, "model", confidence)
            write_result_log(template_hash, template_text, result, "model", confidence)
            db.queue_registry_remove(template_hash)
            logger.info(
                "Model decision  hash=%s  result=%s  confidence=%s",
                template_hash, result, confidence,
            )
            model_count += 1

    queue_manager.commit_batch()
    total_processed = rule_count + model_count
    logger.info(
        "Batch complete  total=%d  rule=%d  model=%d",
        total_processed, rule_count, model_count,
    )
    return WORKER_BATCH_SIZE, total_processed, rule_count, model_count


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def run_loop() -> None:
    """Run the worker in a continuous loop."""
    logger.info("Worker starting loop  batch_size=%d  sleep=%.1fs",
                WORKER_BATCH_SIZE, WORKER_SLEEP_SECONDS)
    while True:
        try:
            _, processed, _, _ = process_batch()
            if processed == 0:
                time.sleep(WORKER_SLEEP_SECONDS)
        except KeyboardInterrupt:
            logger.info("Worker interrupted – exiting")
            break
        except Exception as exc:
            logger.error("Unexpected worker error: %s", exc, exc_info=True)
            time.sleep(WORKER_SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _setup_worker_logging() -> None:
    from app.config import APP_LOG_FILE, LOG_DIR
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(APP_LOG_FILE), encoding="utf-8"),
        ],
    )


if __name__ == "__main__":
    _setup_worker_logging()
    ensure_directories()
    db.init_db()
    queue_manager.recover_processing_file()
    # Do NOT preload the model here either.
    # It will lazy-load only if a batch actually contains undecided items.

    if "--once" in sys.argv:
        logger.info("Worker running one batch (--once mode)")
        _, processed, rule_c, model_c = process_batch()
        print(f"Processed {processed} templates  rule={rule_c}  model={model_c}")
    else:
        run_loop()
