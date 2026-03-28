"""
FastAPI application entry point.

Routes:
  POST /api/v1/sms/ingest
  POST /api/v1/sms/check
  GET  /api/v1/template/{template_hash}
  GET  /api/v1/health
  GET  /api/v1/stats
  POST /api/v1/worker/run-once   (dev/testing only)
"""

import logging
import logging.config
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Body

from app import classifier, db, queue_manager, services, stats
from app.config import APP_LOG_FILE, LOG_DIR, ensure_directories
from app.schemas import (
    CacheRecord,
    CheckRequest,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MessageResult,
    StatsResponse,
    WorkerRunOnceResponse,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    ensure_directories()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": str(APP_LOG_FILE),
                    "formatter": "standard",
                    "encoding": "utf-8",
                },
            },
            "root": {
                "handlers": ["console", "file"],
                "level": "INFO",
            },
        }
    )


logger = logging.getLogger(__name__)

_start_time = time.time()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    setup_logging()
    ensure_directories()
    db.init_db()
    # Recover any leftover processing file from a previous crash
    queue_manager.recover_processing_file()
    # Do NOT preload the ML model in the API process.
    # Model startup is expensive and the API should become healthy fast.
    # The worker / classifier path will lazy-load it only when actually needed.
    logger.info("SMS Identifier API started on port configured in config.py")
    yield
    logger.info("SMS Identifier API shutting down")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SMS Entity Detector",
    description="Flags SMS templates that do not identify the sender.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/v1/sms/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Ingest one or many SMS messages.
    Returns per-message normalized template, hash, and cache/queue status.
    """
    results = services.ingest_sms(request.messages)
    return IngestResponse(processed=len(results), results=results)


@app.post("/api/v1/sms/check", response_model=MessageResult)
def check(body: CheckRequest = Body(...)):
    """
    Check a single SMS message for sender identity.
    Returns cached result if available, otherwise queues and returns PENDING.

    Body: { "message": "<raw sms text>" }
    """
    if not body.message:
        raise HTTPException(status_code=422, detail="Field 'message' is required")
    return services.check_sms(body.message)


@app.get("/api/v1/template/{template_hash}", response_model=CacheRecord)
def get_template(template_hash: str):
    """Return cached classification record for a given template hash."""
    record = db.cache_get(template_hash)
    if not record:
        raise HTTPException(status_code=404, detail="Template not found in cache")
    return CacheRecord(**record)


@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    """Service health check – used by Docker healthcheck and monitoring."""
    db_ok = True
    try:
        db.cache_count()  # simple DB probe
    except Exception:
        db_ok = False

    from app.config import PENDING_QUEUE_FILE

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db_ok=db_ok,
        model_loaded=classifier.is_model_ready(),
        model_error=classifier.model_error(),
        queue_file_exists=PENDING_QUEUE_FILE.exists(),
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/api/v1/stats", response_model=StatsResponse)
def get_stats():
    """Return operational metrics."""
    s = stats.get_stats()
    return StatsResponse(**s)


@app.post("/api/v1/worker/run-once", response_model=WorkerRunOnceResponse)
def worker_run_once():
    """
    Manually trigger one worker batch cycle.
    For development and testing only.
    """
    from worker.run_worker import process_batch

    batch_size, processed, rule_count, model_count = process_batch()
    return WorkerRunOnceResponse(
        batch_size=batch_size,
        processed=processed,
        rule_decisions=rule_count,
        model_decisions=model_count,
        message=f"Processed {processed} template(s) from queue",
    )
