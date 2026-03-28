"""
Pydantic schemas for API request and response models.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingest / Check
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    messages: list[str] = Field(
        ...,
        min_length=1,
        description="One or more raw SMS texts",
        examples=[
            [
                "Your OTP is 123456",
                "bKash OTP is 123456",
                "Daraz: Order confirmed",
            ]
        ],
    )


class CheckRequest(BaseModel):
    message: str = Field(
        ...,
        description="One raw SMS text",
        examples=["bKash OTP is 123456"],
    )


class MessageResult(BaseModel):
    original_text: str
    template_text: str
    template_hash: str
    status: str                          # "cached" | "queued" | "pending"
    result: Optional[str] = None         # "PASS" | "FLAG" – only when cached
    confidence: Optional[float] = None
    source: Optional[str] = None         # "rule" | "model" – only when cached


class IngestResponse(BaseModel):
    processed: int
    results: list[MessageResult]


# ---------------------------------------------------------------------------
# Template lookup
# ---------------------------------------------------------------------------

class CacheRecord(BaseModel):
    template_hash: str
    template_text: str
    result: str
    confidence: Optional[float]
    source: str
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str                           # "ok" | "degraded"
    db_ok: bool
    model_loaded: bool
    model_error: Optional[str]
    queue_file_exists: bool
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class StatsResponse(BaseModel):
    total_cached: int
    total_queued: int
    queue_file_lines: int
    cached_by_source: dict[str, int]
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Worker run-once
# ---------------------------------------------------------------------------

class WorkerRunOnceResponse(BaseModel):
    batch_size: int
    processed: int
    rule_decisions: int
    model_decisions: int
    message: str
