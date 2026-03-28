"""
Configuration management via environment variables.
All paths default to ./data/* relative to the project root.
"""

import os
from pathlib import Path

# Base data directory – override with DATA_DIR env var
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))

# SQLite
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "sqlite" / "sms_identifier.db")))

# Queue files
QUEUE_DIR = Path(os.getenv("QUEUE_DIR", str(DATA_DIR / "queue")))
PENDING_QUEUE_FILE = QUEUE_DIR / "pending.jsonl"
PROCESSING_QUEUE_FILE = QUEUE_DIR / "processing.jsonl"

# Log files
LOG_DIR = Path(os.getenv("LOG_DIR", str(DATA_DIR / "logs")))
RESULTS_LOG_FILE = LOG_DIR / "results.jsonl"
APP_LOG_FILE = LOG_DIR / "app.log"

# Worker settings
WORKER_BATCH_SIZE = int(os.getenv("WORKER_BATCH_SIZE", "50"))
WORKER_SLEEP_SECONDS = float(os.getenv("WORKER_SLEEP_SECONDS", "2.0"))

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8117"))

# Model settings
NER_MODEL_NAME = os.getenv("NER_MODEL_NAME", "Davlan/xlm-roberta-base-ner-hrl")
# Set to "false" to disable model loading (useful for tests / lightweight mode)
MODEL_ENABLED = os.getenv("MODEL_ENABLED", "true").lower() == "true"
# NER entity labels considered as ORG / sender identity signals
ORG_LABELS = {"ORG", "PER", "LOC"}  # xlm-roberta-base-ner-hrl uses B-ORG / I-ORG
ORG_SCORE_THRESHOLD = float(os.getenv("ORG_SCORE_THRESHOLD", "0.80"))

# Minimum token length to be treated as a meaningful entity name (guards against
# very short false-positive NER hits like "I" or single-digit tokens)
MIN_ENTITY_LENGTH = int(os.getenv("MIN_ENTITY_LENGTH", "3"))


def ensure_directories() -> None:
    """Create all required runtime directories if they do not exist."""
    for directory in [DATA_DIR, DB_PATH.parent, QUEUE_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
