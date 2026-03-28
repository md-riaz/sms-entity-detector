# SMS Entity Detector

A production-safe Python system that flags SMS templates which **do not** clearly identify the sender (brand / company / app / website / domain / service name).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Why Template Caching Matters](#why-template-caching-matters)
3. [Project Structure](#project-structure)
4. [Quick Start – Local (no Docker)](#quick-start--local-no-docker)
5. [Quick Start – Docker Compose](#quick-start--docker-compose)
6. [Environment Variables](#environment-variables)
7. [API Reference](#api-reference)
8. [Sample SMS Examples](#sample-sms-examples)
9. [Worker Modes](#worker-modes)
10. [Running Tests](#running-tests)
11. [Design Decisions](#design-decisions)
12. [Single-Node Assumptions](#single-node-assumptions)
13. [Extending the System](#extending-the-system)

---

## Architecture Overview

```
Raw SMS
   │
   ▼
Normalize → Template Text + SHA-1 Hash
   │
   ├─► Cache Hit?  ──YES──► Return cached result
   │
   └─► No → Queued?  ──YES──► Skip (duplicate prevention)
               │
               └─► NO → Append to JSONL queue
                          Register in SQLite queued_templates
                          Return PENDING

                ┌────────────────────┐
                │   Worker Process   │
                │ (batch, looping)   │
                └────────┬───────────┘
                         │
                    Take batch from queue
                         │
                    Rule Engine (cheap)
                    ├── PASS/FLAG → write to cache (source=rule)
                    └── Undecided → ML model (source=model)
                         │
                    Write result to template_cache
                    Remove from queued_templates
                    Write audit log
```

### Classification goal

| Condition | Result |
|-----------|--------|
| SMS clearly identifies sender (brand / app / domain / service) | **PASS** |
| SMS does NOT identify the sender | **FLAG** |

---

## Why Template Caching Matters

High-volume SMS platforms process millions of messages, but most are **repeated templates** – OTPs, payment confirmations, order updates – where only dynamic values change (amounts, dates, codes).

Without caching, every SMS would trigger an ML inference call (slow, expensive).

With template normalization + SHA-1 caching:

1. **Normalize** – replace dynamic values with typed placeholders (`{NUM}`, `{URL}`, etc.)
2. **Hash** – compute a stable SHA-1 fingerprint of the normalized template
3. **Cache hit** – if we've seen this template before, return the stored result instantly (no ML)
4. **Cache miss** – queue the template once, run ML once, store the result

**In practice**: if 10,000 SMS use the same OTP template, only the very first one ever goes to the ML model. All 9,999 others are cache hits.

---

## Project Structure

```
sms_identifier/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, lifespan, routes
│   ├── config.py         # All env-var backed configuration
│   ├── db.py             # SQLite schema + helpers
│   ├── models.py         # Audit log writer
│   ├── schemas.py        # Pydantic request/response models
│   ├── normalization.py  # SMS → template, SHA-1 hash
│   ├── rules.py          # Cheap structural rule engine
│   ├── classifier.py     # HuggingFace NER wrapper
│   ├── queue_manager.py  # JSONL file-based queue
│   ├── services.py       # Business logic (ingest / check)
│   └── stats.py          # Operational metrics
├── worker/
│   ├── __init__.py
│   └── run_worker.py     # Worker loop + batch processor
├── data/
│   ├── queue/
│   │   ├── pending.jsonl       # Append-only queue file
│   │   └── processing.jsonl    # Temp file during batch processing
│   ├── logs/
│   │   ├── results.jsonl       # Classification audit trail
│   │   └── app.log             # Application log
│   └── sqlite/
│       └── sms_identifier.db   # Persistent SQLite database
├── tests/
│   ├── __init__.py
│   ├── test_normalization.py
│   ├── test_rules.py
│   └── test_queue.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Quick Start – Local (no Docker)

### Prerequisites

- Python 3.11+
- ~2 GB disk space for the ML model (downloaded automatically on first run)

### Setup

```bash
cd sms_identifier

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8117
```

The API will be available at `http://localhost:8117`.

### Start the worker (separate terminal)

```bash
source .venv/bin/activate
cd sms_identifier
python -m worker.run_worker
```

---

## Quick Start – Docker Compose

```bash
cd sms_identifier

# Copy and review environment config
cp .env.example .env

# Build and start both services
docker compose up --build

# Or run in background
docker compose up --build -d

# View logs
docker compose logs -f api
docker compose logs -f worker
```

Services:
- **API**: `http://localhost:8117`
- **Worker**: runs in background, processes queue continuously

The `./data` directory is mounted into both containers so the SQLite database,
queue files, and logs persist across restarts.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `./data` | Root directory for all runtime data |
| `DB_PATH` | `./data/sqlite/sms_identifier.db` | SQLite database path |
| `QUEUE_DIR` | `./data/queue` | Queue file directory |
| `LOG_DIR` | `./data/logs` | Log file directory |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8117` | API port |
| `WORKER_BATCH_SIZE` | `50` | Templates processed per batch |
| `WORKER_SLEEP_SECONDS` | `2.0` | Sleep between empty-queue polls |
| `NER_MODEL_NAME` | `Davlan/xlm-roberta-base-ner-hrl` | HuggingFace model identifier |
| `MODEL_ENABLED` | `true` | Set `false` for rule-only mode |
| `ORG_SCORE_THRESHOLD` | `0.80` | Minimum NER confidence for PASS |
| `MIN_ENTITY_LENGTH` | `3` | Minimum entity name length |

---

## API Reference

### POST `/api/v1/sms/ingest`

Ingest one or many SMS messages.

**Request:**
```json
{
  "messages": [
    "Your OTP is 123456",
    "bKash OTP is 123456",
    "Daraz: Order confirmed",
    "Payment received successfully",
    "Visit https://company.com to continue"
  ]
}
```

**Response:**
```json
{
  "processed": 5,
  "results": [
    {
      "original_text": "Your OTP is 123456",
      "template_text": "Your OTP is {NUM}",
      "template_hash": "a3f1...",
      "status": "queued"
    },
    {
      "original_text": "bKash OTP is 123456",
      "template_text": "bKash OTP is {NUM}",
      "template_hash": "b2e9...",
      "status": "queued"
    },
    {
      "original_text": "Daraz: Order confirmed",
      "template_text": "Daraz: Order confirmed",
      "template_hash": "c4d2...",
      "status": "cached",
      "result": "PASS",
      "confidence": 0.95,
      "source": "rule"
    }
  ]
}
```

**cURL:**
```bash
curl -X POST http://localhost:8117/api/v1/sms/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      "Your OTP is 123456",
      "bKash OTP is 123456",
      "Daraz: Order confirmed"
    ]
  }'
```

---

### POST `/api/v1/sms/check`

Check a single SMS immediately.

**Request:**
```json
{ "message": "bKash OTP is 123456" }
```

**Response (cached):**
```json
{
  "original_text": "bKash OTP is 123456",
  "template_text": "bKash OTP is {NUM}",
  "template_hash": "b2e9...",
  "status": "cached",
  "result": "PASS",
  "confidence": 0.92,
  "source": "model"
}
```

**cURL:**
```bash
curl -X POST http://localhost:8117/api/v1/sms/check \
  -H "Content-Type: application/json" \
  -d '{"message": "bKash OTP is 123456"}'
```

---

### GET `/api/v1/template/{template_hash}`

Look up a cached result by template hash.

**cURL:**
```bash
curl http://localhost:8117/api/v1/template/b2e9abc123def456...
```

**Response:**
```json
{
  "template_hash": "b2e9...",
  "template_text": "bKash OTP is {NUM}",
  "result": "PASS",
  "confidence": 0.92,
  "source": "model",
  "created_at": "2024-06-01T10:00:00",
  "updated_at": "2024-06-01T10:00:00"
}
```

---

### GET `/api/v1/health`

Service health check.

**cURL:**
```bash
curl http://localhost:8117/api/v1/health
```

**Response:**
```json
{
  "status": "ok",
  "db_ok": true,
  "model_loaded": true,
  "model_error": null,
  "queue_file_exists": true,
  "uptime_seconds": 120.5
}
```

---

### GET `/api/v1/stats`

Operational metrics.

**cURL:**
```bash
curl http://localhost:8117/api/v1/stats
```

**Response:**
```json
{
  "total_cached": 142,
  "total_queued": 3,
  "queue_file_lines": 3,
  "cached_by_source": {
    "rule": 98,
    "model": 44
  },
  "uptime_seconds": 3600.0
}
```

---

### POST `/api/v1/worker/run-once`

Manually trigger one batch cycle (development / testing).

**cURL:**
```bash
curl -X POST http://localhost:8117/api/v1/worker/run-once
```

**Response:**
```json
{
  "batch_size": 50,
  "processed": 5,
  "rule_decisions": 3,
  "model_decisions": 2,
  "message": "Processed 5 template(s) from queue"
}
```

---

## Sample SMS Examples

| SMS | Template | Expected |
|-----|----------|----------|
| `Your OTP is 123456` | `Your OTP is {NUM}` | **FLAG** – no sender |
| `bKash OTP is 123456` | `bKash OTP is {NUM}` | **PASS** – bKash identified |
| `Daraz: Order confirmed` | `Daraz: Order confirmed` | **PASS** – Daraz prefix |
| `Payment received successfully` | `Payment received successfully` | **FLAG** – no sender |
| `Visit https://company.com` | `Visit {URL}` | **PASS** – URL present |
| `Pathao: Your ride is confirmed` | `Pathao: Your ride is confirmed` | **PASS** – Pathao prefix |
| `Transaction ID TXN123ABC` | `Transaction ID {ID}` | **FLAG** – no sender |
| `Notification from Grameenphone` | `Notification from Grameenphone` | **PASS** – named sender |
| `Your package is shipped - Daraz` | `Your package is shipped - Daraz` | **PASS** – signature suffix |

---

## Worker Modes

### Production loop mode
```bash
python -m worker.run_worker
```
Runs continuously, polling the queue every `WORKER_SLEEP_SECONDS`.

### One-shot mode (for testing)
```bash
python -m worker.run_worker --once
```
Processes one batch and exits. Useful in CI or for debugging.

### Via API (development)
```bash
curl -X POST http://localhost:8117/api/v1/worker/run-once
```

---

## Running Tests

```bash
cd sms_identifier
pip install -r requirements.txt
pytest tests/ -v
```

Tests cover:
- Normalization: URL, email, date, time, number, ID replacement
- Rules: PASS / FLAG / undecided cases
- Queue: enqueue, take_batch, commit, rollback, line count

The tests do **not** require a running API server or ML model.

---

## Design Decisions

### Rule engine before ML

Structural rules (URL presence, sender prefix, signature suffix, capitalized brand token)
can resolve ~60–80% of templates without any ML inference. This dramatically reduces
latency and cost. Only genuinely ambiguous templates reach the model.

### File-based queue (JSONL)

A JSONL file is chosen over Redis/Kafka because:
- No external dependencies
- Persists naturally on disk
- Atomic rename strategy prevents double-processing
- Crash recovery via leftover `processing.jsonl` merge

### SQLite for cache and registry

SQLite is sufficient for single-node deployments handling millions of templates.
WAL mode enables concurrent readers. The template cache is the primary hot path.

### Pretrained NER model only

`Davlan/xlm-roberta-base-ner-hrl` is multilingual and recognizes ORG, PER, LOC
entities out of the box. No training data required. The model is loaded once at
startup and reused for all inferences.

---

## Single-Node Assumptions

The file-based JSONL queue and SQLite database are designed for **single-node deployment**.

If you need to scale to multiple worker nodes:
- Replace the JSONL queue with a distributed queue (Redis Streams, SQS, etc.)
- Replace SQLite with PostgreSQL
- Keep the rule engine and classifier logic unchanged

The code is structured so only `queue_manager.py` and `db.py` need to change for
a multi-node upgrade.

---

## Extending the System

### Adding a second classifier

1. Add a new function in `classifier.py` (e.g., `classify_zero_shot()`)
2. Update `classify()` to call the new function as a second fallback if the
   primary model returns low confidence

### Adding new normalization patterns

Edit `normalization.py` – add a new compiled regex and a replacement call in
`normalize()`. Order matters; more-specific patterns should run first.

### Adding new rules

Edit `rules.py` – add a new `if` block in `apply_rules()`. Rules are evaluated
in order; return early with high confidence when the signal is strong.
