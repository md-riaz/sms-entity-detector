# sms-entity-detector

A complete, runnable Python system that flags SMS templates which do **not** clearly identify the sender.

See [`sms_identifier/README.md`](sms_identifier/README.md) for full documentation, API reference, and usage instructions.

## Quick Start

```bash
cd sms_identifier
cp .env.example .env
docker compose up --build
```

API available at **http://localhost:8117**