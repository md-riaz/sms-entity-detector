"""
Tests for the queue manager.
Uses a temporary directory so it does not pollute the real queue.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest


@pytest.fixture()
def tmp_queue(tmp_path, monkeypatch):
    """
    Override queue paths to use a temporary directory for each test.
    Patches the module-level constants in queue_manager.
    """
    import app.queue_manager as qm
    import app.config as cfg

    pending = tmp_path / "pending.jsonl"
    processing = tmp_path / "processing.jsonl"

    monkeypatch.setattr(qm, "PENDING_QUEUE_FILE", pending)
    monkeypatch.setattr(qm, "PROCESSING_QUEUE_FILE", processing)
    monkeypatch.setattr(qm, "QUEUE_DIR", tmp_path)
    monkeypatch.setattr(cfg, "PENDING_QUEUE_FILE", pending)
    monkeypatch.setattr(cfg, "PROCESSING_QUEUE_FILE", processing)

    return tmp_path


class TestEnqueue:
    def test_enqueue_creates_file(self, tmp_queue):
        from app.queue_manager import enqueue, PENDING_QUEUE_FILE

        enqueue("abc123", "Your OTP is {NUM}")
        assert PENDING_QUEUE_FILE.exists()

    def test_enqueue_writes_valid_json(self, tmp_queue):
        from app.queue_manager import enqueue, PENDING_QUEUE_FILE

        enqueue("abc123", "Your OTP is {NUM}")
        lines = PENDING_QUEUE_FILE.read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["template_hash"] == "abc123"
        assert rec["template_text"] == "Your OTP is {NUM}"
        assert "queued_at" in rec

    def test_enqueue_multiple(self, tmp_queue):
        from app.queue_manager import enqueue, PENDING_QUEUE_FILE

        enqueue("h1", "Template one")
        enqueue("h2", "Template two")
        lines = PENDING_QUEUE_FILE.read_text().strip().splitlines()
        assert len(lines) == 2


class TestTakeBatch:
    def test_take_batch_empty_queue(self, tmp_queue):
        from app.queue_manager import take_batch

        batch = take_batch(10)
        assert batch == []

    def test_take_batch_returns_records(self, tmp_queue):
        from app.queue_manager import enqueue, take_batch

        enqueue("h1", "Template one")
        enqueue("h2", "Template two")
        batch = take_batch(10)
        assert len(batch) == 2

    def test_take_batch_respects_size(self, tmp_queue):
        from app.queue_manager import enqueue, take_batch, PENDING_QUEUE_FILE

        for i in range(5):
            enqueue(f"h{i}", f"Template {i}")

        batch = take_batch(3)
        assert len(batch) == 3
        # Remaining 2 should be back in pending
        remaining_lines = PENDING_QUEUE_FILE.read_text().strip().splitlines()
        assert len(remaining_lines) == 2

    def test_take_batch_deduplicates(self, tmp_queue):
        from app.queue_manager import enqueue, take_batch, PENDING_QUEUE_FILE

        # Enqueue same hash twice
        enqueue("same_hash", "Template A")
        enqueue("same_hash", "Template A")
        batch = take_batch(10)
        assert len(batch) == 1


class TestCommitRollback:
    def test_commit_removes_processing_file(self, tmp_queue):
        from app.queue_manager import enqueue, take_batch, commit_batch, PROCESSING_QUEUE_FILE

        enqueue("h1", "Template one")
        take_batch(10)
        commit_batch()
        assert not PROCESSING_QUEUE_FILE.exists()

    def test_rollback_recovers_items(self, tmp_queue):
        from app.queue_manager import enqueue, take_batch, rollback_batch, PENDING_QUEUE_FILE

        enqueue("h1", "Template one")
        take_batch(10)
        rollback_batch()
        # Items should be back in pending
        assert PENDING_QUEUE_FILE.exists()
        lines = PENDING_QUEUE_FILE.read_text().strip().splitlines()
        assert len(lines) == 1


class TestQueueLineCount:
    def test_line_count_zero_when_empty(self, tmp_queue):
        from app.queue_manager import queue_line_count

        assert queue_line_count() == 0

    def test_line_count_matches_enqueued(self, tmp_queue):
        from app.queue_manager import enqueue, queue_line_count

        enqueue("h1", "T1")
        enqueue("h2", "T2")
        assert queue_line_count() == 2
