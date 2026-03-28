"""
Operational statistics helpers.
"""

import time
from typing import Any

from app import db, queue_manager

_start_time: float = time.time()


def get_uptime() -> float:
    return round(time.time() - _start_time, 2)


def get_stats() -> dict[str, Any]:
    return {
        "total_cached": db.cache_count(),
        "total_queued": db.queue_registry_count(),
        "queue_file_lines": queue_manager.queue_line_count(),
        "cached_by_source": db.cache_count_by_source(),
        "uptime_seconds": get_uptime(),
    }
