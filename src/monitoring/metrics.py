from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

COUNTERS: dict[str, int] = defaultdict(int)
LATENCY: dict[str, list[float]] = defaultdict(list)


@contextmanager
def track(endpoint: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        COUNTERS[f"{endpoint}_requests_total"] += 1
        LATENCY[f"{endpoint}_latency_seconds"].append(time.perf_counter() - start)


def snapshot() -> dict:
    latency_summary = {}
    for key, values in LATENCY.items():
        if values:
            latency_summary[key] = {
                "count": len(values),
                "avg": round(sum(values) / len(values), 4),
                "max": round(max(values), 4),
            }
    return {
        "counters": dict(COUNTERS),
        "latency": latency_summary,
        "drift": {"status": "placeholder", "score": None},
    }

