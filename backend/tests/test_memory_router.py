import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from backend.knowledge.router import MemoryRouter


class StubConsolidator:
    def __init__(self) -> None:
        self.calls = []

    def record_statement(self, text: str, *, source: str, metadata=None) -> None:
        self.calls.append({"text": text, "source": source, "metadata": metadata or {}})


def test_memory_router_adds_and_queries_short_term():
    consolidator = StubConsolidator()
    router = MemoryRouter(consolidator)

    router.add_observation("agent discovered new pattern", source="task:alpha")
    router.add_observation(
        "optimize neural controller parameters",
        source="task:beta",
        metadata={"priority": "high"},
    )

    results = router.query("controller parameters", top_k=2)
    assert results
    assert any("controller" in item["text"] for item in results)
    priority_entries = [item for item in results if item["metadata"].get("priority") == "high"]
    assert priority_entries, "metadata should be included in query results"
    assert router.stats()["total_entries"] == 2
    assert not consolidator.calls  # nothing promoted yet


def test_memory_router_promotes_after_usage_threshold():
    consolidator = StubConsolidator()
    router = MemoryRouter(consolidator)

    router.add_observation("reinforcement schedule update", source="task:gamma")
    # Simulate repeated access
    for _ in range(3):
        router.query("reinforcement schedule", top_k=1)

    promoted = router.review(usage_threshold=2)
    assert promoted, "entry should be promoted after sufficient usage"
    assert consolidator.calls
    promoted_entry = consolidator.calls[0]
    assert promoted_entry["source"] == "task:gamma"
    assert promoted_entry["metadata"]["promotion"] is True


def test_memory_router_shrink_removes_stale_entries():
    consolidator = StubConsolidator()
    router = MemoryRouter(consolidator)

    recent_id = router.add_observation("monitor critical subsystem", source="task:delta")
    stale_id = router.add_observation("archive obsolete logs", source="task:epsilon")

    router._entries[recent_id]["usage"] = 5
    router._entries[stale_id]["usage"] = 0
    router._entries[stale_id]["created_at"] = time.time() - 4000

    removed = router.shrink(max_entries=1, max_age=1800, min_usage=1)

    assert removed == 1
    assert stale_id not in router._entries
    results = router.query("critical subsystem", top_k=1)
    assert results and results[0]["id"] == recent_id
