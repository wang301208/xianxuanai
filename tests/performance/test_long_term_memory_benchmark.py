import random
from typing import List

import pytest

pytest.importorskip("pytest_benchmark")

from backend.memory.long_term import LongTermMemory


@pytest.fixture()
def memory_store(tmp_path) -> LongTermMemory:
    db_path = tmp_path / "memory.db"
    store = LongTermMemory(db_path, max_entries=None)
    with store.batch():
        for idx in range(500):
            tags = [f"tag{idx % 5}", f"grp{idx % 3}"]
            store.add(
                "general",
                f"content-{idx}",
                tags=tags,
                timestamp=1_700_000_000 + idx,
            )
    return store


def test_memory_get_latest_benchmark(benchmark, memory_store):
    def fetch_latest() -> List[str]:
        return list(memory_store.get("general", newest_first=True, limit=20))

    results = benchmark(fetch_latest)
    assert len(results) == 20
    assert results[0].startswith("content-")


def test_memory_tag_filter_benchmark(benchmark, memory_store):
    def fetch_with_tags() -> List[str]:
        tag = random.choice(["tag0", "tag1", "tag2", "tag3", "tag4"])
        return list(memory_store.get("general", tags=[tag], newest_first=True, limit=30))

    results = benchmark(fetch_with_tags)
    assert 0 < len(results) <= 30
