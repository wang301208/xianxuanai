import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repository root on path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[3]))
from backend.memory.long_term import LongTermMemory


def test_tag_filtering(tmp_path):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    try:
        mem.add("news", "item1", tags=["urgent", "important"])
        mem.add("news", "item2", tags=["routine"])
        assert list(mem.get(tags=["urgent"])) == ["item1"]
    finally:
        mem.close()


def test_time_range_query(tmp_path):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    now = datetime.utcnow().timestamp()
    try:
        mem.add("news", "old", timestamp=now - 10)
        mem.add("news", "current", timestamp=now)
        mem.add("news", "new", timestamp=now + 10)
        results = list(mem.get(start_ts=now - 1, end_ts=now + 1))
        assert results == ["current"]
    finally:
        mem.close()


def test_large_dataset_performance(tmp_path):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    try:
        for i in range(1000):
            tag = "target" if i == 999 else "other"
            mem.add("cat", f"content {i}", tags=[tag])
        start = time.time()
        result = list(mem.get(tags=["target"]))
        duration = time.time() - start
        assert result == ["content 999"]
        assert duration < 1.0
    finally:
        mem.close()


def test_embedding_storage(tmp_path):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    try:
        mem.add_embedding("key", [0.1, 0.2, 0.3], {"tag": "test"})
        stored = mem.get_embedding("key")
        assert stored is not None
        vector, meta = stored
        assert vector == [0.1, 0.2, 0.3]
        assert meta == {"tag": "test"}
    finally:
        mem.close()


def test_embedding_similarity_search(tmp_path):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    try:
        apples = mem.add("facts", "Apples are red", tags=["fruit", "red"], metadata={"source": "note"})
        bananas = mem.add("facts", "Bananas are yellow", tags=["fruit", "yellow"], metadata={"source": "note"})

        mem.add_embedding(
            str(apples), [1.0, 0.0], {"memory_id": apples, "category": "facts", "tags": ["fruit", "red"]}
        )
        mem.add_embedding(
            str(bananas), [0.0, 1.0], {"memory_id": bananas, "category": "facts", "tags": ["fruit", "yellow"]}
        )

        results = mem.similarity_search([0.9, 0.1], top_k=2)
        assert results and results[0]["memory_id"] == apples
        assert results[0]["memory"]["content"] == "Apples are red"

        yellow_only = mem.similarity_search([1.0, 0.0], top_k=5, tags=["yellow"])
        assert yellow_only and yellow_only[0]["memory_id"] == bananas

        category_filtered = mem.similarity_search([0.0, 1.0], category="facts")
        assert all(hit.get("memory", {}).get("category") == "facts" for hit in category_filtered)
    finally:
        mem.close()


def test_additional_metadata_and_filters(tmp_path):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    try:
        entry_id = mem.add(
            "facts",
            "gravity ~9.8",
            tags=["physics"],
            confidence=0.85,
            status="verified",
            metadata={"source": "textbook"},
        )
        results = list(
            mem.get(
                status=["verified"],
                include_metadata=True,
                min_confidence=0.5,
            )
        )
        assert len(results) == 1
        record = results[0]
        assert record["id"] == entry_id
        assert record["confidence"] == 0.85
        assert record["status"] == "verified"
        assert record["metadata"] == {"source": "textbook"}
        assert record["tags"] == ["physics"]
    finally:
        mem.close()


def test_update_and_mark_entries(tmp_path):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db)
    try:
        entry_id = mem.add("facts", "pending datum")
        mem.update_entry(entry_id, confidence=0.4, status="review", tags=["pending"])
        updated = list(mem.get(status=["review"], include_metadata=True))
        assert updated and updated[0]["confidence"] == 0.4
        assert updated[0]["tags"] == ["pending"]
        mem.mark_entries([entry_id], status="stale", confidence=0.1)
        stale = list(mem.get(status=["stale"], include_metadata=True))
        assert stale and stale[0]["confidence"] == 0.1
    finally:
        mem.close()


def test_priority_forgetting_strategy(tmp_path):
    db = tmp_path / "mem.db"
    mem = LongTermMemory(db, forget_interval=1)
    mem.configure_priority_forgetting(
        stale_statuses=("stale",),
        confidence_threshold=0.3,
        max_removed_per_cycle=5,
    )
    try:
        mem.add("facts", "keep this", confidence=0.9)
        mem.add("facts", "remove stale", status="stale")
        remaining = [r["content"] for r in mem.get(include_metadata=True)]
        assert "remove stale" not in remaining
        assert "keep this" in remaining
        mem.add("facts", "remove low confidence", confidence=0.1)
        remaining = [r["content"] for r in mem.get(include_metadata=True)]
        assert "remove low confidence" not in remaining
        assert "keep this" in remaining
    finally:
        mem.close()
