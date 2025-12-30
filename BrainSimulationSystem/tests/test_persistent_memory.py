"""Tests for persistent memory manager."""

from __future__ import annotations

from BrainSimulationSystem.models.persistent_memory import PersistentMemoryManager


def test_persistent_memory_add_and_search(tmp_path):
    store_path = tmp_path / "memory.json"
    manager = PersistentMemoryManager(
        {
            "path": str(store_path),
            "embedding_dim": 32,
            "max_entries": 10,
            "working_memory_size": 4,
            "embedding_cache_path": str(tmp_path / "cache"),
        }
    )

    manager.add_memory("learned physics concept", {"tag": "physics"})
    manager.add_memory("cooking recipe", {"tag": "cooking"})

    results = manager.search("physics", top_k=1)
    assert results
    assert results[0]["metadata"].get("tag") == "physics"
    assert "id" in results[0]
    assert manager.working_items()


def test_persistent_memory_persistence(tmp_path):
    store_path = tmp_path / "memory.json"
    manager = PersistentMemoryManager({"path": str(store_path), "embedding_dim": 16})
    manager.add_memory("first entry")

    # reload from disk
    manager2 = PersistentMemoryManager({"path": str(store_path), "embedding_dim": 16})
    results = manager2.search("entry", top_k=1)
    assert results


def test_semantic_query_scores_paraphrase_higher(tmp_path):
    store_path = tmp_path / "memory.json"
    manager = PersistentMemoryManager(
        {
            "path": str(store_path),
            "embedding_dim": 64,
            "max_entries": 50,
            "working_memory_size": 4,
        }
    )

    manager.add_memory("The spacecraft launch was successful", {"source": "mission-log"})
    manager.add_memory("Gardening tips for spring", {"source": "blog"})

    paraphrased_query = "Successful rocket launch"
    unrelated_query = "How to bake bread"

    paraphrased_score = manager.search(paraphrased_query, top_k=1)[0]["score"]
    unrelated_score = manager.search(unrelated_query, top_k=1)[0]["score"]

    assert paraphrased_score >= unrelated_score
