import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.memory.maintenance import MemoryDecayPolicy, MemoryMaintenanceDaemon
from modules.memory.vector_store import VectorMemoryStore


def _simple_embedder(text: str) -> list[float]:
    length = float(len(text) or 1)
    vowels = sum(1 for char in text.lower() if char in "aeiou") or 1
    consonants = max(1.0, length - vowels)
    checksum = float(sum(ord(char) for char in text) % 17 or 1)
    vector = [length, float(vowels), float(consonants), checksum]
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


def test_episodic_consolidation(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    store = VectorMemoryStore(store_path, embedder=_simple_embedder, backend="brute")

    episode_log = tmp_path / "episodes.jsonl"
    episodes = []
    for idx in range(5):
        episodes.append(
            {
                "task_id": "perception:agent",
                "policy_version": "semantic_bridge",
                "total_reward": 0.5 + idx,
                "steps": 3 + idx,
                "success": bool(idx % 2 == 0),
                "created_at": f"2024-01-01T00:0{idx}:00Z",
                "metadata": {
                    "summary": f"vision: item-{idx}; audio: cue-{idx}",
                    "cycle": idx,
                },
            }
        )
    with episode_log.open("w", encoding="utf-8") as handle:
        for record in episodes:
            handle.write(json.dumps(record) + "\n")

    policy = MemoryDecayPolicy(
        episodic_summary_interval=0.0,
        episodic_summary_min_records=3,
        episodic_summary_window=4,
        vector_optimization_interval=10_000.0,
    )
    daemon = MemoryMaintenanceDaemon(
        store,
        policy=policy,
        episode_log_path=episode_log,
    )

    events = daemon.tick()

    assert "episodic_summaries" in events
    summary_ids = events["episodic_summaries"]
    assert len(summary_ids) == 1
    assert len(store) == 1
    stored = next(iter(store.iter_records()))
    metadata = stored["metadata"]
    assert metadata["type"] == "episodic_digest"
    assert metadata["episode_count"] == 4
    assert "key_facts" in metadata


def test_vector_optimisation_archives_redundant(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    store = VectorMemoryStore(store_path, embedder=_simple_embedder, backend="brute")

    first_id = store.add_text("alpha event", metadata={"importance": 0.9})
    second_id = store.add_text("alpha event", metadata={"importance": 0.2})
    novel_id = store.add_text("beta unique", metadata={"importance": 0.3})

    policy = MemoryDecayPolicy(
        vector_optimization_interval=0.0,
        vector_redundancy_similarity=0.99,
        vector_novelty_top_k=2,
        episodic_summary_interval=10_000.0,
    )
    daemon = MemoryMaintenanceDaemon(store, policy=policy)

    events = daemon.tick()

    assert "vector_optimization" in events
    optimisation = events["vector_optimization"]
    assert second_id in events.get("archived", [])
    assert second_id in optimisation["archived"]
    remaining_ids = {record["id"] for record in store.iter_records()}
    assert second_id not in remaining_ids
    assert any(item["id"] == novel_id for item in optimisation["novel"])
    assert all(record["metadata"].get("novelty_score") is not None for record in store.iter_records())
