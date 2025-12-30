from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "third_party/autogpt"))

from third_party.autogpt.autogpt.core.brain.memory_manager import HierarchicalMemorySystem


def create_memory_system(**overrides) -> HierarchicalMemorySystem:
    params = {
        "working_capacity": 3,
        "episodic_limit": 4,
        "consolidation_importance": 0.6,
        "consolidation_window": 0.0,
        "consolidation_batch_size": 10,
        "long_term_path": ":memory:",
        "long_term_max_entries": 100,
        "decay_half_life": 1.0,
        "interference_penalty": 0.05,
        "semantic_limit": 3,
    }
    params.update(overrides)
    return HierarchicalMemorySystem(**params)


def test_hierarchical_memory_consolidates_and_retrieves():
    memory = create_memory_system()

    trace = memory.encode_experience(
        {"observation": "Design new feature"},
        modality="task",
        importance=0.9,
        tags=["planning"],
    )

    assert trace.trace_id  # trace is recorded

    results = memory.retrieve("Design new feature", sources=("long_term",))
    assert results
    assert results[0]["source"] == "long_term"
    assert "content" in results[0]

    memory.shutdown()


def test_memory_system_applies_decay_and_interference():
    memory = create_memory_system(episodic_limit=2, decay_half_life=0.1, interference_penalty=0.2)

    first = memory.encode_experience("Episode A", importance=0.5)
    second = memory.encode_experience("Episode B", importance=0.5)
    third = memory.encode_experience("Episode C", importance=0.5)

    assert len(memory._episodic_traces) == 2  # oldest trimmed
    assert first.importance < 0.5  # interference decreased the importance before eviction

    # Age out the oldest trace
    memory._episodic_traces[0].timestamp -= 10.0
    memory.apply_decay()
    assert all(trace.text != "Episode A" for trace in memory._episodic_traces)

    memory.shutdown()


def test_semantic_memory_limit_enforces_capacity():
    memory = create_memory_system(semantic_limit=2)

    memory.add_semantic_fact("fact1", "value1")
    memory.add_semantic_fact("fact2", "value2")
    memory.add_semantic_fact("fact3", "value3")

    retrieved = memory.retrieve(None, sources=("semantic",), limit=5)
    keys = {entry["id"] for entry in retrieved}

    assert "fact3" in keys
    assert len(keys) <= 2  # limit enforced
    assert "fact1" not in keys  # oldest fact evicted

    memory.shutdown()
