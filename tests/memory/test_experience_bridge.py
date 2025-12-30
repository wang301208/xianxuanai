from modules.learning import EpisodeRecord
from modules.memory.experience_bridge import (
    ExperienceMemoryBridge,
    curiosity_weighted_summarizer,
)
from modules.memory.lifecycle import MemoryLifecycleManager
from modules.memory.task_memory import ExperiencePayload
from modules.memory.vector_store import VectorMemoryStore


def _embedder(text: str):
    return [float((len(text) % 7) + 1)]


def test_experience_bridge_records_and_recalls(tmp_path):
    store = VectorMemoryStore(tmp_path / "vs", embedder=_embedder, backend="brute")
    lifecycle = MemoryLifecycleManager(
        store,
        summarizer=curiosity_weighted_summarizer,
        short_term_limit=3,
        working_memory_limit=3,
    )
    bridge = ExperienceMemoryBridge(lifecycle, consolidate_every=1)

    episode = EpisodeRecord(
        task_id="explore-cave",
        policy_version="v1",
        total_reward=4.0,
        steps=3,
        success=True,
        metadata={"novelty": 0.6},
    )
    bridge.record_episode(
        episode,
        metrics={"loss": 0.1},
        curiosity_samples=[{"metadata": {"novelty": 0.9}}],
    )

    results = bridge.recall("explore cave")
    assert results, "memory bridge should store queryable episodes"


def test_curiosity_summarizer_emphasises_high_importance():
    payload_low = ExperiencePayload(
        task_id="low",
        summary="Routine maintenance task.",
        metadata={"importance_hint": 0.2, "reward": 0.1},
    )
    payload_high = ExperiencePayload(
        task_id="high",
        summary="Recovered rare artifact with high novelty.",
        metadata={"importance_hint": 0.9, "reward": 5.0, "novelty": 0.8},
    )

    summary = curiosity_weighted_summarizer([payload_low, payload_high])

    assert summary["metadata"]["summary_strategy"] == "curiosity_weighted"
    assert "rare artifact" in summary["text"]
    assert summary["symbols"]["highlights"], "highlights should be populated"
