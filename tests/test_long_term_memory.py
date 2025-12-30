from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pytest

from modules.knowledge import KnowledgeFact
from modules.knowledge.long_term_memory import LongTermMemoryCoordinator


class _StubImporter:
    def __init__(self) -> None:
        self.facts: List[KnowledgeFact] = []

    def ingest_facts(self, facts: Iterable[KnowledgeFact]) -> Dict[str, Any]:
        for fact in facts:
            self.facts.append(fact)
        return {"imported": len(self.facts)}


class _StubRepository:
    def __init__(self, relations: Dict[str, List[Dict[str, Any]]]) -> None:
        self._relations = relations

    def query(self, *, node_id: str | None = None, **_: Any) -> Dict[str, Any]:
        if not node_id:
            return {"edges": []}
        return {"edges": list(self._relations.get(node_id, []))}


@pytest.fixture()
def coordinator(tmp_path: Path) -> LongTermMemoryCoordinator:
    importer = _StubImporter()
    repo = _StubRepository({})
    return LongTermMemoryCoordinator(
        storage_root=tmp_path,
        knowledge_importer=importer,
        knowledge_repository=repo,
    )


def test_record_fact_updates_importer_and_vector_store(coordinator: LongTermMemoryCoordinator) -> None:
    fact = KnowledgeFact(subject="sky", predicate="color", obj="blue", metadata={"source": "observation"})
    summary = coordinator.record_fact(fact)

    assert summary.metadata["subject"] == "sky"
    assert summary.metadata["predicate"] == "color"
    assert summary.metadata["object"] == "blue"
    assert summary.vector_id is not None

    results = coordinator.query_similar("sky color", top_k=1)
    assert results
    assert results[0].metadata["subject"] == "sky"


def test_record_facts_batch_returns_summaries(coordinator: LongTermMemoryCoordinator) -> None:
    facts = [
        KnowledgeFact(subject="water", predicate="state", obj="liquid"),
        KnowledgeFact(subject="sun", predicate="emits", obj="light"),
    ]

    summaries = coordinator.record_facts(facts, base_metadata={"source": "simulation"})

    assert len(summaries) == 2
    assert all(summary.vector_id for summary in summaries)
    assert all(summary.metadata["source"] == "simulation" for summary in summaries)


def test_consolidate_episode_creates_summary(tmp_path: Path) -> None:
    coordinator = LongTermMemoryCoordinator(storage_root=tmp_path, knowledge_importer=_StubImporter())

    summary = coordinator.consolidate_episode(
        task_id="collect_samples",
        policy_version="v1",
        total_reward=12.5,
        steps=8,
        success=True,
        metadata={"environment": "lab"},
    )

    assert "collect_samples" in summary.text
    assert summary.metadata["environment"] == "lab"
    assert summary.vector_id is not None


def test_known_fact_queries_repository(tmp_path: Path) -> None:
    repo = _StubRepository(
        {
            "agent": [
                {"relation_type": "prefers", "target": "exploration"},
                {"relation_type": "learns", "target": "autonomously"},
            ]
        }
    )
    coordinator = LongTermMemoryCoordinator(
        storage_root=tmp_path, knowledge_importer=_StubImporter(), knowledge_repository=repo
    )

    assert coordinator.known_fact(subject="agent", predicate="prefers", obj="exploration") is True
    assert coordinator.known_fact(subject="agent", predicate="prefers", obj="stability") is False
    assert coordinator.known_fact(subject="agent", predicate="collaborates") is False


def test_consolidation_replays_hippocampal_traces(coordinator: LongTermMemoryCoordinator) -> None:
    fact = KnowledgeFact(subject="agent", predicate="learned", obj="new lexicon")

    coordinator.stage_interaction(
        "Learned a new way to describe self-updates.", facts=[fact], metadata={"context": "dialogue"}
    )
    coordinator.stage_interaction("Observed environment drift in sensors.", metadata={"context": "monitor"})

    summary = coordinator.consolidate_hippocampal_traces()

    assert summary is not None
    assert summary.metadata["trace_count"] == 2
    assert summary.vector_id is not None

    results = coordinator.query_similar("describe self-updates", top_k=5)
    assert any(record.metadata.get("stage") == "hippocampal" for record in results)
    assert len(coordinator._hippocampal_traces) == 0
    assert isinstance(coordinator._importer, _StubImporter)
    assert coordinator._importer.facts and coordinator._importer.facts[0].predicate == "learned"
