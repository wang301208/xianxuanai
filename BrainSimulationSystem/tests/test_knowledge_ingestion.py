"""Tests for the knowledge ingestion manager."""

from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.knowledge.ingestion import KnowledgeIngestionManager
from BrainSimulationSystem.models.knowledge_graph import KnowledgeGraph


def test_topic_request_generates_triples():
    manager = KnowledgeIngestionManager(
        {
            "refresh_interval": 0.0,
            "topic_interval": 0.0,
            "topic_template": {"tags": ["auto"]},
        }
    )
    graph = KnowledgeGraph()
    manager.request_topic("fusion_drive", metadata={"summary": "Investigate fusion drive status"})

    summary = manager.tick(time.time(), graph)

    assert summary["added"] > 0
    assert ("fusion_drive", "summary", "Investigate fusion drive status") in graph.triples
