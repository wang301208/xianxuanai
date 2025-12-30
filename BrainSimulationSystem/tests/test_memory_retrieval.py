"Tests for automatic memory retrieval advisor."

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.models.memory_retrieval import MemoryRetrievalAdvisor


def test_memory_retrieval_advisor_builds_semantic_query():
    advisor = MemoryRetrievalAdvisor({"min_query_length": 3})
    query = advisor.build_query(
        goals=["diagnose power failure"],
        planner={"candidates": [{"action": "check_fuses"}]},
        dialogue_state={"topics": ["safety"], "entities": ["Generator A"]},
        summary="Investigate generator alarm",
    )

    assert query is not None
    assert query["memory_type"] == "SEMANTIC"
    assert query["query"]["keywords"]
    assert "goal" in query["context"]["tags"]
