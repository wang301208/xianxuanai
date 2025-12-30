import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.knowledge.consolidation import KnowledgeConsolidator
from backend.autogpt.autogpt.core.knowledge_graph.graph_store import get_graph_store
from backend.autogpt.autogpt.core.knowledge_graph.ontology import EntityType


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.01)


def test_consolidator_merges_similar_statements():
    consolidator = KnowledgeConsolidator()
    consolidator.record_statement(
        "Autonomous agents learn from feedback loops.",
        source="task:alpha",
    )
    consolidator.wait_idle()
    concepts = list(consolidator.concepts)
    assert len(concepts) == 1
    concept = concepts[0]
    assert concept.metadata["occurrences"] == 1
    assert concept.metadata["sources"] == ["task:alpha"]

    consolidator.record_statement(
        "Autonomous agents learn from feedback loops.",
        source="task:beta",
    )
    consolidator.wait_idle()
    concepts = list(consolidator.concepts)
    assert len(concepts) == 1
    concept = concepts[0]
    assert concept.metadata["occurrences"] == 2
    assert "task:beta" in concept.metadata["sources"]

    store = get_graph_store()
    nodes = store.query()["nodes"]
    assert any(node.id == concept.id and node.type == EntityType.CONCEPT for node in nodes)


def test_consolidator_creates_new_concept_for_distinct_statement():
    consolidator = KnowledgeConsolidator()
    consolidator.record_statement(
        "Large language models require careful prompt design.",
        source="task:prompting",
    )
    consolidator.record_statement(
        "Photosynthesis converts light into chemical energy in plants.",
        source="task:biology",
    )
    consolidator.wait_idle()
    concepts = list(consolidator.concepts)
    assert len(concepts) == 2
    sources = sorted(sorted(node.metadata["sources"])[0] for node in concepts)
    assert sources == ["task:biology", "task:prompting"]


def test_hot_cold_rotation_and_controls():
    consolidator = KnowledgeConsolidator(hot_limit=1)
    consolidator.record_statement(
        "Transformer models benefit from long-context memories.",
        source="task:first",
    )
    consolidator.wait_idle()
    consolidator.record_statement(
        "Reinforcement systems adjust policies based on reward signals.",
        source="task:second",
    )
    consolidator.wait_idle()

    concepts = list(consolidator.concepts)
    ids_by_source = {}
    for concept in concepts:
        for src in concept.metadata["sources"]:
            ids_by_source[src] = concept.id

    first_id = ids_by_source["task:first"]
    second_id = ids_by_source["task:second"]

    assert consolidator.hot_concepts == [second_id]

    consolidator.demote_concept(second_id)
    assert second_id not in consolidator.hot_concepts

    consolidator.promote_concept(first_id)
    assert consolidator.hot_concepts == [first_id]
