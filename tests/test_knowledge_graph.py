import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.autogpt.autogpt.core.knowledge_graph.graph_store import GraphStore
from backend.autogpt.autogpt.core.knowledge_graph.ontology import (
    EntityType,
    RelationType,
)
from backend.autogpt.autogpt.core.knowledge_graph.reasoner import (
    infer_potential_relations,
)
from modules.common import ConceptNode

import types

capability_module = types.ModuleType("capability")
librarian_module = types.ModuleType("capability.librarian")


class Librarian:
    def search(self, *args, **kwargs):
        return []


librarian_module.Librarian = Librarian
capability_module.librarian = librarian_module
sys.modules["capability"] = capability_module
sys.modules["capability.librarian"] = librarian_module

from backend.concept_alignment import ConceptAligner  # noqa: E402


def test_dynamic_graph_and_version():
    store = GraphStore()
    store.add_node("a", EntityType.SKILL)
    store.add_node("b", EntityType.TASK)
    store.add_edge("a", "b", RelationType.RELATED_TO)
    version = store.get_version()
    snapshot = store.get_snapshot()
    assert len(snapshot["nodes"]) == 2
    assert version > 0
    store.remove_node("b")
    assert len(store.query()["nodes"]) == 1
    assert store.get_version() == version + 1


def test_infer_potential_relations():
    store = GraphStore()
    store.add_node("a", EntityType.SKILL)
    store.add_node("b", EntityType.TASK)
    store.add_node("c", EntityType.SKILL)
    store.add_edge("a", "b", RelationType.RELATED_TO)
    store.add_edge("b", "c", RelationType.RELATED_TO)
    suggestions = infer_potential_relations(store)
    assert ("a", "c", "related_to->related_to") in suggestions


def test_concept_alignment_distill_and_transfer():
    entities = {
        "x": ConceptNode(id="x", label="X", modalities={"text": [1.0, 0.0]})
    }
    aligner = ConceptAligner(Librarian(), entities)
    external = {
        "y": ConceptNode(id="y", label="Y", modalities={"text": [1.0, 0.0]})
    }
    aligner.distill_from_graph(external)
    assert "y" in aligner.entities
    nodes = [ConceptNode(id="z", label="Z", modalities={"text": [1.0, 0.0]})]
    enriched = aligner.transfer_knowledge(nodes)
    assert enriched[0].metadata.get("related_concepts") is not None
