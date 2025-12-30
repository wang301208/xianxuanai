import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.concept_alignment import ConceptAligner
from backend.knowledge.importer import BulkKnowledgeImporter
from backend.autogpt.autogpt.core.knowledge_graph.graph_store import GraphStore


class DummyIndex:
    def __init__(self) -> None:
        self.items = []

    def add(self, *args, **kwargs):
        self.items.append((args, kwargs))


class DummyLibrarian:
    def __init__(self) -> None:
        self.index = DummyIndex()

    def search(self, *args, **kwargs):
        return []


def test_bulk_importer_ingests_directory(tmp_path):
    nodes_csv = tmp_path / "nodes.csv"
    nodes_csv.write_text(
        "id,label,description\n"
        "agent1,Agent,A self-improving autonomous agent.\n"
        "env1,Environment,The operating environment.\n",
        encoding="utf-8",
    )
    relations_csv = tmp_path / "relations.csv"
    relations_csv.write_text(
        "source,relation,target,weight\n"
        "agent1,interacts_with,env1,0.75\n",
        encoding="utf-8",
    )

    aligner = ConceptAligner(DummyLibrarian(), {})
    store = GraphStore()
    importer = BulkKnowledgeImporter(aligner, graph_store=store)

    summary = importer.ingest_directory(tmp_path)

    assert summary["relations_added"] == 1
    assert "agent1" in aligner.entities
    assert "env1" in aligner.entities
    assert "text" in aligner.entities["agent1"].modalities
    graph_snapshot = store.query()
    node_ids = {node.id for node in graph_snapshot["nodes"]}
    assert {"agent1", "env1"}.issubset(node_ids)
    edge = graph_snapshot["edges"][0]
    assert edge.source == "agent1"
    assert edge.target == "env1"
    assert edge.properties.get("label") == "interacts_with"


def test_bulk_importer_ingests_single_csv(tmp_path):
    triples_path = tmp_path / "triples.csv"
    triples_path.write_text(
        "source,relation,target\n"
        "concept_a,related_to,concept_b\n",
        encoding="utf-8",
    )

    aligner = ConceptAligner(DummyLibrarian(), {})
    importer = BulkKnowledgeImporter(aligner, graph_store=GraphStore())

    summary = importer.ingest_file(triples_path)

    assert summary["relations_added"] == 1
    assert "concept_a" in aligner.entities
    assert "concept_b" in aligner.entities
    text_vec = aligner.entities["concept_a"].modalities.get("text")
    assert text_vec is not None and len(text_vec) > 0
