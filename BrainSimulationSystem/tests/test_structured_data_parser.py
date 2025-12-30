"""Tests for the structured data parser and ingestion helpers."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.models.knowledge_graph import KnowledgeGraph
from BrainSimulationSystem.models.structured_data_parser import StructuredDataParser


def test_structured_parser_generates_triples_from_records():
    parser = StructuredDataParser({"subject_field": "entity"})
    rows = [
        {"entity": "scene:1", "label": "tree", "count": 2},
        {"entity": "scene:2", "label": "rock", "count": 1},
    ]

    batch = parser.parse(rows, source={"name": "environment"})

    assert len(batch.triples) >= 4
    assert "summary" in batch.embeddings
    assert batch.metadata["rows"] == 2
    assert "label" in batch.metadata["columns"]


def test_structured_parser_ingest_updates_graph_metadata():
    parser = StructuredDataParser({"subject_field": "id"})
    graph = KnowledgeGraph()
    rows = [
        {"id": "row-1", "label": "tree", "score": 0.92},
    ]

    batch = parser.ingest(graph, rows, source={"name": "telemetry"})

    assert graph.exists("row-1", "label", "tree")
    metadata = graph.get_metadata("row-1", "label", "tree")
    assert metadata.get("provenance") == "telemetry"
    assert metadata.get("rows") == batch.metadata["rows"]
