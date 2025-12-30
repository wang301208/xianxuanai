"""Knowledge-graph import/query actions for ToolEnvironmentBridge."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.environment.security_manager import SecurityManager  # noqa: E402
from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge  # noqa: E402
from backend.autogpt.autogpt.core.knowledge_graph.graph_store import GraphStore  # noqa: E402
from backend.concept_alignment import ConceptAligner  # noqa: E402
from backend.knowledge.registry import set_default_aligner, set_graph_store  # noqa: E402
from capability.librarian import Librarian  # noqa: E402


def _reset_graph_state() -> None:
    set_graph_store(GraphStore())
    set_default_aligner(ConceptAligner(librarian=Librarian(), entities={}))


def test_knowledge_import_directory_then_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.knowledge.importer as importer_mod

    monkeypatch.setattr(importer_mod, "SentenceTransformer", None)
    _reset_graph_state()

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "dijkstra.md").write_text("# Dijkstra\n\nShortest path algorithm.\n", encoding="utf-8")

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    obs, reward, terminated, info = bridge.step(
        {"type": "knowledge_import_directory", "path": str(corpus), "source_name": "algo_pack"}
    )
    assert terminated is False
    assert reward > 0
    assert info.get("processed_files") == 1

    obs, reward, terminated, info = bridge.step(
        {"type": "knowledge_query", "query": "shortest path algorithm", "top_k": 5}
    )
    assert terminated is False
    assert reward > 0
    assert info.get("returned") >= 1

    results = obs.get("results")
    assert isinstance(results, list)
    assert any(
        isinstance(hit, dict)
        and isinstance(hit.get("metadata"), dict)
        and hit["metadata"].get("relative_path") == "dijkstra.md"
        for hit in results
    )

    refs = obs.get("references")
    assert isinstance(refs, list)
    assert any(isinstance(ref, dict) and ref.get("url") == "dijkstra.md" for ref in refs)


def test_security_manager_allows_knowledge_actions_at_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.knowledge.importer as importer_mod

    monkeypatch.setattr(importer_mod, "SentenceTransformer", None)
    _reset_graph_state()

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text("Quick note about BFS.", encoding="utf-8")

    security = SecurityManager({"enabled": True, "permission_level": "read"})
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], security_manager=security)

    obs, reward, terminated, info = bridge.step(
        {"type": "knowledge_import_directory", "path": str(corpus), "source_name": "notes"}
    )
    assert terminated is False
    assert reward > 0
    assert info.get("blocked") is not True

    obs, reward, terminated, info = bridge.step({"type": "knowledge_query", "query": "bfs", "top_k": 3})
    assert terminated is False
    assert reward > 0
    assert info.get("blocked") is not True

