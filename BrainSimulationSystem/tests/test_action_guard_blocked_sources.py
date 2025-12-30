"""ActionGuard blocks knowledge sources marked as untrusted/blocked in the KG."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from backend.autogpt.autogpt.core.knowledge_graph.ontology import EntityType  # noqa: E402
from modules.knowledge.action_guard import ActionGuard  # noqa: E402


class _Node:
    def __init__(self, node_id: str, properties: dict) -> None:
        self.id = node_id
        self.type = EntityType.CONCEPT
        self.properties = dict(properties)


class _Graph:
    def __init__(self, nodes) -> None:
        self._nodes = list(nodes)

    def get_snapshot(self):
        return {"nodes": list(self._nodes), "edges": []}


def test_action_guard_blocks_web_domain_source_node() -> None:
    guard = ActionGuard(
        graph_store=_Graph(
            [
                _Node(
                    "src:blocked:example",
                    {"source_kind": "web_domain", "domain": "example.com", "blocked": True, "reason": "bad_source"},
                )
            ]
        )
    )

    denied = guard.evaluate("web_scrape", {"url": "https://example.com/x"})
    assert denied.allowed is False
    assert "bad_source" in (denied.reason or "")

    allowed = guard.evaluate("web_scrape", {"url": "https://good.com/x"})
    assert allowed.allowed is True


def test_action_guard_blocks_github_repo_source_node() -> None:
    guard = ActionGuard(
        graph_store=_Graph(
            [
                _Node(
                    "src:blocked:repo",
                    {"source_kind": "github_repo", "repo": "evil/repo", "blocked": True},
                )
            ]
        )
    )

    denied = guard.evaluate("github_repo_ingest", {"repo": "https://github.com/evil/repo"})
    assert denied.allowed is False
    assert "evil/repo" in (denied.reason or "").lower()

