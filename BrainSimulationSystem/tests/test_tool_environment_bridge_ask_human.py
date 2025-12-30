"""Human-in-the-loop interaction tool for ToolEnvironmentBridge."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.environment.security_manager import SecurityManager  # noqa: E402
from BrainSimulationSystem.environment.tool_bridge import ToolEnvironmentBridge  # noqa: E402


def test_ask_human_round_trip_in_memory(tmp_path: Path) -> None:
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])

    obs, reward, terminated, info = bridge.step({"type": "ask_human", "question": "How to tune learning rate?"})
    assert terminated is False
    assert reward > 0
    assert info.get("status") == "pending"
    assert info.get("requires_human") is True
    request_id = info.get("request_id")
    assert isinstance(request_id, str) and request_id

    # Poll before response.
    obs, reward, terminated, info = bridge.step({"type": "ask_human", "request_id": request_id})
    assert terminated is False
    assert reward > 0
    assert info.get("status") == "pending"

    # Provide answer.
    obs, reward, terminated, info = bridge.step(
        {"type": "ask_human", "request_id": request_id, "answer": "Try 1e-3 and decay on plateau."}
    )
    assert terminated is False
    assert reward > 0
    assert info.get("status") == "answered"

    # Poll after response.
    obs, reward, terminated, info = bridge.step({"type": "ask_human", "request_id": request_id})
    assert terminated is False
    assert reward > 0
    assert info.get("status") == "answered"
    request = obs.get("request")
    assert isinstance(request, dict)
    assert "decay" in str(request.get("answer") or "")


def test_ask_human_simulates_answer_from_jsonl_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "qa.jsonl"
    records = [
        {"question": "Dijkstra shortest path in graphs with non-negative weights", "answer": "It finds shortest paths in graphs with non-negative edge weights."},
        {"question": "How to parse python code?", "answer": "Use the ast module: ast.parse(source)."},
    ]
    dataset.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")

    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path])
    obs, reward, terminated, info = bridge.step(
        {
            "type": "ask_human",
            "question": "shortest path in a graph, non negative weights?",
            "dataset_path": str(dataset),
        }
    )
    assert terminated is False
    assert reward > 0
    assert info.get("simulated") is True
    assert info.get("status") == "answered"
    assert "shortest" in str(obs.get("answer") or "").lower()


def test_security_manager_blocks_ask_human_in_read_only_mode(tmp_path: Path) -> None:
    security = SecurityManager({"enabled": True, "permission_level": "read"})
    bridge = ToolEnvironmentBridge(allowed_roots=[tmp_path], security_manager=security)

    obs, reward, terminated, info = bridge.step({"type": "ask_human", "question": "hello"})
    assert terminated is False
    assert reward < 0
    assert info.get("blocked") is True
    assert info.get("reason") == "permission_denied"
