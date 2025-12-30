from __future__ import annotations

import json
import time
from pathlib import Path

from modules.knowledge.knowledge_base import KnowledgeBase
from modules.learning.agent_reflector import AgentReflector, build_reflection_callback


def test_agent_reflector_persists_self_monitoring_record(tmp_path: Path) -> None:
    kb = KnowledgeBase(db_path=tmp_path / "memory.db", enabled=True, embedding_enabled=False)
    reflector = AgentReflector(knowledge_base=kb, enabled=True)

    payload = {
        "episodes": [
            {
                "task_id": "t1",
                "policy_version": "v1",
                "total_reward": 1.0,
                "steps": 4,
                "success": True,
                "metadata": {},
                "trajectory_path": None,
                "created_at": "now",
            }
        ],
        "metrics_summary": {"avg_latency": 0.2},
        "issues": [{"kind": "high_latency", "metric": "latency", "value": 0.2, "threshold": 0.1, "module": "m"}],
        "regressions": [{"kind": "regression", "detail": "x"}],
        "strategy_suggestions": {
            "updates": {"policy_exploration_rate": 0.2},
            "actions": [{"parameter": "policy_exploration_rate", "value": 0.2, "reason": "low_success_rate"}],
        },
    }

    record = reflector.reflect(payload)
    assert record.get("memory_id") is not None
    assert isinstance(record.get("evaluation"), dict)
    assert "summary" in record and "revision" in record

    stored = kb.recent(limit=5, category="self_monitoring")
    assert stored
    decoded = json.loads(stored[0].content)
    assert decoded["summary"]
    assert decoded["revision"]


def test_build_reflection_callback_returns_true(tmp_path: Path) -> None:
    kb = KnowledgeBase(db_path=tmp_path / "memory.db", enabled=True, embedding_enabled=False)
    reflector = AgentReflector(knowledge_base=kb, enabled=True)
    cb = build_reflection_callback(reflector)
    assert cb({"timestamp": time.time()}) is True

