from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parents[2]
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)
execution_pkg = types.ModuleType("backend.execution")
execution_pkg.__path__ = [str(ROOT / "backend" / "execution")]
sys.modules.setdefault("backend.execution", execution_pkg)

MODULE_PATH = ROOT / "backend" / "execution" / "self_reflection.py"
spec = util.spec_from_file_location("backend.execution.self_reflection", MODULE_PATH)
self_reflection = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.self_reflection", self_reflection)
spec.loader.exec_module(self_reflection)

SelfReflectionConfig = self_reflection.SelfReflectionConfig
SelfReflectionLoop = self_reflection.SelfReflectionLoop


class _StubBus:
    def __init__(self) -> None:
        self.published: List[Dict[str, Any]] = []

    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        self.published.append({"topic": topic, "event": dict(event)})


class _StubRouter:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []
        self._counter = 0

    def add_observation(self, text: str, *, source: str, metadata: Mapping[str, Any] | None = None, promote: bool = False) -> str:  # noqa: E501
        _ = promote
        self._counter += 1
        entry_id = f"mem-{self._counter}"
        self.entries.append({"id": entry_id, "text": text, "source": source, "metadata": dict(metadata or {})})
        return entry_id

    def query(self, text: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
        _ = text
        results: List[Dict[str, Any]] = []
        for entry in list(self.entries)[: max(1, int(top_k))]:
            results.append(
                {
                    "id": entry["id"],
                    "text": entry["text"],
                    "source": entry["source"],
                    "metadata": dict(entry.get("metadata") or {}),
                    "similarity": 1.0,
                }
            )
        return results


def test_self_reflection_loop_stores_reflection_and_emits_event() -> None:
    bus = _StubBus()
    router = _StubRouter()

    def llm_stub(prompt: str) -> str:
        assert "自我反思模块" in prompt
        return (
            '{'
            '"reflection":"需要补齐输入校验并减少假设。",'
            '"lessons":["先校验输入","记录失败上下文"],'
            '"checklist":["检查输入是否为空"],'
            '"tags":["data_validation"],'
            '"confidence":0.4'
            '}'
        )

    loop = SelfReflectionLoop(
        event_bus=bus,  # type: ignore[arg-type]
        memory_router=router,  # type: ignore[arg-type]
        llm=llm_stub,
        config=SelfReflectionConfig(
            enabled=True,
            cooldown_secs=0.0,
            min_event_chars=0,
            retrieve_min_similarity=0.0,
        ),
    )

    record = loop.reflect_task(
        {
            "time": 123.0,
            "task_id": "t-1",
            "agent_id": "a-1",
            "status": "completed",
            "summary": "生成报告",
            "detail": "输入为空导致处理失败",
        },
        trace=[{"type": "knowledge_blindspot", "timestamp": 120.0}],
    )
    assert record is not None
    assert record["used_llm"] is True
    assert record.get("memory_id") == "mem-1"

    assert router.entries
    assert router.entries[0]["metadata"]["category"] == "autobiographical_reflection"

    topics = [item["topic"] for item in bus.published]
    assert "diagnostics.self_reflection" in topics


def test_self_reflection_loop_retrieves_hints() -> None:
    bus = _StubBus()
    router = _StubRouter()
    loop = SelfReflectionLoop(
        event_bus=bus,  # type: ignore[arg-type]
        memory_router=router,  # type: ignore[arg-type]
        llm=None,
        config=SelfReflectionConfig(
            enabled=True,
            cooldown_secs=0.0,
            min_event_chars=0,
            retrieve_top_k=3,
            retrieve_min_similarity=0.0,
        ),
    )

    loop.reflect_task(
        {
            "time": 1.0,
            "task_id": "t-2",
            "status": "failed",
            "summary": "处理传感器数据",
            "detail": "未检查数据有效性",
        },
        trace=[],
    )

    hints = loop.retrieve_hints("传感器 数据 校验")
    assert hints
    assert hints[0]["metadata"]["category"] == "autobiographical_reflection"
