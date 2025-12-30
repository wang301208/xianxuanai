import asyncio
import sys
import types
from collections import deque
from importlib import util
from pathlib import Path
from typing import Any, Callable, Dict, List


ROOT = Path(__file__).resolve().parents[2]
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)
execution_pkg = types.ModuleType("backend.execution")
execution_pkg.__path__ = [str(ROOT / "backend" / "execution")]
sys.modules.setdefault("backend.execution", execution_pkg)

MODULE_PATH = ROOT / "backend" / "execution" / "self_diagnoser.py"
spec = util.spec_from_file_location("backend.execution.self_diagnoser", MODULE_PATH)
module = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.self_diagnoser", module)
spec.loader.exec_module(module)

SelfDiagnoser = module.SelfDiagnoser
SelfDiagnoserConfig = module.SelfDiagnoserConfig


class SyncEventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], Any]]] = {}
        self._queue: deque[tuple[str, Dict[str, Any]]] = deque()
        self._dispatching = False

    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        self._queue.append((topic, event))
        if self._dispatching:
            return
        self._dispatching = True
        try:
            while self._queue:
                queued_topic, queued_event = self._queue.popleft()
                for handler in list(self._subscribers.get(queued_topic, [])):
                    result = handler(queued_event)
                    if asyncio.iscoroutine(result):
                        asyncio.run(result)
        finally:
            self._dispatching = False

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], Any]) -> Callable[[], None]:
        self._subscribers.setdefault(topic, []).append(handler)
        return lambda: self.unsubscribe(topic, handler)

    def unsubscribe(self, topic: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        handlers = self._subscribers.get(topic, [])
        if handler in handlers:
            handlers.remove(handler)
        if not handlers and topic in self._subscribers:
            del self._subscribers[topic]


def test_self_diagnoser_emits_diagnosis_and_plan_for_task_failure() -> None:
    bus = SyncEventBus()
    diag_events: list[dict] = []
    plan_events: list[dict] = []
    learning_events: list[dict] = []
    bus.subscribe("diagnostics.self_diagnosis", lambda e: diag_events.append(dict(e)))
    bus.subscribe("planner.plan_ready", lambda e: plan_events.append(dict(e)))
    bus.subscribe("learning.request", lambda e: learning_events.append(dict(e)))

    def llm(_prompt: str) -> str:
        return (
            "{"
            "\"category\":\"dependency\","
            "\"confidence\":0.9,"
            "\"summary\":\"Missing dependency\","
            "\"recommendations\":[\"Install the missing package\"],"
            "\"actions\":["
            "{\"kind\":\"learning.request\",\"reason\":\"dependency\"},"
            "{\"kind\":\"planner.replan\",\"instruction\":\"Add dependency install step\"}"
            "]"
            "}"
        )

    mgr = SelfDiagnoser(  # type: ignore[arg-type]
        event_bus=bus,
        llm=llm,
        model="test",
        config=SelfDiagnoserConfig(enabled=True, cooldown_secs=0.0, publish_plan=True, publish_learning_request=True),
    )

    bus.publish(
        "task_manager.task_completed",
        {
            "time": 1.0,
            "status": "failed",
            "task_id": "t1",
            "name": "job",
            "category": "unit",
            "error": "No module named 'x'",
        },
    )

    assert diag_events and diag_events[-1]["category"] == "dependency"
    assert plan_events and plan_events[-1]["source"] == "self_diagnoser"
    assert learning_events and "self_diagnosis" in str(learning_events[-1].get("reason", ""))
    mgr.close()


def test_self_diagnoser_handles_answer_mismatch_event() -> None:
    bus = SyncEventBus()
    diag_events: list[dict] = []
    bus.subscribe("diagnostics.self_diagnosis", lambda e: diag_events.append(dict(e)))

    def llm(_prompt: str) -> str:
        return (
            "{"
            "\"category\":\"knowledge_gap\","
            "\"confidence\":0.7,"
            "\"summary\":\"Needs more domain knowledge\","
            "\"recommendations\":[\"Search docs\"],"
            "\"actions\":[{\"kind\":\"knowledge.update\",\"query\":\"topic\"}]"
            "}"
        )

    mgr = SelfDiagnoser(  # type: ignore[arg-type]
        event_bus=bus,
        llm=llm,
        model="test",
        config=SelfDiagnoserConfig(enabled=True, cooldown_secs=0.0, publish_plan=False, publish_learning_request=False),
    )

    bus.publish(
        "diagnostics.answer_mismatch",
        {"time": 1.0, "question": "Q", "answer": "A", "reference": "B"},
    )

    assert diag_events and diag_events[-1]["category"] == "knowledge_gap"
    mgr.close()

