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

MODULE_PATH = ROOT / "backend" / "execution" / "self_correction_manager.py"
spec = util.spec_from_file_location("backend.execution.self_correction_manager", MODULE_PATH)
module = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.self_correction_manager", module)
spec.loader.exec_module(module)

SelfCorrectionManager = module.SelfCorrectionManager
CorrectionConfig = module.CorrectionConfig


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


class DummyMemoryRouter:
    def __init__(self) -> None:
        self.observations: list[tuple[str, str, dict]] = []

    def add_observation(self, text: str, *, source: str, metadata: dict) -> None:
        self.observations.append((text, source, dict(metadata)))


def test_self_correction_emits_plan_and_learning_request(tmp_path: Path) -> None:
    # Prepare a doc file that the docs-search can find.
    doc = tmp_path / "docs.md"
    doc.write_text("PermissionError: access denied", encoding="utf-8")

    bus = SyncEventBus()
    memory = DummyMemoryRouter()
    correction_events: list[dict] = []
    plan_events: list[dict] = []
    learning_events: list[dict] = []

    bus.subscribe("diagnostics.self_correction", lambda e: correction_events.append(dict(e)))
    bus.subscribe("planner.plan_ready", lambda e: plan_events.append(dict(e)))
    bus.subscribe("learning.request", lambda e: learning_events.append(dict(e)))

    mgr = SelfCorrectionManager(  # type: ignore[arg-type]
        event_bus=bus,
        memory_router=memory,
        workspace_root=str(tmp_path),
        docs_roots=(str(tmp_path),),
        config=CorrectionConfig(
            enabled=True,
            cooldown_secs=0.0,
            publish_plan=True,
            publish_learning_request=True,
            docs_search=True,
            web_search=False,
            max_doc_hits=2,
            max_subquestions=3,
            add_capability_hints=True,
        ),
    )

    bus.publish(
        "diagnostics.self_debug",
        {
            "time": 1.0,
            "trigger": "failure_burst",
            "last_failure": {"reason": "PermissionError: access denied", "action": "write_file"},
            "evidence": [{"time": 1.0, "source": "action_outcome", "reason": "PermissionError"}],
        },
    )

    assert correction_events
    assert plan_events and plan_events[-1]["source"] == "self_correction"
    assert learning_events and "self_correction" in str(learning_events[-1].get("reason", ""))
    assert memory.observations
    assert correction_events[-1].get("doc_hits")
    assert "capabilities:" in str(plan_events[-1].get("goal", ""))

    mgr.close()


def test_self_correction_handles_task_manager_failures() -> None:
    bus = SyncEventBus()
    correction_events: list[dict] = []
    bus.subscribe("diagnostics.self_correction", lambda e: correction_events.append(dict(e)))

    mgr = SelfCorrectionManager(  # type: ignore[arg-type]
        event_bus=bus,
        config=CorrectionConfig(enabled=True, cooldown_secs=0.0, docs_search=False, publish_learning_request=False),
    )

    bus.publish(
        "task_manager.task_completed",
        {"time": 1.0, "status": "failed", "task_id": "t1", "name": "job", "error": "boom"},
    )

    assert correction_events and correction_events[-1]["trigger"] == "task_failure"
    mgr.close()

