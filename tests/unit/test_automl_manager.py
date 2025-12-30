import asyncio
from importlib import util
from pathlib import Path
import sys
import types
from typing import Any, Callable, Dict, List


ROOT = Path(__file__).resolve().parents[2]
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)
execution_pkg = types.ModuleType("backend.execution")
execution_pkg.__path__ = [str(ROOT / "backend" / "execution")]
sys.modules.setdefault("backend.execution", execution_pkg)

MODULE_PATH = ROOT / "backend" / "execution" / "automl_manager.py"
spec = util.spec_from_file_location("backend.execution.automl_manager", MODULE_PATH)
module = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.automl_manager", module)
spec.loader.exec_module(module)

AutoMLManager = module.AutoMLManager


class SyncEventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], Any]]] = {}
        self.published: List[tuple[str, Dict[str, Any]]] = []

    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        self.published.append((topic, dict(event)))
        for handler in list(self._subscribers.get(topic, [])):
            result = handler(dict(event))
            if asyncio.iscoroutine(result):
                asyncio.run(result)

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], Any]) -> Callable[[], None]:
        self._subscribers.setdefault(topic, []).append(handler)
        return lambda: self.unsubscribe(topic, handler)

    def unsubscribe(self, topic: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        handlers = self._subscribers.get(topic, [])
        if handler in handlers:
            handlers.remove(handler)
        if not handlers and topic in self._subscribers:
            del self._subscribers[topic]


class InlineTaskManager:
    def __init__(self) -> None:
        self.submissions: List[Dict[str, Any]] = []

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> object:
        self.submissions.append({"func": func, "args": args, "kwargs": dict(kwargs)})
        func(*args)
        return object()


def test_automl_manager_emits_suggestion_and_accepts_feedback() -> None:
    bus = SyncEventBus()
    tasks = InlineTaskManager()
    manager = AutoMLManager(
        event_bus=bus,  # type: ignore[arg-type]
        task_manager=tasks,  # type: ignore[arg-type]
        enabled=True,
        backend="random",
        cooldown_secs=0.0,
        seed=123,
    )

    bus.publish("automl.request", {"metric": "decision_success_rate", "time": 1.0})
    suggestions = [event for topic, event in bus.published if topic == "automl.suggestion"]
    assert len(suggestions) == 1
    first = suggestions[0]
    suggestion_id = str(first.get("suggestion_id") or "")
    assert suggestion_id

    # In-flight suggestions block re-request until feedback is reported.
    bus.publish("automl.request", {"metric": "decision_success_rate", "time": 1.1})
    suggestions = [event for topic, event in bus.published if topic == "automl.suggestion"]
    assert len(suggestions) == 1

    bus.publish(
        "automl.feedback",
        {"suggestion_id": suggestion_id, "metric": "decision_success_rate", "objective_value": 0.5},
    )
    bus.publish("automl.request", {"metric": "decision_success_rate", "time": 2.0})

    suggestions = [event for topic, event in bus.published if topic == "automl.suggestion"]
    assert len(suggestions) == 2
    assert suggestions[-1]["suggestion_id"] != suggestion_id
    manager.close()

