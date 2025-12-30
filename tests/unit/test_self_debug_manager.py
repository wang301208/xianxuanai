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

MODULE_PATH = ROOT / "backend" / "execution" / "self_debug_manager.py"
spec = util.spec_from_file_location("backend.execution.self_debug_manager", MODULE_PATH)
self_debug_module = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.self_debug_manager", self_debug_module)
spec.loader.exec_module(self_debug_module)
SelfDebugManager = self_debug_module.SelfDebugManager


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


def test_self_debug_triggers_on_failure_burst() -> None:
    bus = SyncEventBus()
    plan_events: list[dict] = []
    debug_events: list[dict] = []
    bus.subscribe("planner.plan_ready", lambda e: plan_events.append(dict(e)))
    bus.subscribe("diagnostics.self_debug", lambda e: debug_events.append(dict(e)))

    mgr = SelfDebugManager(  # type: ignore[arg-type]
        event_bus=bus,
        enabled=True,
        window_secs=60.0,
        cooldown_secs=999.0,
        max_failures_window=2,
        max_replans_window=0,
        max_same_action_failures=0,
        max_bad_plans_window=0,
    )

    bus.publish("agent.action.outcome", {"status": "error", "command": "a", "time": 1.0})
    assert not debug_events

    bus.publish("agent.action.outcome", {"status": "error", "command": "b", "time": 2.0})
    assert debug_events and debug_events[-1]["trigger"] == "failure_burst"
    assert plan_events and plan_events[-1]["source"] == "self_debug"

    mgr.close()


def test_self_debug_triggers_on_action_loop() -> None:
    bus = SyncEventBus()
    debug_events: list[dict] = []
    bus.subscribe("diagnostics.self_debug", lambda e: debug_events.append(dict(e)))

    mgr = SelfDebugManager(  # type: ignore[arg-type]
        event_bus=bus,
        enabled=True,
        window_secs=60.0,
        cooldown_secs=0.0,
        max_failures_window=0,
        max_replans_window=0,
        max_same_action_failures=3,
        max_bad_plans_window=0,
    )

    for t in (1.0, 2.0, 3.0):
        bus.publish("agent.action.outcome", {"status": "error", "command": "repeat", "time": t})

    assert any(ev.get("trigger") == "action_loop" for ev in debug_events)
    mgr.close()


def test_self_debug_triggers_on_replan_storm() -> None:
    bus = SyncEventBus()
    debug_events: list[dict] = []
    bus.subscribe("diagnostics.self_debug", lambda e: debug_events.append(dict(e)))

    mgr = SelfDebugManager(  # type: ignore[arg-type]
        event_bus=bus,
        enabled=True,
        window_secs=60.0,
        cooldown_secs=0.0,
        max_failures_window=0,
        max_replans_window=2,
        max_same_action_failures=0,
        max_bad_plans_window=0,
    )

    bus.publish("agent.conductor.directive", {"directive": {"requires_replan": True}, "time": 1.0})
    bus.publish("agent.conductor.directive", {"directive": {"requires_replan": True}, "time": 2.0})

    assert any(ev.get("trigger") == "replan_storm" for ev in debug_events)
    mgr.close()


def test_self_debug_triggers_on_bad_plan_burst() -> None:
    bus = SyncEventBus()
    debug_events: list[dict] = []
    bus.subscribe("diagnostics.self_debug", lambda e: debug_events.append(dict(e)))

    mgr = SelfDebugManager(  # type: ignore[arg-type]
        event_bus=bus,
        enabled=True,
        window_secs=60.0,
        cooldown_secs=0.0,
        max_failures_window=0,
        max_replans_window=0,
        max_same_action_failures=0,
        max_bad_plans_window=1,
    )

    # many duplicates -> tasks_many_duplicates
    bus.publish("planner.plan_ready", {"goal": "x", "tasks": ["a", "a", "a"], "source": "test"})

    assert any(ev.get("trigger") == "bad_plan_burst" for ev in debug_events)
    mgr.close()
