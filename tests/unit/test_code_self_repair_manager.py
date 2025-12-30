import asyncio
from importlib import util
from pathlib import Path
import sys
import types
from typing import Any, Callable, Dict, List

import pytest


ROOT = Path(__file__).resolve().parents[2]
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)
execution_pkg = types.ModuleType("backend.execution")
execution_pkg.__path__ = [str(ROOT / "backend" / "execution")]
sys.modules.setdefault("backend.execution", execution_pkg)

MODULE_PATH = ROOT / "backend" / "execution" / "code_self_repair.py"
spec = util.spec_from_file_location("backend.execution.code_self_repair", MODULE_PATH)
module = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.code_self_repair", module)
spec.loader.exec_module(module)

CodeSelfRepairManager = module.CodeSelfRepairManager
PatchExecutor = module.PatchExecutor
PatchValidationResult = module.PatchValidationResult


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


class DummyExecutor(PatchExecutor):
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def validate(self, proposal: Any) -> Any:
        self.calls.append(proposal)
        return PatchValidationResult(
            proposal_id=proposal.proposal_id,
            metric=proposal.metric,
            success=True,
            branch="self-repair/test",
            worktree="X:/tmp/worktree",
            commit="deadbeef",
            returncode=0,
            stdout="ok",
            stderr="",
            reason="ok",
        )


def test_self_repair_request_emits_proposal_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELF_REPAIR_AUTO_VALIDATE", "1")
    bus = SyncEventBus()
    tasks = InlineTaskManager()
    executor = DummyExecutor()
    manager = CodeSelfRepairManager(
        event_bus=bus,  # type: ignore[arg-type]
        task_manager=tasks,  # type: ignore[arg-type]
        executor=executor,
        enabled=True,
    )

    diff = (
        "diff --git a/backend/example.py b/backend/example.py\n"
        "--- a/backend/example.py\n"
        "+++ b/backend/example.py\n"
        "@@ -1 +1 @@\n"
        "-x=1\n"
        "+x=2\n"
    )
    bus.publish("self_repair.request", {"metric": "code_health", "summary": "Fix range clamp", "diff": diff})

    topics = [topic for topic, _ in bus.published]
    assert "self_repair.patch_proposed" in topics
    assert "self_repair.patch_validated" in topics
    assert "self_repair.review_required" in topics
    assert len(executor.calls) == 1
    manager.close()


def test_self_repair_rejects_protected_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELF_REPAIR_AUTO_VALIDATE", "1")
    bus = SyncEventBus()
    tasks = InlineTaskManager()
    executor = DummyExecutor()
    manager = CodeSelfRepairManager(
        event_bus=bus,  # type: ignore[arg-type]
        task_manager=tasks,  # type: ignore[arg-type]
        executor=executor,
        enabled=True,
    )

    diff = (
        "diff --git a/third_party/vendor.py b/third_party/vendor.py\n"
        "--- a/third_party/vendor.py\n"
        "+++ b/third_party/vendor.py\n"
        "@@ -1 +1 @@\n"
        "-x=1\n"
        "+x=2\n"
    )
    bus.publish("self_repair.request", {"summary": "Try to patch vendor", "diff": diff})

    topics = [topic for topic, _ in bus.published]
    assert "self_repair.patch_rejected" in topics
    assert "self_repair.patch_validated" not in topics
    assert not executor.calls
    manager.close()

