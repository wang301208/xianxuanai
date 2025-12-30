from __future__ import annotations

import asyncio
import hashlib
import sys
import types
from pathlib import Path
import importlib.util

import pytest

# Stub dependencies required by Executor
errors_module = types.ModuleType("third_party.autogpt.autogpt.core.errors")
errors_module.SkillExecutionError = type("SkillExecutionError", (Exception,), {})
errors_module.SkillSecurityError = type("SkillSecurityError", (Exception,), {})
sys.modules["autogpts"] = types.ModuleType("autogpts")
sys.modules["third_party.autogpt"] = types.ModuleType("third_party.autogpt")
sys.modules["third_party.autogpt.autogpt"] = types.ModuleType("third_party.autogpt.autogpt")
sys.modules["third_party.autogpt.autogpt.core"] = types.ModuleType(
    "third_party.autogpt.autogpt.core"
)
sys.modules["third_party.autogpt.autogpt.core.errors"] = errors_module
skill_lib_module = types.ModuleType("capability.skill_library")
skill_lib_module.SkillLibrary = object
sys.modules["capability"] = types.ModuleType("capability")
sys.modules["capability.skill_library"] = skill_lib_module

sys.modules.setdefault("common", types.ModuleType("common"))

def _run_async(coro):
    return asyncio.run(coro)

common_async_utils = types.ModuleType("common.async_utils")
common_async_utils.run_async = _run_async
sys.modules["common.async_utils"] = common_async_utils


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[3] / path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


task_graph_mod = _load("backend.execution.task_graph", "backend/execution/task_graph.py")
scheduler_mod = _load("backend.execution.scheduler", "backend/execution/scheduler.py")
executor_mod = _load("backend.execution.executor", "backend/execution/executor.py")

TaskGraph = task_graph_mod.TaskGraph
Scheduler = scheduler_mod.Scheduler
Executor = executor_mod.Executor


class DummySkillLibrary:
    def __init__(self) -> None:
        self.skills = {}
        self.add_skill("a", "def a():\n    return 'A'")
        self.add_skill("b", "def b():\n    return 'B'")

    def add_skill(self, name: str, code: str) -> None:
        sig = hashlib.sha256(code.encode("utf-8")).hexdigest()
        self.skills[name] = (code, {"signature": sig})

    def list_skills(self):  # pragma: no cover - trivial
        return list(self.skills.keys())

    async def get_skill(self, name: str):  # pragma: no cover - simple awaitable
        return self.skills[name]


def test_scheduler_executes_tasks_async():
    scheduler = Scheduler()
    scheduler.add_agent("x")
    scheduler.add_agent("y")
    graph = TaskGraph()
    graph.add_task("a", description="a", skill="a")
    graph.add_task("b", description="b", skill="b", dependencies=["a"])

    async def worker(agent: str, skill: str) -> str:
        await asyncio.sleep(0.01)
        return f"{agent}:{skill}"

    async def main():
        return await scheduler.submit(graph, worker)

    results = asyncio.run(main())
    assert results["a"].endswith("a")
    assert results["b"].endswith("b")


def test_executor_sync_wrapper():
    lib = DummySkillLibrary()
    executor = Executor(lib)
    plans = [("a", 0.1), ("b", 0.9)]
    result = executor.execute_sync(plans)
    assert result["b"] == "B"


def test_scheduler_cancel_cleans_up():
    scheduler = Scheduler()
    scheduler.add_agent("x")
    graph = TaskGraph()
    graph.add_task("a", description="a", skill="a")

    async def worker(agent: str, skill: str) -> str:
        await asyncio.sleep(1)
        return "done"

    async def main():
        task = asyncio.create_task(scheduler.submit(graph, worker))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert scheduler._agents["x"]["tasks"] == 0
