"""Tests for planning and execution helpers."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence


class _AsyncFile:
    def __init__(self, path: Path, encoding: str) -> None:
        self._path = path
        self._encoding = encoding
        self._content = ""

    async def __aenter__(self) -> "_AsyncFile":
        self._content = self._path.read_text(encoding=self._encoding)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial
        return False

    async def read(self) -> str:
        return self._content


aiofiles = types.ModuleType("aiofiles")


def _aio_open(path, mode="r", encoding="utf-8"):
    if "r" not in mode:
        raise NotImplementedError("Stub aiofiles only supports read mode")
    return _AsyncFile(Path(path), encoding)


aiofiles.open = _aio_open  # type: ignore[attr-defined]
sys.modules["aiofiles"] = aiofiles


class _BaseCache(dict):
    def __init__(self, maxsize: int, ttl: int | None = None) -> None:
        super().__init__()
        self.maxsize = maxsize
        self.ttl = ttl
        self.currsize = 0

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.currsize = len(self)

    def pop(self, key, default=None):
        value = super().pop(key, default)
        self.currsize = len(self)
        return value

    def clear(self):  # pragma: no cover - trivial
        super().clear()
        self.currsize = 0


class LRUCache(_BaseCache):
    pass


class TTLCache(_BaseCache):
    pass


cachetools = types.ModuleType("cachetools")
cachetools.LRUCache = LRUCache
cachetools.TTLCache = TTLCache
sys.modules["cachetools"] = cachetools


class MultiHopAssociator:
    def __init__(self, graph):
        self.graph = graph

    def find_path(self, start, goal):
        queue = [(start, [start])]
        visited = {start}
        while queue:
            current, path = queue.pop(0)
            if current == goal:
                return path
            for neighbour in self.graph.get(current, []):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, path + [neighbour]))
        return []


class ReflectionModule:
    def reflect(self, text):  # pragma: no cover - trivial
        return SimpleNamespace(confidence=1.0), text


class DivergentConvergentSolver:
    def __init__(self, associator, reflection):
        self.associator = associator
        self.reflection = reflection

    def _generate_paths(self, start, goal, strategies):
        paths = []
        for strat in strategies:
            first = self.associator.find_path(start, strat)
            second = self.associator.find_path(strat, goal)
            if first and second:
                paths.append(first + second[1:])
        return paths

    def solve(self, start, goal, strategies):
        candidates = []
        for path in self._generate_paths(start, goal, strategies):
            evaluation, _ = self.reflection.reflect(" ".join(path))
            score = int(getattr(evaluation, "confidence", 1.0) * len(path)) or len(path)
            candidates.append((score, path))
        if not candidates:
            return [], 0
        candidates.sort(key=lambda item: item[0])
        score, best = candidates[-1]
        return best, score


@dataclass(slots=True)
class TaskContext:
    name: str
    required_capabilities: Sequence[str] = field(default_factory=tuple)
    metadata: Optional[Mapping[str, Any]] = None

    def requirement_set(self) -> set[str]:  # pragma: no cover - trivial
        return {cap.lower() for cap in self.required_capabilities}


@dataclass(slots=True)
class SpecialistModule:
    name: str
    capabilities: Iterable[str]
    solver: Callable[[Dict[str, float], TaskContext], Any]
    priority: float = 0.0
    usage_count: int = 0
    total_score: float = 0.0

    def __post_init__(self) -> None:  # pragma: no cover - simple setup
        self.capabilities = {cap.lower() for cap in self.capabilities}

    def matches(self, requirements: Iterable[str]) -> bool:
        requirements = {req.lower() for req in requirements}
        return bool(requirements) and requirements.issubset(self.capabilities)

    def record_performance(self, score: float) -> None:  # pragma: no cover - trivial
        self.usage_count += 1
        self.total_score += score

    @property
    def average_score(self) -> float:  # pragma: no cover - trivial
        if not self.usage_count:
            return 0.0
        return self.total_score / self.usage_count


class SpecialistModuleRegistry:
    def __init__(self, modules: Optional[Iterable[SpecialistModule]] = None) -> None:
        self._modules: Dict[str, SpecialistModule] = {}
        if modules:
            for module in modules:
                self.register(module)

    def register(self, module: SpecialistModule) -> None:  # pragma: no cover - trivial
        self._modules[module.name] = module

    def get(self, name: str) -> Optional[SpecialistModule]:  # pragma: no cover
        return self._modules.get(name)

    def matching_modules(self, task: TaskContext) -> list[SpecialistModule]:
        requirements = task.requirement_set()
        return [module for module in self._modules.values() if module.matches(requirements)]

    def select_best(self, task: TaskContext) -> Optional[SpecialistModule]:
        candidates = self.matching_modules(task)
        if not candidates:
            return None

        def score(candidate: SpecialistModule) -> tuple[float, float]:
            if candidate.usage_count:
                return (candidate.average_score, candidate.priority)
            return (candidate.priority, candidate.priority)

        return max(candidates, key=score)

    def update_performance(self, name: str, score: float) -> None:  # pragma: no cover
        module = self.get(name)
        if module is not None:
            module.record_performance(score)


def _load(name: str, path: str):
    abs_path = Path(__file__).resolve().parent.parent / path
    spec = importlib.util.spec_from_file_location(name, str(abs_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


import types, sys

ps = types.ModuleType("backend.creative_engine.problem_solver")
ps.DivergentConvergentSolver = DivergentConvergentSolver
backend_ce = types.ModuleType("backend.creative_engine")
backend_ce.problem_solver = ps
sys.modules["backend.creative_engine"] = backend_ce
sys.modules["backend.creative_engine.problem_solver"] = ps
backend_execution = types.ModuleType("backend.execution")
backend_execution.__path__ = []
sys.modules["backend.execution"] = backend_execution


class TaskGraph:
    def __init__(self) -> None:  # pragma: no cover - trivial container
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def add_task(self, name: str, *, description: str, skill: str, dependencies: Sequence[str]):
        self.tasks[name] = {
            "description": description,
            "skill": skill,
            "dependencies": list(dependencies),
        }


class Scheduler:
    def __init__(self) -> None:  # pragma: no cover - trivial scheduler
        self._agents: list[str] = []

    def add_agent(self, name: str) -> None:
        if name not in self._agents:
            self._agents.append(name)

    async def submit(self, graph: TaskGraph, callback: Callable[[str, str], Any]):
        results: Dict[str, Any] = {}
        for name, task in graph.tasks.items():
            result = await callback("local", task["skill"])
            results[name] = result
        return results


task_graph_module = types.ModuleType("backend.execution.task_graph")
task_graph_module.TaskGraph = TaskGraph
sys.modules["backend.execution.task_graph"] = task_graph_module
backend_execution.task_graph = task_graph_module

scheduler_module = types.ModuleType("backend.execution.scheduler")
scheduler_module.Scheduler = Scheduler
sys.modules["backend.execution.scheduler"] = scheduler_module
backend_execution.scheduler = scheduler_module

# Stub modules required by executor
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
skill_lib_pkg = types.ModuleType("capability")
skill_lib_pkg.__path__ = []  # type: ignore[attr-defined]
skill_lib_module = _load("capability.skill_library", "backend/capability/skill_library.py")
skill_lib_pkg.skill_library = skill_lib_module
sys.modules["capability"] = skill_lib_pkg
sys.modules["capability.skill_library"] = skill_lib_module
async_utils = types.ModuleType("common.async_utils")

def _run_async(coro):
    import asyncio

    return asyncio.run(coro)

async_utils.run_async = _run_async
sys.modules["common"] = types.ModuleType("common")
sys.modules["common.async_utils"] = async_utils

modules_pkg = types.ModuleType("modules")
modules_pkg.__path__ = []
sys.modules["modules"] = modules_pkg
modules_evolution_pkg = types.ModuleType("modules.evolution")
modules_evolution_pkg.__path__ = []
sys.modules["modules.evolution"] = modules_evolution_pkg
evolution_mod = types.ModuleType("modules.evolution.evolution_engine")
evolution_mod.TaskContext = TaskContext
evolution_mod.SpecialistModule = SpecialistModule
evolution_mod.SpecialistModuleRegistry = SpecialistModuleRegistry
sys.modules["modules.evolution.evolution_engine"] = evolution_mod
modules_pkg.evolution = modules_evolution_pkg
modules_evolution_pkg.evolution_engine = evolution_mod

planner_mod = _load("backend.execution.planner", "backend/execution/planner.py")
executor_mod = _load("backend.execution.executor", "backend/execution/executor.py")

Planner = planner_mod.Planner
Executor = executor_mod.Executor
register_specialist_skill = skill_lib_module.register_specialist_skill


class DummySkillLibrary:
    def __init__(self) -> None:
        self.skills = {}
        self.add_skill("plan_a", "def plan_a():\n    return 'A'")
        self.add_skill("plan_b", "def plan_b():\n    return 'B'")

    def add_skill(self, name: str, code: str) -> None:
        sig = hashlib.sha256(code.encode("utf-8")).hexdigest()
        self.skills[name] = (code, {"signature": sig})

    def list_skills(self):  # pragma: no cover - trivial
        return list(self.skills.keys())

    async def get_skill(self, name: str):  # pragma: no cover - simple awaitable
        return self.skills[name]


def test_divergent_convergent_solver():
    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["E"],
        "E": ["D"],
        "D": [],
    }
    solver = DivergentConvergentSolver(MultiHopAssociator(graph), ReflectionModule())
    path, score = solver.solve("A", "D", ["B", "C"])
    assert path == ["A", "C", "E", "D"]
    assert score == 4


def test_planner_solve_delegates_solver():
    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["E"],
        "E": ["D"],
        "D": [],
    }
    solver = DivergentConvergentSolver(MultiHopAssociator(graph), ReflectionModule())
    planner = Planner(solver=solver)
    result = planner.solve({"start": "A", "goal": "D", "strategies": ["B", "C"]})
    assert result == ["A", "C", "E", "D"]


def test_planner_uses_specialist_for_tagged_goal():
    def specialist_solver(_state, context):
        return {"plan": f"Complete {context.name}"}

    specialist = SpecialistModule(
        name="analysis_guru", capabilities={"analysis"}, solver=specialist_solver
    )
    registry = SpecialistModuleRegistry([specialist])
    planner = Planner(registry=registry)

    goal = "Perform deep analysis [capability:analysis]"

    plan = planner.solve(goal)

    assert plan
    assert plan[0].startswith("[analysis_guru] plan: Complete Perform deep analysis")
    assert any("analysis_guru" in step for step in plan)


def test_planner_falls_back_to_heuristic_when_no_specialist():
    planner = Planner()
    goal = "Check the logs; summarise findings"

    plan = planner.solve(goal)

    assert plan == ["Check the logs", "summarise findings"]


def test_executor_selects_best_plan():
    lib = DummySkillLibrary()
    executor = Executor(lib)
    plans = [("plan_a", 0.1), ("plan_b", 0.9)]
    result = executor.execute_sync(plans)
    assert result["plan_b"] == "B"


def test_executor_executes_specialist_callable(tmp_path):
    skill_lib = skill_lib_module.SkillLibrary(tmp_path)

    def specialist_solver(architecture, task):
        return {
            "architecture": architecture,
            "task": task.name,
            "capabilities": list(task.required_capabilities),
        }

    specialist = SpecialistModule(
        name="analysis_guru",
        capabilities={"analysis", "review"},
        solver=specialist_solver,
        priority=1.5,
    )

    skill_name = register_specialist_skill(
        skill_lib,
        specialist,
        default_architecture={"score": 1.0},
        task_metadata={"origin": "unit-test"},
    )

    executor = Executor(skill_lib)
    result = executor.execute_sync([(skill_name, 1.0)])

    outcome = result[skill_name]
    assert outcome["architecture"] == {"score": 1.0}
    assert outcome["task"] == "analysis_guru"
    assert outcome["capabilities"] == ["analysis", "review"]
