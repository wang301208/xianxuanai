from __future__ import annotations

import asyncio
import importlib.util
import logging
import asyncio
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


if "numpy" not in sys.modules:  # pragma: no cover - lightweight stub for tests
    class _SimpleArray:
        def __init__(self, rows):
            self._rows = rows

        def __matmul__(self, other):
            result = []
            for row in self._rows:
                total = 0.0
                for value, weight in zip(row, other):
                    total += float(value) * float(weight)
                result.append(total)
            return result

        def __getitem__(self, index):
            return self._rows[index]

        def __setitem__(self, index, value):
            self._rows[index] = list(value)

        def fill(self, value):
            for row in self._rows:
                for idx in range(len(row)):
                    row[idx] = value

    def _zeros(shape, dtype=float):
        rows, cols = shape
        return _SimpleArray([[dtype() if callable(dtype) else dtype for _ in range(cols)] for _ in range(rows)])

    def _argmax(values):
        best_idx = 0
        best_val = None
        for idx, value in enumerate(values):
            if best_val is None or value > best_val:
                best_val = value
                best_idx = idx
        return best_idx

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.zeros = _zeros
    numpy_stub.argmax = _argmax
    numpy_stub.ndarray = _SimpleArray  # type: ignore[attr-defined]
    sys.modules["numpy"] = numpy_stub

if "yaml" not in sys.modules:  # pragma: no cover - minimal YAML stub for tests
    yaml_stub = types.ModuleType("yaml")

    def _safe_load(_stream):
        return {}

    def _dump(_data, *_args, **_kwargs):
        return ""

    yaml_stub.safe_load = _safe_load
    yaml_stub.dump = _dump
    sys.modules["yaml"] = yaml_stub

if "PIL" not in sys.modules:  # pragma: no cover - stub pillow dependency
    pil_module = types.ModuleType("PIL")
    pil_image_module = types.ModuleType("PIL.Image")

    class _DummyImage:
        pass

    def _open(*_args, **_kwargs):
        return _DummyImage()

    pil_image_module.Image = _DummyImage  # type: ignore[attr-defined]
    pil_image_module.open = _open
    pil_module.Image = pil_image_module  # type: ignore[attr-defined]
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = pil_image_module


def _load(name: str, path: str):
    module_path = ROOT / path
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Create package-like modules so relative imports resolve
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)

backend_pkg.memory = _load("backend.memory", "backend/memory/__init__.py")
backend_pkg.reflection = _load("backend.reflection", "backend/reflection/__init__.py")
backend_pkg.self_monitoring = _load("backend.self_monitoring", "backend/self_monitoring/__init__.py")

world_model_module = _load(
    "backend.world_model", "backend/world_model/__init__.py"
)
goal_module = _load(
    "backend.execution.goal_generator", "backend/execution/goal_generator.py"
)
memory_module = _load(
    "backend.memory.long_term", "backend/memory/long_term.py"
)
reflection_module = _load(
    "backend.reflection.reflection", "backend/reflection/reflection.py"
)
self_teacher_module = _load(
    "modules.knowledge.self_teacher", "modules/knowledge/self_teacher.py"
)

GoalGenerator = goal_module.GoalGenerator
GoalListener = goal_module.GoalListener
WorldModel = world_model_module.WorldModel
LongTermMemory = memory_module.LongTermMemory
ReflectionModule = reflection_module.ReflectionModule
ReflectionResult = reflection_module.ReflectionResult
SelfTeacher = self_teacher_module.SelfTeacher


def test_goal_generator_detects_blocked_task() -> None:
    world_model = WorldModel()
    generator = GoalGenerator(world_model=world_model, start_listener=False)
    try:
        world_model.add_task(
            "repair-pipeline",
            {"status": "blocked", "description": "Awaiting credentials"},
        )
        generator.process_world_state(world_model.get_state())
        goal = generator.generate()
        assert goal is not None
        assert "Unblock task 'repair-pipeline'" in goal
    finally:
        generator.stop()


def test_goal_generator_responds_to_environment_signal() -> None:
    generator = GoalGenerator(start_listener=False, world_model=None)
    try:
        generator.observe_environment(
            "New partnership opportunity detected in the logistics sector.",
            source="news.feed",
            severity=0.8,
        )
        goal = generator.generate()
        assert goal is not None
        assert "Respond to news.feed" in goal
    finally:
        generator.stop()


def test_goal_listener_scans_world_model() -> None:
    world_model = WorldModel()
    generator = GoalGenerator(world_model=world_model, start_listener=False)
    listener = GoalListener(world_model, generator, start=False)
    try:
        world_model.add_task(
            "launch-campaign",
            {
                "status": "in_progress",
                "description": "Marketing launch activities",
                "updated_at": "2000-01-01T00:00:00",
            },
        )
        listener.evaluate_once()
        goal = generator.generate()
        assert goal is not None
        assert "Review stalled task 'launch-campaign'" in goal
    finally:
        listener.stop()
        generator.stop()


def test_goal_generator_handles_failed_actions(tmp_path) -> None:
    world_model = WorldModel()

    def evaluation(text: str) -> ReflectionResult:
        score = 0.2 if "failed" in text else 0.9
        sentiment = "negative" if "failed" in text else "positive"
        return ReflectionResult(score, sentiment, raw=text)

    reflection = ReflectionModule(evaluate=evaluation, rewrite=lambda text: text + " :: revised", max_passes=1)
    memory = LongTermMemory(tmp_path / "goal_monitor.db")
    generator = GoalGenerator(
        reflection=reflection,
        memory=memory,
        world_model=world_model,
        start_listener=False,
    )
    try:
        world_model.record_action(
            "agent-1",
            "execute plan",
            status="failed",
            result="Plan execution failed with timeout",
            error="timeout",
            metrics={"latency": 5.0},
            metadata={"task_id": "plan-7"},
        )
        generator.process_world_state(world_model.get_state())

        goals: list[str] = []
        while True:
            goal = generator.generate()
            if goal is None:
                break
            goals.append(goal)
        assert goals
        combined = " ".join(goals).lower()

        assert "diagnose failure" in combined
        assert "retry action" in combined
    finally:
        generator.stop()
        memory.close()


def test_goal_generator_intrinsic_goal_generation(tmp_path) -> None:
    world_model = WorldModel()
    memory = LongTermMemory(tmp_path / "intrinsic.db")
    generator = GoalGenerator(
        world_model=world_model,
        memory=memory,
        start_listener=False,
        intrinsic_motivation=True,
        simulation_horizon=2,
    )

    try:
        world_model.update_competence("robotics", 0.3, source="test")
        goal = generator.generate()
        assert goal is not None
        assert "robotics" in goal.lower()
    finally:
        generator.stop()
        memory.close()


def test_self_teacher_failure_updates_world_model_and_goals() -> None:
    class StubAbilityRegistry:
        def __init__(self) -> None:
            self._question_returned = False

        def dump_abilities(self):
            return [types.SimpleNamespace(name="query_language_model")]

        async def perform(self, ability_name, **kwargs):
            query = kwargs.get("query", "")
            if "self-test question" in query and not self._question_returned:
                self._question_returned = True
                return types.SimpleNamespace(success=True, message="What is quantum mechanics?")
            return types.SimpleNamespace(success=False, message="UNKNOWN")

    class StubMemory:
        def __init__(self) -> None:
            self.entries: list[str] = []

        def add(self, *values: str) -> None:
            self.entries.append("".join(values))

    world_model = WorldModel()
    teacher = SelfTeacher(logger=logging.getLogger("test-self-teacher"), world_model=world_model)
    registry = StubAbilityRegistry()
    memory = StubMemory()
    concept = {"label": "quantum mechanics", "description": "Physics of the very small"}

    asyncio.run(
        teacher._exercise_concept(
            concept,
            ability_registry=registry,
            query_ability="query_language_model",
            ability_specs=registry.dump_abilities(),
            memory=memory,
            knowledge_acquisition=None,
            cognition_metadata={"context": "unit-test"},
        )
    )

    targets = world_model.suggest_learning_targets(limit=1)
    assert concept["label"] in targets

    generator = GoalGenerator(world_model=world_model, start_listener=False)
    try:
        generator.process_world_state(world_model.get_state())
        goal = generator.generate()
        assert goal is not None
        assert "self-improve" in goal.lower()
        assert "quantum mechanics" in goal.lower()
    finally:
        generator.stop()
