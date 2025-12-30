import csv
import random
import sys
import types

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    class _Tensor:
        pass

    def _tensor(*args, **kwargs):
        return _Tensor()

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    torch_stub.Tensor = _Tensor
    torch_stub.tensor = _tensor
    torch_stub.zeros = _tensor
    torch_stub.zeros_like = lambda *_args, **_kwargs: _Tensor()
    torch_stub.float32 = "float32"
    torch_stub.save = lambda *args, **kwargs: None
    torch_stub.load = lambda *args, **kwargs: {}
    torch_stub.no_grad = _NoGrad

    nn_stub = types.ModuleType("torch.nn")
    class _Module:
        def parameters(self):
            return []
    nn_stub.Module = _Module
    class _MSELoss:
        def __call__(self, *args, **kwargs):
            return 0.0
    nn_stub.MSELoss = _MSELoss

    optim_stub = types.ModuleType("torch.optim")
    class _Optimizer:
        def __init__(self, *args, **kwargs):
            pass

        def zero_grad(self):
            pass

        def step(self):
            pass
    optim_stub.Optimizer = _Optimizer
    optim_stub.Adam = _Optimizer
    optim_stub.AdamW = _Optimizer

    torch_stub.nn = nn_stub
    torch_stub.optim = optim_stub

    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.optim"] = optim_stub

import importlib
from pathlib import Path

if "backend.ml" not in sys.modules:
    backend_pkg = importlib.import_module("backend")
    ml_path = Path(__file__).resolve().parents[2] / "backend" / "ml"
    ml_module = types.ModuleType("backend.ml")
    ml_module.__path__ = [str(ml_path)]

    class _StubConfig:
        def __init__(self) -> None:
            self.train_after_samples = 1

    ml_module.TrainingConfig = _StubConfig
    ml_module.DEFAULT_TRAINING_CONFIG = _StubConfig()
    ml_module.get_model = lambda *args, **kwargs: None
    sys.modules["backend.ml"] = ml_module
    setattr(backend_pkg, "ml", ml_module)

    continual_stub = types.ModuleType("backend.ml.continual_trainer")

    class _StubTrainer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_sample(self, sample) -> None:
            pass

        def train(self) -> None:
            pass

    continual_stub.ContinualTrainer = _StubTrainer
    sys.modules["backend.ml.continual_trainer"] = continual_stub

if "pandas" not in sys.modules:
    import csv as _csv

    class _StubDataFrame:
        def __init__(self, data=None, columns=None):
            rows = data or []
            if rows and isinstance(rows[0], dict):
                self.columns = columns or list(rows[0].keys())
                self.data = [
                    {col: row.get(col, "") for col in self.columns}
                    for row in rows
                ]
            else:
                self.columns = columns or []
                self.data = []

        def __len__(self) -> int:
            return len(self.data)

        def to_csv(self, path, index=False):
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = _csv.writer(handle)
                writer.writerow(self.columns)
                for row in self.data:
                    writer.writerow([row.get(col, "") for col in self.columns])

        def drop_duplicates(self, subset, keep="last", inplace=False):
            seen = {}
            for row in self.data:
                key = tuple(row.get(col) for col in subset)
                seen[key] = row
            result = list(seen.values())
            if inplace:
                self.data = result
                return None
            return _StubDataFrame(result, columns=self.columns)

    def _stub_read_csv(path):
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = _csv.DictReader(handle)
            rows = list(reader)
            return _StubDataFrame(rows, columns=reader.fieldnames or [])

    def _stub_concat(frames, ignore_index=True):
        columns = frames[0].columns if frames else []
        data = []
        for frame in frames:
            data.extend(frame.data)
        return _StubDataFrame(data, columns=columns)

    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = _StubDataFrame
    pandas_stub.read_csv = _stub_read_csv
    pandas_stub.concat = _stub_concat
    sys.modules["pandas"] = pandas_stub

if "autogpts" not in sys.modules:
    autogpts_pkg = types.ModuleType("autogpts")
    autogpts_pkg.__path__ = []
    sys.modules["autogpts"] = autogpts_pkg

    autogpt_pkg = types.ModuleType("third_party.autogpt")
    autogpt_pkg.__path__ = []
    sys.modules["third_party.autogpt"] = autogpt_pkg

    autogpt_sub_pkg = types.ModuleType("third_party.autogpt.autogpt")
    autogpt_sub_pkg.__path__ = []
    sys.modules["third_party.autogpt.autogpt"] = autogpt_sub_pkg

    core_pkg = types.ModuleType("third_party.autogpt.autogpt.core")
    core_pkg.__path__ = []
    sys.modules["third_party.autogpt.autogpt.core"] = core_pkg

    errors_module = types.ModuleType("third_party.autogpt.autogpt.core.errors")

    class _AutoGPTError(Exception):
        pass

    errors_module.AutoGPTError = _AutoGPTError
    sys.modules["third_party.autogpt.autogpt.core.errors"] = errors_module

    logging_module = types.ModuleType("third_party.autogpt.autogpt.core.logging")

    def _handle_exception(exc: Exception | None = None) -> None:  # type: ignore[unused-arg]
        return None

    logging_module.handle_exception = _handle_exception
    sys.modules["third_party.autogpt.autogpt.core.logging"] = logging_module

if "events" not in sys.modules:
    events_stub = types.ModuleType("events")

    class _StubEventBus:
        def publish(self, topic: str, event: dict) -> None:
            return None

        def subscribe(self, topic: str, handler):  # type: ignore[missing-return-type]
            return lambda: None

        def unsubscribe(self, topic: str, handler) -> None:
            return None

    def _create_event_bus():
        return _StubEventBus()

    def _publish(bus, topic, event) -> None:  # type: ignore[unused-arg]
        return None

    events_stub.create_event_bus = _create_event_bus
    events_stub.publish = _publish
    sys.modules["events"] = events_stub

from backend.memory.long_term import LongTermMemory
import backend.ml.experience_collector as collector
from backend.ml.learning_loop import (
    ExecutionFeedback,
    LearningCycleConfig,
    LearningLoopOrchestrator,
)
from backend.ml.retraining_pipeline import TRAINING_INTERACTION_CATEGORY
from backend.reflection.reflection import ReflectionModule, ReflectionResult


class _DummyTrainer:
    def __init__(self) -> None:
        self.samples: list[dict] = []

    def add_sample(self, sample: dict) -> None:
        self.samples.append(sample)


def _make_reflector() -> ReflectionModule:
    return ReflectionModule(
        evaluate=lambda text: ReflectionResult(confidence=0.8, sentiment="positive", raw=text),
        rewrite=lambda text: text + " refined",
        max_passes=1,
        quality_threshold=0.5,
    )


def test_learning_loop_explores_and_logs(tmp_path):
    original_log_file = collector.LOG_FILE
    original_trainer = collector.TRAINER
    original_selector = collector.SELECTOR

    collector.LOG_FILE = tmp_path / "new_logs.csv"
    collector.TRAINER = _DummyTrainer()
    collector.SELECTOR = collector.ActiveCuriositySelector()

    memory_path = tmp_path / "memory.db"
    memory = LongTermMemory(memory_path)

    planner_calls: list[dict] = []
    executor_calls: list[str] = []
    reflector_module = _make_reflector()

    def planner(task, strategy=None):
        planner_calls.append({"task": task, "strategy": strategy})
        return {
            "strategies": ["baseline", "explore"],
            "confidence": 0.7,
            "prompt": "solve-x",
        }

    def executor(plan, *, strategy, context):
        executor_calls.append(strategy)
        return ExecutionFeedback(
            result={"output": f"{strategy}-result"},
            reward=1.0 if strategy == "baseline" else 0.4,
            success=True,
            logs=[f"log-{strategy}"],
            metrics={"latency": 0.1},
            confidence=0.65 if strategy == "baseline" else 0.5,
        )

    def reflector(plan, feedback):
        evaluation, revision = reflector_module.reflect(f"{plan['prompt']}|{feedback.result}")
        return {"revision": revision, "confidence": evaluation.confidence}

    try:
        orchestrator = LearningLoopOrchestrator(
            planner,
            executor,
            reflector=reflector,
            config=LearningCycleConfig(exploration_probability=1.0, exploration_attempts=1),
            memory=memory,
            exploration_rng=random.Random(7),
        )

        summary = orchestrator.run_cycle({"id": "task-123", "ability": "demo"})

        assert summary["exploration_triggered"] is True
        assert len(summary["exploration_runs"]) == 1
        assert {"baseline", "explore"} == set(executor_calls)
        assert collector.LOG_FILE.exists()
        memory_reader = LongTermMemory(memory_path)
        try:
            stored = list(memory_reader.get(category=TRAINING_INTERACTION_CATEGORY))
        finally:
            memory_reader.close()
        assert stored, "expected training interactions persisted to memory"
    finally:
        memory.close()
        collector.LOG_FILE = original_log_file
        collector.TRAINER = original_trainer
        collector.SELECTOR = original_selector





