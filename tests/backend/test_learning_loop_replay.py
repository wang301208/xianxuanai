import json
import random
import sys
import time
import types

if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.array = lambda *args, **kwargs: args  # type: ignore[assignment]
    numpy_stub.ndarray = object  # type: ignore[assignment]
    sys.modules["numpy"] = numpy_stub

if "pandas" not in sys.modules:
    import csv as _csv

    class _StubDataFrame:
        def __init__(self, data=None, columns=None):
            rows = data or []
            self.columns = columns or []
            self.data = rows

        def __len__(self):
            return len(self.data)

        def to_csv(self, path, index=False):  # noqa: ARG002
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = _csv.writer(handle)
                writer.writerow(self.columns)
                for row in self.data:
                    writer.writerow(row if isinstance(row, list) else [])

        def drop_duplicates(self, subset=None, keep="last", inplace=False):  # noqa: ARG002
            return self

    def _stub_read_csv(path, dtype=None):  # noqa: ARG002
        return _StubDataFrame([], columns=[])

    def _stub_concat(frames, ignore_index=True):  # noqa: ARG002
        return _StubDataFrame([], columns=[])

    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = _StubDataFrame
    pandas_stub.read_csv = _stub_read_csv
    pandas_stub.concat = _stub_concat
    sys.modules["pandas"] = pandas_stub

if "events" not in sys.modules:
    events_stub = types.ModuleType("events")

    class _StubEventBus:
        def publish(self, topic: str, event: dict):  # noqa: ARG002
            return None

        def subscribe(self, topic: str, handler):  # noqa: ARG002
            return lambda: None

        def unsubscribe(self, topic: str, handler):  # noqa: ARG002
            return None

    events_stub.create_event_bus = lambda: _StubEventBus()
    events_stub.publish = lambda bus, topic, event: None  # noqa: ARG002
    sys.modules["events"] = events_stub

import backend.ml.experience_collector as collector
from backend.memory.long_term import LongTermMemory
from backend.ml.learning_loop import (
    ExecutionFeedback,
    LearningCycleConfig,
    LearningLoopOrchestrator,
    ReplayConfig,
)
from backend.ml.retraining_pipeline import REFLECTION_CATEGORY, TRAINING_INTERACTION_CATEGORY


def _executor_factory(executor_calls):
    def _executor(plan, *, strategy, context):
        executor_calls.append({"plan": plan, "strategy": strategy, "context": context})
        return ExecutionFeedback(result={"echo": plan}, success=True, confidence=0.6, reward=0.8)

    return _executor


def test_replay_prioritises_low_confidence(tmp_path):
    original_log_file = collector.LOG_FILE
    collector.LOG_FILE = tmp_path / "logs.csv"
    memory = LongTermMemory(tmp_path / "mem.db")

    memory.add(
        TRAINING_INTERACTION_CATEGORY,
        json.dumps(
            {
                "task": "t-high",
                "ability": "agent",
                "plan": {"plan_name": "high"},
                "strategy": "baseline",
            }
        ),
        confidence=0.9,
        timestamp=time.time() - 200,
    )
    memory.add(
        TRAINING_INTERACTION_CATEGORY,
        json.dumps(
            {
                "task": "t-low",
                "ability": "agent",
                "plan": {"plan_name": "low"},
                "strategy": "baseline",
            }
        ),
        confidence=0.1,
        timestamp=time.time() - 100,
    )

    executor_calls: list[dict] = []
    orchestrator = LearningLoopOrchestrator(
        planner=lambda task, strategy=None: {"strategies": ["baseline"]},
        executor=_executor_factory(executor_calls),
        config=LearningCycleConfig(),
        replay_config=ReplayConfig(
            enable_replay=True,
            replay_interval_seconds=0,
            replay_batch_size=1,
            sampling_strategy="low_confidence",
            categories=(TRAINING_INTERACTION_CATEGORY,),
        ),
        memory=memory,
        exploration_rng=random.Random(42),
    )

    results = orchestrator.run_replay_if_idle(now=time.time())

    assert executor_calls, "Executor should be invoked for replayed entries"
    assert executor_calls[0]["plan"].get("plan_name") == "low"
    assert results and results[0]["strategy"] == "baseline"

    collector.LOG_FILE = original_log_file


def test_replay_mixed_curriculum_balances_domains(tmp_path):
    original_log_file = collector.LOG_FILE
    collector.LOG_FILE = tmp_path / "logs.csv"
    memory = LongTermMemory(tmp_path / "mem.db")

    memory.add(
        TRAINING_INTERACTION_CATEGORY,
        json.dumps(
            {
                "task": "t-train",
                "ability": "agent",
                "plan": {"plan_name": "train"},
                "strategy": "baseline",
            }
        ),
        confidence=0.6,
    )
    memory.add(
        REFLECTION_CATEGORY,
        json.dumps(
            [
                {
                    "revision": "rev1",
                    "evaluation": {"confidence": 0.3, "sentiment": "positive"},
                }
            ]
        ),
        confidence=0.2,
    )

    executor_calls: list[dict] = []
    orchestrator = LearningLoopOrchestrator(
        planner=lambda task, strategy=None: {"strategies": ["baseline"]},
        executor=_executor_factory(executor_calls),
        config=LearningCycleConfig(),
        replay_config=ReplayConfig(
            enable_replay=True,
            replay_interval_seconds=0,
            replay_batch_size=2,
            sampling_strategy="uniform",
            categories=(TRAINING_INTERACTION_CATEGORY, REFLECTION_CATEGORY),
            mixed_curriculum=True,
        ),
        memory=memory,
        exploration_rng=random.Random(7),
    )

    results = orchestrator.run_replay_if_idle(now=time.time())

    assert len(results) == 2, "Mixed replay should include both domains"
    assert any(r["strategy"] == "baseline" for r in results)
    assert any(r["ability"] == "reflection" for r in results)
    assert executor_calls, "Training entries should still exercise the executor"
    assert len(list(memory.get(category=TRAINING_INTERACTION_CATEGORY))) >= 1

    collector.LOG_FILE = original_log_file
