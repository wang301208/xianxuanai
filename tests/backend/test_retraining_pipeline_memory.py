import csv
import json
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

    class _MSELoss:
        def __call__(self, *args, **kwargs):
            return 0.0

    nn_stub.Module = _Module
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
import backend.ml.retraining_pipeline as pipeline


def test_augment_dataset_with_memory(tmp_path):
    memory_path = tmp_path / "memory.db"
    dataset_path = tmp_path / "dataset.csv"
    memory = LongTermMemory(memory_path)
    try:
        memory.add(
            pipeline.REFLECTION_CATEGORY,
            json.dumps(
                [
                    {
                        "evaluation": {
                            "confidence": 0.72,
                            "sentiment": "positive",
                            "raw": "score:0.72",
                        },
                        "revision": "improved answer",
                    }
                ]
            ),
        )
        memory.add(
            pipeline.SELF_MONITORING_CATEGORY,
            json.dumps(
                {
                    "summary": "action=plan,status=retry",
                    "revision": "apply new tool order",
                    "evaluation": {"confidence": 0.6},
                }
            ),
        )
        memory.add(
            pipeline.TRAINING_INTERACTION_CATEGORY,
            json.dumps(
                {
                    "task": "task-001",
                    "ability": "demo",
                    "strategy": "baseline",
                    "plan": {"prompt": "foo"},
                    "analysis": {"confidence": 0.5},
                    "reflection": {"notes": "ok"},
                    "result": {"output": "bar"},
                    "logs": ["log-entry"],
                    "metrics": {"score": 1.0},
                    "reward": 0.9,
                    "success": True,
                }
            ),
        )
    finally:
        memory.close()

    added = pipeline.augment_dataset_with_memory(
        memory_path=memory_path, dataset=dataset_path, limit=9
    )
    assert added >= 3

    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == pipeline.MEMORY_DATASET_COLUMNS
        rows = list(reader)
        assert rows



