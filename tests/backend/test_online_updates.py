import time
from collections import deque
from importlib import util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "backend" / "execution" / "online_updates.py"
spec = util.spec_from_file_location("backend.execution.online_updates", MODULE_PATH)
online_updates = util.module_from_spec(spec)
assert spec and spec.loader  # for mypy
spec.loader.exec_module(online_updates)
apply_online_model_updates = online_updates.apply_online_model_updates


class FakePredictiveModel:
    def __init__(self):
        self.calls = []

    def observe(self, data, metadata=None):
        self.calls.append((data, metadata))
        # Return varying values so averaging can be checked.
        idx = len(self.calls)
        return {
            "reconstruction_loss": 0.1 * idx,
            "prediction_loss": 0.2 * idx,
            "prediction_error": 0.05 * idx,
        }


def test_apply_online_model_updates_trains_new_perceptions():
    model = FakePredictiveModel()
    now = time.time()
    working_memory = deque(
        [
            {"type": "perception", "data": {"a": 1}, "timestamp": now - 5},
            {"type": "note", "data": {}, "timestamp": now - 4},
            {"type": "perception", "data": {"b": 2}, "timestamp": now - 1, "metadata": {"reward": 1.0}},
            {"type": "perception", "data": {"c": 3}, "timestamp": now - 2, "trained": True},
        ]
    )

    metrics = apply_online_model_updates(model, working_memory, max_samples=2)

    assert metrics["online_updates"] == 2
    # Only two most recent untrained perception entries should be processed.
    assert len(model.calls) == 2
    assert model.calls[0][0] == {"b": 2}
    assert model.calls[0][1]["reward"] == 1.0
    assert model.calls[1][0] == {"a": 1}
    # Entries should be marked to avoid retraining next cycle.
    assert working_memory[0]["trained"] is True
    assert working_memory[2]["trained"] is True
    # Metrics are averaged across processed samples.
    assert metrics["online_reconstruction_loss"] == 0.15000000000000002
    assert metrics["online_prediction_loss"] == 0.30000000000000004
    assert metrics["online_prediction_error"] == 0.07500000000000001


class FakeBatchPredictiveModel(FakePredictiveModel):
    def __init__(self):
        super().__init__()
        self.batch_calls = []

    def observe_batch(self, batch, metadata=None):
        self.batch_calls.append((list(batch), list(metadata or [])))
        # Return already-averaged losses for the whole batch.
        return {"reconstruction_loss": 0.25, "prediction_loss": 0.5, "prediction_error": 0.125}


def test_apply_online_model_updates_prefers_batch_updates():
    model = FakeBatchPredictiveModel()
    now = time.time()
    working_memory = deque(
        [
            {"type": "perception", "data": {"a": 1}, "timestamp": now - 5},
            {"type": "perception", "data": {"b": 2}, "timestamp": now - 1},
        ]
    )

    metrics = apply_online_model_updates(model, working_memory, max_samples=2)

    assert metrics["online_updates"] == 2
    assert metrics["online_reconstruction_loss"] == 0.25
    assert metrics["online_prediction_loss"] == 0.5
    assert metrics["online_prediction_error"] == 0.125
    assert model.batch_calls and model.batch_calls[0][0] == [{"b": 2}, {"a": 1}]
    assert model.calls == []
    assert working_memory[0]["trained"] is True
    assert working_memory[1]["trained"] is True


def test_apply_online_model_updates_no_model_returns_empty_metrics():
    working_memory = deque([{"type": "perception", "data": {"a": 1}, "timestamp": time.time()}])

    metrics = apply_online_model_updates(None, working_memory)

    assert metrics == {}
    # Entries remain untouched without a model.
    assert "trained" not in working_memory[0]
