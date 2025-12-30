from __future__ import annotations

import time
from collections import deque
from importlib import util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "backend" / "execution" / "imitation_updates.py"
spec = util.spec_from_file_location("backend.execution.imitation_updates", MODULE_PATH)
imitation_updates = util.module_from_spec(spec)
assert spec and spec.loader  # for mypy
spec.loader.exec_module(imitation_updates)
apply_online_imitation_updates = imitation_updates.apply_online_imitation_updates


class _FakeImitationModel:
    def __init__(self) -> None:
        self.batch_calls = []

    def observe_batch(self, batch):
        self.batch_calls.append(list(batch))
        return {"loss": 0.25, "entropy": 0.5, "accuracy": 1.0}


def test_apply_online_imitation_updates_trains_new_samples() -> None:
    model = _FakeImitationModel()
    now = time.time()
    demo_buffer = deque(
        [
            {"state": {"fused_embedding": [0.0]}, "action": "a", "timestamp": now - 10.0},
            {"state": {"fused_embedding": [1.0]}, "action": "b", "timestamp": now - 1.0},
            {"state": {"fused_embedding": [2.0]}, "action": "c", "timestamp": now - 2.0, "trained": True},
        ]
    )

    metrics = apply_online_imitation_updates(model, demo_buffer, max_samples=2)

    assert metrics["imitation_updates"] == 2.0
    assert metrics["imitation_loss"] == 0.25
    assert metrics["imitation_entropy"] == 0.5
    assert metrics["imitation_accuracy"] == 1.0
    assert len(model.batch_calls) == 1
    # Most recent untrained samples first: b, then a.
    assert [sample["action"] for sample in model.batch_calls[0]] == ["b", "a"]
    assert demo_buffer[0].get("trained") is True
    assert demo_buffer[1].get("trained") is True
