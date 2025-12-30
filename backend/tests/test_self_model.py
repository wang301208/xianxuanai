import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.self_model import SelfModel
from backend.reflection import ReflectionModule, ReflectionResult
from backend.memory.long_term import LongTermMemory


def test_capabilities_adjust_on_negative_feedback(tmp_path):
    memory = LongTermMemory(tmp_path / "mem.db")

    def negative_eval(_text: str) -> ReflectionResult:
        return ReflectionResult(confidence=0.2, sentiment="negative", raw="low")

    reflection = ReflectionModule(evaluate=negative_eval, rewrite=lambda x: x)
    model = SelfModel(memory=memory)
    model._reflection = reflection
    model.set_capability("compute", 1.0)

    data = {"cpu": 0.0, "memory": 0.0}
    env = {"avg_cpu": 0.0, "avg_memory": 0.0}

    model.introspect(data, env, "compute")
    first = model._self_state["capabilities"]["compute"]
    model.introspect(data, env, "compute")
    second = model._self_state["capabilities"]["compute"]

    assert second < first < 1.0
    stored = list(memory.get("reflection_scores"))
    assert stored
    memory.close()

