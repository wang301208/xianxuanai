from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evolution.founder import Founder


class DummyModel:
    def predict_next(self):
        return {"cpu_percent": 90, "memory_percent": 90}


def test_founder_uses_policy_model():
    founder = Founder(model=DummyModel())
    metrics = {"cpu_percent": 10, "memory_percent": 10}
    suggestions = founder._generate_suggestions(metrics)
    assert "CPU usage high; consider distributing tasks." in suggestions
    assert "Memory usage high; investigate memory leaks." in suggestions
