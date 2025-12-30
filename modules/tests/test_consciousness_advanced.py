import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.consciousness import ConsciousnessModel
from modules.brain.consciousness_advanced import (
    ConsciousnessAdvanced,
    AdaptiveAttention,
)


def _dataset():
    # Ground truth salience threshold of roughly 0.65
    return [
        {"score": 0.55, "ground_truth": 0},
        {"score": 0.60, "ground_truth": 0},
        {"score": 0.70, "ground_truth": 1},
        {"score": 0.80, "ground_truth": 1},
        {"score": 0.58, "ground_truth": 0},
        {"score": 0.90, "ground_truth": 1},
    ]


def test_hierarchical_workspace_and_metacognition():
    model = ConsciousnessAdvanced()
    data = _dataset()
    acc = model.evaluate_dataset(data)
    assert acc >= 0.66
    assert len(model.workspace.global_broadcasts) >= sum(d["ground_truth"] for d in data)
    assert sum(len(v) for v in model.workspace.local.values()) > 0
    assert model.meta.accuracy() == acc


def test_accuracy_improvement_over_simple_model():
    data = _dataset()
    simple = ConsciousnessModel()
    correct = 0
    for item in data:
        info = {"data": "x", "is_salient": item["score"] > 0.5}
        pred = simple.conscious_access(info)
        correct += int(pred == bool(item["ground_truth"]))
    simple_acc = correct / len(data)

    advanced = ConsciousnessAdvanced(attention=AdaptiveAttention())
    adv_acc = advanced.evaluate_dataset(data)

    assert adv_acc > simple_acc
