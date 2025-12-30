import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.consciousness import ConsciousnessModel
from modules.brain.consciousness.hierarchical_model import HierarchicalConsciousnessModel


def _task_switching_dataset():
    return [
        {"module": "taskA", "score": 0.6, "ground_truth": 1},
        {"module": "taskB", "score": 0.55, "ground_truth": 0},
        {"module": "taskB", "score": 0.58, "ground_truth": 0},
        {"module": "taskA", "score": 0.4, "ground_truth": 0},
    ]


def test_hierarchical_model_improves_task_switching():
    data = _task_switching_dataset()

    simple = ConsciousnessModel()
    preds = []
    for item in data:
        info = {"is_salient": item["score"] > 0.5}
        preds.append(simple.conscious_access(info))
    simple_acc = sum(int(p == bool(d["ground_truth"])) for p, d in zip(preds, data)) / len(data)

    model = HierarchicalConsciousnessModel()
    for item in data:
        model.focus_attention(item["module"], item)
    advanced_acc = model.accuracy()

    assert advanced_acc > simple_acc


def test_hierarchical_model_recovers_from_anomalies():
    simple = ConsciousnessModel()
    simple.conscious_access({"is_salient": False})
    assert not simple.workspace.broadcasts

    model = HierarchicalConsciousnessModel()
    model.intervene("taskA", threshold=0.9)
    model.focus_attention("taskA", {"score": 0.8, "ground_truth": 1})
    assert not model.global_broadcasts()

    model.recover_anomaly("taskA")
    assert model.global_broadcasts() and model.global_broadcasts()[-1]["score"] == 0.8
