import json
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from modules.evolution.self_improvement import SelfImprovement


def test_load_metrics_json(tmp_path):
    metrics = [{"cpu": {"percent": 50}, "memory": {"percent": 60}}]
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics))

    si = SelfImprovement(metrics_path=path)
    assert si._load_metrics() == [{"cpu.percent": 50.0, "memory.percent": 60.0}]


def test_load_metrics_yaml(tmp_path):
    metrics = [{"cpu": {"percent": 50}, "memory": {"percent": 60}}]
    path = tmp_path / "metrics.yaml"
    path.write_text(yaml.safe_dump(metrics))

    si = SelfImprovement(metrics_path=path)
    assert si._load_metrics() == [{"cpu.percent": 50.0, "memory.percent": 60.0}]


def test_record_ga_history_json(tmp_path):
    path = tmp_path / "ga_metrics.json"
    si = SelfImprovement(ga_metrics_path=path)
    history1 = [{"generation": 1, "elapsed_time": 0.1, "best_fitness": 0.5}]
    si._record_ga_history(history1)
    assert json.loads(path.read_text()) == history1

    history2 = [{"generation": 2, "elapsed_time": 0.2, "best_fitness": 0.6}]
    si._record_ga_history(history2)
    assert json.loads(path.read_text()) == history1 + history2
