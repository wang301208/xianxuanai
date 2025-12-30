import importlib
import os
import sys
import types
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sys.modules.setdefault("events", types.SimpleNamespace(EventBus=object))
monitoring_dir = Path(__file__).resolve().parent.parent / "monitoring"
pkg = types.ModuleType("monitoring")
pkg.__path__ = [str(monitoring_dir)]
sys.modules["monitoring"] = pkg

alerting = importlib.import_module("monitoring.alerting")
AlertRule = alerting.AlertRule
evaluate_alerts = alerting.evaluate_alerts


def test_alert_triggered() -> None:
    metrics = {"precision": 0.5, "recall": 0.7}
    rules = [AlertRule(metric="precision", threshold=0.8, op="lt", message="low precision")]
    alerts = evaluate_alerts(metrics, rules)
    assert alerts and alerts[0]["metric"] == "precision"
