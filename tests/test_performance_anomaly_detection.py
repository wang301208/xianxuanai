from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modules"))
sys.modules.setdefault("events", types.SimpleNamespace(EventBus=object))

import importlib.util
pkg = types.ModuleType("monitoring")
pkg.__path__ = [str(ROOT / "backend" / "monitoring")]
sys.modules.setdefault("monitoring", pkg)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

performance_monitor = load_module(
    "monitoring.performance_monitor",
    ROOT / "backend" / "monitoring" / "performance_monitor.py",
)
storage_module = load_module(
    "monitoring.storage", ROOT / "backend" / "monitoring" / "storage.py"
)
PerformanceMonitor = performance_monitor.PerformanceMonitor
TimeSeriesStorage = storage_module.TimeSeriesStorage


def test_anomaly_detection_triggers_alert(tmp_path):
    storage = TimeSeriesStorage(tmp_path / "mon.db")
    alerts = []
    monitor = PerformanceMonitor(
        storage,
        training_accuracy=0.0,
        degradation_threshold=0.0,
        alert_handlers=[lambda s, m: alerts.append((s, m))],
        enable_anomaly_detection=True,
        model_update_interval=9999,
    )

    for cpu, mem in [(10, 10), (11, 11), (9, 9), (10, 9), (9, 10)]:
        monitor.log_resource_usage("agent", cpu=cpu, memory=mem)
    monitor.check_performance()

    storage.store("agent.lifecycle", {"stage": "database", "status": "error"})
    storage.store("agent.lifecycle", {"stage": "database", "status": "error"})

    monitor.log_resource_usage("agent", cpu=95, memory=95)
    monitor.check_performance()

    assert any("Anomalous performance metrics" in subj for subj, _ in alerts)
    assert "database" in alerts[0][1]
