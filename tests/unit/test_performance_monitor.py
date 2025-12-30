import sys
import types

# Stub modules required by performance_monitor
events_mod = types.ModuleType("events")
class EventBus: ...
events_mod.EventBus = EventBus
sys.modules.setdefault("events", events_mod)

common_mod = types.ModuleType("common")
class AutoGPTException(Exception):
    pass
def log_and_format_exception(e):
    pass
common_mod.AutoGPTException = AutoGPTException
common_mod.log_and_format_exception = log_and_format_exception
sys.modules.setdefault("common", common_mod)

sk_module = types.ModuleType("sklearn")
ensemble_mod = types.ModuleType("sklearn.ensemble")
class IsolationForest:
    def __init__(self, *a, **k):
        pass
    def fit(self, data):
        return self
    def predict(self, X):
        return [1 for _ in X]
sk_module.ensemble = ensemble_mod
ensemble_mod.IsolationForest = IsolationForest
sys.modules.setdefault("sklearn", sk_module)
sys.modules.setdefault("sklearn.ensemble", ensemble_mod)
sys.modules.setdefault("psutil", types.ModuleType("psutil"))

from pathlib import Path
import importlib.util
import types

ROOT = Path(__file__).resolve().parents[2]

monitoring_pkg = types.ModuleType("monitoring")
sys.modules.setdefault("monitoring", monitoring_pkg)

spec_storage = importlib.util.spec_from_file_location(
    "monitoring.storage", ROOT / "backend/monitoring/storage.py"
)
storage_module = importlib.util.module_from_spec(spec_storage)
sys.modules["monitoring.storage"] = storage_module
spec_storage.loader.exec_module(storage_module)
TimeSeriesStorage = storage_module.TimeSeriesStorage

spec_pm = importlib.util.spec_from_file_location(
    "monitoring.performance_monitor", ROOT / "backend/monitoring/performance_monitor.py"
)
pm_module = importlib.util.module_from_spec(spec_pm)
sys.modules["monitoring.performance_monitor"] = pm_module
spec_pm.loader.exec_module(pm_module)
PerformanceMonitor = pm_module.PerformanceMonitor
SpikeRateMonitor = pm_module.SpikeRateMonitor


def test_spike_rate_monitor_average(tmp_path):
    storage = TimeSeriesStorage(":memory:")
    monitor = SpikeRateMonitor(storage)
    monitor.log("a1", 2.0)
    monitor.log("a1", 4.0)
    assert monitor.average("a1") == 3.0


def test_performance_monitor_alert_on_degradation(tmp_path):
    storage = TimeSeriesStorage(":memory:")
    alerts = []

    def handler(subject, message):
        alerts.append((subject, message))

    monitor = PerformanceMonitor(
        storage,
        training_accuracy=0.9,
        degradation_threshold=0.1,
        alert_handlers=[handler],
    )
    monitor.log_prediction(1, 1)
    monitor.log_prediction(1, 0)
    monitor.check_performance()
    assert any("Model performance degraded" in s for s, _ in alerts)
