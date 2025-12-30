import importlib
import os
import sys
import types
from pathlib import Path

# prepare module paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sys.modules.setdefault("events", types.SimpleNamespace(EventBus=object))
sys.modules.setdefault(
    "common", types.SimpleNamespace(AutoGPTException=Exception, log_and_format_exception=lambda err: None)
)

monitoring_dir = Path(__file__).resolve().parent.parent / "monitoring"
monitoring_pkg = types.ModuleType("monitoring")
monitoring_pkg.__path__ = [str(monitoring_dir)]
sys.modules["monitoring"] = monitoring_pkg

auto_opt_mod = importlib.import_module("monitoring.auto_optimizer")
perf_mod = importlib.import_module("monitoring.performance_monitor")
storage_mod = importlib.import_module("monitoring.storage")

AutoOptimizer = auto_opt_mod.AutoOptimizer
PerformanceMonitor = perf_mod.PerformanceMonitor
TimeSeriesStorage = storage_mod.TimeSeriesStorage


def test_auto_optimizer_adjusts_and_logs(tmp_path: Path) -> None:
    storage = TimeSeriesStorage(tmp_path / "opt.db")
    monitor = PerformanceMonitor(storage, training_accuracy=1.0, degradation_threshold=0.1)
    monitor.log_resource_usage("agent", cpu=90.0, memory=10.0)
    optimizer = AutoOptimizer(monitor, storage, cpu_threshold=80.0)

    optimizer.step()

    assert optimizer.resource_allocation["cpu_limit"] > 1.0
    events = storage.events("optimization")
    assert events
    assert any(e.get("metric") == "cpu" for e in events)
