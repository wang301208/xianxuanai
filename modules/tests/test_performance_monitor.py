import sys, os, types, importlib.util
sys.path.insert(0, os.path.abspath(os.getcwd()))
sys.modules.setdefault("events", types.SimpleNamespace(EventBus=object))

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
pkg = types.ModuleType("monitoring")
pkg.__path__ = [str(ROOT / "backend" / "monitoring")]
sys.modules.setdefault("monitoring", pkg)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


performance_monitor = load_module(
    "monitoring.performance_monitor", ROOT / "backend" / "monitoring" / "performance_monitor.py"
)
storage_module = load_module(
    "monitoring.storage", ROOT / "backend" / "monitoring" / "storage.py"
)

PerformanceMonitor = performance_monitor.PerformanceMonitor
TimeSeriesStorage = storage_module.TimeSeriesStorage
from common import AutoGPTException


def test_performance_monitor_resource_alert(tmp_path: Path) -> None:
    storage = TimeSeriesStorage(tmp_path / "monitoring.db")
    alerts: list[tuple[str, str]] = []

    def handler(subj: str, msg: str) -> None:
        alerts.append((subj, msg))

    monitor = PerformanceMonitor(
        storage,
        training_accuracy=1.0,
        degradation_threshold=0.1,
        alert_handlers=[handler],
        cpu_threshold=50.0,
    )
    monitor.log_resource_usage("agent", 90.0, 10.0)
    monitor.check_performance()
    assert any("CPU" in a[0] for a in alerts)


def test_alert_handler_exception_logged(tmp_path: Path, caplog) -> None:
    storage = TimeSeriesStorage(tmp_path / "monitoring.db")

    def handler(subj: str, msg: str) -> None:
        raise AutoGPTException("boom")

    monitor = PerformanceMonitor(
        storage,
        training_accuracy=1.0,
        degradation_threshold=0.1,
        alert_handlers=[handler],
        cpu_threshold=50.0,
    )
    monitor.log_resource_usage("agent", 90.0, 10.0)
    with caplog.at_level("ERROR"):
        monitor.check_performance()
    assert any("AutoGPTException" in r.message for r in caplog.records)


def test_extra_metrics_and_memory_leak(tmp_path: Path) -> None:
    storage = TimeSeriesStorage(tmp_path / "monitoring.db")
    alerts: list[tuple[str, str]] = []

    def handler(subj: str, msg: str) -> None:
        alerts.append((subj, msg))

    monitor = PerformanceMonitor(
        storage,
        training_accuracy=1.0,
        degradation_threshold=0.1,
        alert_handlers=[handler],
        spike_rate_threshold=5.0,
        energy_threshold=10.0,
        memory_leak_threshold=1.0,
    )

    monitor.log_spike_rate("agent", 10.0)
    monitor.log_energy_consumption("agent", 20.0)

    # allocate memory to exceed leak threshold
    leak = ["x" * 1024 for _ in range(1024)]
    monitor.check_performance()

    assert any("Spike rate high" in a[0] for a in alerts)
    assert any("Energy consumption high" in a[0] for a in alerts)
    assert any("Potential memory leak" in a[0] for a in alerts)

    # prevent the allocated memory from being optimized away
    assert leak
