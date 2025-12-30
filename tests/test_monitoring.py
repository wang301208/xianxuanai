import sys
import pathlib
import time
import types

# ensure repository root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

if "psutil" not in sys.modules:
    psutil_stub = types.ModuleType("psutil")

    class _StubProcess:
        def __init__(self, pid: int | None = None) -> None:  # pragma: no cover - stub
            self.pid = pid

        def cpu_times(self):  # pragma: no cover - stub
            return types.SimpleNamespace(user=0.0, system=0.0)

        def memory_percent(self):  # pragma: no cover - stub
            return 0.0

    psutil_stub.Process = _StubProcess
    sys.modules["psutil"] = psutil_stub

if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")

    class _StubFastAPI:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            pass

        def get(self, *args, **kwargs):  # pragma: no cover - stub
            def decorator(func):
                return func

            return decorator

        def post(self, *args, **kwargs):  # pragma: no cover - stub
            def decorator(func):
                return func

            return decorator

    fastapi_stub.FastAPI = _StubFastAPI
    sys.modules["fastapi"] = fastapi_stub

if "matplotlib" not in sys.modules:
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("matplotlib.pyplot")

    def _noop(*args, **kwargs):  # pragma: no cover - stub
        return None

    for name in ("figure", "plot", "show", "close", "subplots", "tight_layout"):
        setattr(pyplot_stub, name, _noop)

    matplotlib_stub.pyplot = pyplot_stub
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub

from backend.monitoring.performance_monitor import PerformanceMonitor
from backend.monitoring.storage import TimeSeriesStorage
from modules.monitoring import RealTimeMetricsCollector, BottleneckDetector


def _make_monitor(tmp_path):
    storage = TimeSeriesStorage(tmp_path / "monitoring.db")
    monitor = PerformanceMonitor(
        storage,
        training_accuracy=1.0,
        degradation_threshold=0.5,
        alert_handlers=[],
    )
    return monitor, storage


def test_bottleneck_detection():
    detector = BottleneckDetector(window_size=5)
    collector = RealTimeMetricsCollector(detector)

    for _ in range(5):
        collector.start("fast")
        time.sleep(0.01)
        event_fast = collector.end("fast")
        collector.start("slow")
        time.sleep(0.02)
        event_slow = collector.end("slow")

    # ensure metrics recorded
    assert event_fast.latency > 0
    assert event_fast.throughput > 0
    assert event_slow.energy >= 0

    bottleneck = detector.bottleneck()
    assert bottleneck is not None
    assert bottleneck[0] == "slow"

    # print dashboard for manual inspection
    collector.print_dashboard()


def test_collector_records_outcomes(tmp_path):
    monitor, storage = _make_monitor(tmp_path)
    collector = RealTimeMetricsCollector(monitor=monitor)

    collector.start("agent")
    collector.end(
        "agent",
        status="success",
        prediction="alpha",
        actual="alpha",
        confidence=0.95,
        metadata={"rating": 0.9},
    )

    collector.start("agent")
    pending_event = collector.end("agent")
    collector.record_outcome(
        pending_event,
        prediction="beta",
        actual="gamma",
        confidence=0.2,
        metadata={"note": "late-outcome"},
    )

    events = collector.events()
    assert events[0].status == "success"
    assert events[0].confidence == 0.95
    assert events[0].metadata.get("rating") == 0.9
    assert events[1].status == "failure"
    assert events[1].stage == "agent"
    assert events[1].metadata.get("note") == "late-outcome"

    feedback_event = collector.emit_event(
        "human_feedback",
        status=None,
        confidence=0.8,
        stage="task-1",
        metadata={"rating": 4},
    )
    assert feedback_event.module == "human_feedback"
    assert feedback_event.stage == "task-1"
    assert feedback_event.metadata.get("rating") == 4

    stored = storage.events("task")
    statuses = [entry.get("status") for entry in stored if entry.get("status")]
    assert sorted(statuses) == ["failure", "success"]
    assert storage.success_rate() == 0.5
