from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: str):
    module_path = ROOT / path
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)

monitoring_pkg = types.ModuleType("backend.monitoring")
monitoring_pkg.__path__ = [str(ROOT / "backend/monitoring")]
sys.modules.setdefault("backend.monitoring", monitoring_pkg)

events_pkg = types.ModuleType("events")
events_pkg.EventBus = type("EventBus", (), {})  # minimal stub
sys.modules.setdefault("events", events_pkg)

common_pkg = types.ModuleType("common")


class _AutoGPTException(Exception):
    pass


def _log_and_format_exception(exc: Exception) -> str:
    return str(exc)


common_pkg.AutoGPTException = _AutoGPTException
common_pkg.log_and_format_exception = _log_and_format_exception
sys.modules.setdefault("common", common_pkg)

calibration_module = _load("backend.monitoring.calibration", "backend/monitoring/calibration.py")
performance_module = _load("backend.monitoring.performance_monitor", "backend/monitoring/performance_monitor.py")

PerformanceCalibrator = calibration_module.PerformanceCalibrator
CalibrationProfile = calibration_module.CalibrationProfile
PerformanceMonitor = performance_module.PerformanceMonitor


class DummyStorage:
    def __init__(self) -> None:
        self._events: dict[str, list[dict]] = {}

    def store(self, category: str, payload: dict) -> None:
        entry = dict(payload)
        entry.setdefault("timestamp", time.time())
        self._events.setdefault(category, []).append(entry)

    def events(self, category: str, start_ts: float | None = None) -> list[dict]:
        events = list(self._events.get(category, []))
        if start_ts is None:
            return events
        return [event for event in events if event.get("timestamp", 0.0) >= start_ts]

    def bottlenecks(self) -> dict[str, float]:
        return {}


def test_calibrator_adjusts_bias():
    state = {"threshold": 0.5}
    calibrator = PerformanceCalibrator()

    def adjust(error: float, _: CalibrationProfile) -> None:
        delta = 0.1 if error > 0 else -0.1
        state["threshold"] = min(1.0, max(0.0, state["threshold"] + delta))

    calibrator.register(
        "attention",
        adjust_fn=adjust,
        tolerance=0.2,
        error_fn=lambda pred, actual, _meta: actual - pred,
        ema_beta=0.0,
    )

    triggered = calibrator.record("attention", prediction=0.4, actual=1.0)
    assert triggered is True
    assert state["threshold"] > 0.5
    assert abs(calibrator.bias("attention") - 0.6) < 1e-6


def test_performance_monitor_invokes_calibration():
    storage = DummyStorage()
    state = {"margin": 1.0}
    calibrator = PerformanceCalibrator()

    def adjust(error: float, _profile: CalibrationProfile) -> None:
        factor = 0.2 if error > 0 else -0.2
        state["margin"] = max(0.0, state["margin"] + factor)

    calibrator.register(
        "navigation",
        adjust_fn=adjust,
        tolerance=2.0,
        error_fn=lambda pred, actual, _meta: actual - pred,
        ema_beta=0.0,
    )

    monitor = PerformanceMonitor(
        storage=storage,
        training_accuracy=0.95,
        degradation_threshold=0.05,
        calibrator=calibrator,
    )

    monitor.log_prediction(
        {"eta": 5.0},
        {"eta": 10.0},
        calibration={"profile": "navigation", "prediction": 5.0, "actual": 10.0},
    )

    assert state["margin"] > 1.0
    assert calibrator.bias("navigation") == 5.0
