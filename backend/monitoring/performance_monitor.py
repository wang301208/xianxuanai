"""Utilities for tracking model performance in production."""

from __future__ import annotations

import importlib.util
import smtplib
import sys
import time
import tracemalloc
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover - fallback when sklearn missing
    IsolationForest = None  # type: ignore[misc,assignment]

from .storage import TimeSeriesStorage

# ``performance_monitor`` is imported in the tests using a synthetic module
# hierarchy (``monitoring.performance_monitor``).  When that happens the parent
# ``monitoring`` module is just a simple ``ModuleType`` instance without a
# ``__path__`` attribute, so relative imports such as ``from .calibration``
# raise ``ModuleNotFoundError`` with the message "'monitoring' is not a
# package".  Importing the module directly from the repository package keeps
# the implementation flexible while allowing the lightweight test harness to
# work.  Falling back to loading the sibling ``calibration`` module from its
# file path avoids importing ``backend.monitoring`` (which pulls optional
# dependencies like FastAPI) while still providing the required classes.
try:  # pragma: no cover - exercised indirectly through tests
    from .calibration import CalibrationProfile, PerformanceCalibrator
except ModuleNotFoundError:  # pragma: no cover - used in the stub test harness
    _calibration_path = Path(__file__).with_name("calibration.py")
    _module_name = "backend.monitoring._calibration_fallback"
    _spec = importlib.util.spec_from_file_location(_module_name, _calibration_path)
    if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
        raise
    _calibration_module = importlib.util.module_from_spec(_spec)
    sys.modules.setdefault(_module_name, _calibration_module)
    _spec.loader.exec_module(_calibration_module)
    CalibrationProfile = _calibration_module.CalibrationProfile  # type: ignore[attr-defined]
    PerformanceCalibrator = _calibration_module.PerformanceCalibrator  # type: ignore[attr-defined]
# ``common`` historically lived at the repository root, but newer module
# layouts place the shared helpers under ``modules.common``.  Tests import the
# monitoring package directly (without installing the repository as a package),
# so the simplest approach is to try the legacy location first and gracefully
# fall back to the modular namespace when it is not available.
try:  # pragma: no cover - exercised indirectly when the legacy import works
    from common import AutoGPTException, log_and_format_exception
except ModuleNotFoundError:  # pragma: no cover - used in the lightweight test harness
    from modules.common import AutoGPTException, log_and_format_exception
from smtplib import SMTPException


class SpikeRateMonitor:
    """Track spike rates for agents and store them in the time-series storage."""

    def __init__(self, storage: TimeSeriesStorage) -> None:
        self.storage = storage

    def log(self, agent: str, rate: float) -> None:
        self.storage.store("agent.spike", {"agent": agent, "rate": rate})

    def average(self, agent: str | None = None, interval: float = 60.0) -> float:
        start = time.time() - interval
        events = self.storage.events("agent.spike", start_ts=start)
        samples = [e for e in events if agent is None or e.get("agent") == agent]
        if not samples:
            return 0.0
        return sum(float(e.get("rate", 0.0)) for e in samples) / len(samples)


class EnergyConsumptionMonitor:
    """Track energy consumption for agents."""

    def __init__(self, storage: TimeSeriesStorage) -> None:
        self.storage = storage

    def log(self, agent: str, energy: float) -> None:
        self.storage.store("agent.energy", {"agent": agent, "energy": energy})

    def average(self, agent: str | None = None, interval: float = 60.0) -> float:
        start = time.time() - interval
        events = self.storage.events("agent.energy", start_ts=start)
        samples = [e for e in events if agent is None or e.get("agent") == agent]
        if not samples:
            return 0.0
        return sum(float(e.get("energy", 0.0)) for e in samples) / len(samples)

AlertHandler = Callable[[str, str], None]


class PerformanceMonitor:
    """Monitor predictions against outcomes and trigger alerts on degradation."""

    def __init__(
        self,
        storage: TimeSeriesStorage,
        training_accuracy: float,
        degradation_threshold: float,
        alert_handlers: Iterable[AlertHandler] | None = None,
        cpu_threshold: float | None = None,
        memory_threshold: float | None = None,
        throughput_threshold: float | None = None,
        spike_rate_threshold: float | None = None,
        energy_threshold: float | None = None,
        memory_leak_threshold: float | None = None,
        *,
        enable_anomaly_detection: bool = False,
        model_update_interval: float = 3600.0,
        contamination: float = 0.05,
        calibrator: PerformanceCalibrator | None = None,
    ) -> None:
        self.storage = storage
        self.training_accuracy = training_accuracy
        self.degradation_threshold = degradation_threshold
        self.alert_handlers: List[AlertHandler] = list(alert_handlers or [])
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.throughput_threshold = throughput_threshold
        self.spike_rate_threshold = spike_rate_threshold
        self.energy_threshold = energy_threshold
        self.memory_leak_threshold = memory_leak_threshold
        self.enable_anomaly_detection = enable_anomaly_detection
        self.model_update_interval = model_update_interval
        self.contamination = contamination
        self._anomaly_model: IsolationForest | None = None
        self._last_model_update = 0.0
        self.calibrator: PerformanceCalibrator | None = calibrator

        # monitors for additional metrics
        self.spike_monitor = SpikeRateMonitor(storage)
        self.energy_monitor = EnergyConsumptionMonitor(storage)

        # baseline snapshot for memory leak detection
        tracemalloc.start()
        self._baseline_snapshot = tracemalloc.take_snapshot()

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def _record_calibration(
        self,
        calibration: Optional[Dict[str, Any]],
        prediction: Any,
        outcome: Any,
    ) -> None:
        if not calibration or not self.calibrator:
            return
        profile = calibration.get("profile")
        if not profile:
            return
        pred_value = calibration.get("prediction", prediction)
        actual_value = calibration.get("actual", outcome)
        metadata = calibration.get("metadata")
        try:
            self.calibrator.record(
                profile,
                prediction=pred_value,
                actual=actual_value,
                metadata=metadata,
            )
        except KeyError:
            # Profile not registered; ignore silently so logging still succeeds.
            pass

    # ------------------------------------------------------------------
    def log_prediction(
        self,
        prediction: Any,
        outcome: Any,
        *,
        calibration: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist *prediction* and corresponding *outcome*."""
        status = "success" if prediction == outcome else "failure"
        event = {"prediction": prediction, "outcome": outcome, "status": status}
        self.storage.store("prediction", event)
        self._record_calibration(calibration, prediction, outcome)

    # ------------------------------------------------------------------
    def log_task_result(
        self,
        agent: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        prediction: Any | None = None,
        actual: Any | None = None,
        confidence: Optional[float] = None,
        calibration: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist task-level outcome metrics for ``agent``."""

        resolved_status = status
        if resolved_status is None and prediction is not None and actual is not None:
            resolved_status = "success" if prediction == actual else "failure"
        if resolved_status is None:
            resolved_status = "unknown"

        payload: Dict[str, Any] = {
            "agent": agent,
            "stage": stage or agent,
            "status": resolved_status,
        }
        if prediction is not None:
            payload["prediction"] = prediction
        if actual is not None:
            payload["actual"] = actual
            payload["outcome"] = actual
        if confidence is not None:
            payload["confidence"] = confidence

        self.storage.store("task", payload)

        if prediction is not None and actual is not None:
            self.log_prediction(prediction, actual, calibration=calibration)

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    def current_accuracy(self) -> float:
        """Return accuracy computed from logged predictions."""
        events = self.storage.events("prediction")
        if not events:
            return 0.0
        correct = sum(1 for e in events if e.get("prediction") == e.get("outcome"))
        return correct / len(events)

    # ------------------------------------------------------------------
    # resource metrics
    # ------------------------------------------------------------------
    def log_resource_usage(self, agent: str, cpu: float, memory: float) -> None:
        """Persist resource usage for *agent*."""
        self.storage.store("agent.resource", {"agent": agent, "cpu": cpu, "memory": memory})

    def log_spike_rate(self, agent: str, rate: float) -> None:
        """Persist spike rate for *agent*."""
        self.spike_monitor.log(agent, rate)

    def log_energy_consumption(self, agent: str, energy: float) -> None:
        """Persist energy consumption for *agent*."""
        self.energy_monitor.log(agent, energy)

    def log_task_completion(self, agent: str) -> None:
        """Record completion of a task by *agent*."""
        self.storage.store("task", {"agent": agent})

    def cpu_usage(self, agent: str | None = None, interval: float = 60.0) -> float:
        """Return average CPU usage for *agent* over *interval* seconds."""
        start = time.time() - interval
        events = self.storage.events("agent.resource", start_ts=start)
        samples = [e for e in events if agent is None or e.get("agent") == agent]
        if not samples:
            return 0.0
        return sum(float(e.get("cpu", 0.0)) for e in samples) / len(samples)

    def memory_usage(self, agent: str | None = None, interval: float = 60.0) -> float:
        """Return average memory usage for *agent* over *interval* seconds."""
        start = time.time() - interval
        events = self.storage.events("agent.resource", start_ts=start)
        samples = [e for e in events if agent is None or e.get("agent") == agent]
        if not samples:
            return 0.0
        return sum(float(e.get("memory", 0.0)) for e in samples) / len(samples)

    def task_throughput(self, agent: str | None = None, interval: float = 60.0) -> float:
        """Return tasks completed per second for *agent* over *interval* seconds."""
        start = time.time() - interval
        events = self.storage.events("task", start_ts=start)
        count = sum(1 for e in events if agent is None or e.get("agent") == agent)
        return count / interval if interval > 0 else 0.0

    def spike_rate(self, agent: str | None = None, interval: float = 60.0) -> float:
        """Return average spike rate for *agent* over *interval* seconds."""
        return self.spike_monitor.average(agent, interval)

    def energy_consumption(self, agent: str | None = None, interval: float = 60.0) -> float:
        """Return average energy consumption for *agent* over *interval* seconds."""
        return self.energy_monitor.average(agent, interval)

    def _resource_samples(self) -> list[list[float]]:
        events = self.storage.events("agent.resource")
        return [
            [float(e.get("cpu", 0.0)), float(e.get("memory", 0.0))]
            for e in events
        ]

    def _latest_resource_sample(self) -> list[float] | None:
        events = self.storage.events("agent.resource")
        if events:
            e = events[-1]
            return [float(e.get("cpu", 0.0)), float(e.get("memory", 0.0))]
        return None

    def _maybe_update_model(self) -> None:
        if IsolationForest is None:  # pragma: no cover - optional dependency
            return
        now = time.time()
        if self._anomaly_model is None or now - self._last_model_update > self.model_update_interval:
            data = self._resource_samples()
            if data:
                self._anomaly_model = IsolationForest(contamination=self.contamination)
                self._anomaly_model.fit(data)
                self._last_model_update = now

    def check_performance(self) -> None:
        """Compare live metrics against thresholds and alert on degradation."""
        accuracy = self.current_accuracy()
        allowed_drop = self.training_accuracy - self.degradation_threshold
        if accuracy < allowed_drop:
            self._alert(
                "Model performance degraded",
                f"Accuracy {accuracy:.2%} below threshold {allowed_drop:.2%}",
            )

        # Check resource utilization
        if self.cpu_threshold is not None:
            cpu = self.cpu_usage()
            if cpu > self.cpu_threshold:
                self._alert(
                    "CPU usage high",
                    f"Average CPU usage {cpu:.2f}% exceeds {self.cpu_threshold:.2f}%",
                )
        if self.memory_threshold is not None:
            mem = self.memory_usage()
            if mem > self.memory_threshold:
                self._alert(
                    "Memory usage high",
                    f"Average memory usage {mem:.2f}% exceeds {self.memory_threshold:.2f}%",
                )

        if self.spike_rate_threshold is not None:
            sr = self.spike_rate()
            if sr > self.spike_rate_threshold:
                self._alert(
                    "Spike rate high",
                    f"Average spike rate {sr:.2f} exceeds {self.spike_rate_threshold:.2f}",
                )

        if self.energy_threshold is not None:
            energy = self.energy_consumption()
            if energy > self.energy_threshold:
                self._alert(
                    "Energy consumption high",
                    f"Average energy {energy:.2f} exceeds {self.energy_threshold:.2f}",
                )

        if self.memory_leak_threshold is not None:
            current = tracemalloc.take_snapshot()
            stats = current.compare_to(self._baseline_snapshot, "lineno")
            leak = sum(stat.size_diff for stat in stats)
            self.storage.store("memory.sample", {"leak": leak})
            if leak > self.memory_leak_threshold:
                self._alert(
                    "Potential memory leak",
                    f"Memory usage increased by {leak/1024:.2f} KiB over baseline",
                )

        if self.throughput_threshold is not None:
            tp = self.task_throughput()
            if tp < self.throughput_threshold:
                self._alert(
                    "Task throughput low",
                    f"Throughput {tp:.2f} tasks/sec below {self.throughput_threshold:.2f}",
                )

        if self.enable_anomaly_detection and IsolationForest is not None:
            self._maybe_update_model()
            sample = self._latest_resource_sample()
            if sample and self._anomaly_model is not None:
                pred = self._anomaly_model.predict([sample])[0]
                if pred == -1:
                    bottlenecks = self.storage.bottlenecks()
                    component = max(bottlenecks, key=bottlenecks.get) if bottlenecks else "unknown"
                    self._alert(
                        "Anomalous performance metrics",
                        f"Metrics {sample} flagged as anomalous. Potential bottleneck: {component}",
                    )

    # ------------------------------------------------------------------
    def register_calibration_profile(
        self,
        name: str,
        *,
        adjust_fn: Callable[[float, CalibrationProfile], None],
        tolerance: float = 0.1,
        error_fn: Callable[[Any, Any, Optional[Dict[str, Any]]], float] | None = None,
        ema_beta: float = 0.7,
        max_history: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a calibration profile and ensure a calibrator exists."""

        if self.calibrator is None:
            self.calibrator = PerformanceCalibrator()
        self.calibrator.register(
            name,
            adjust_fn=adjust_fn,
            tolerance=tolerance,
            error_fn=error_fn,
            ema_beta=ema_beta,
            max_history=max_history,
            metadata=metadata,
        )

    def calibration_bias(self, name: str) -> float:
        """Return the current calibration bias for a profile."""

        if self.calibrator is None:
            raise ValueError("No calibrator configured")
        return self.calibrator.bias(name)


    # ------------------------------------------------------------------
    # alerting
    # ------------------------------------------------------------------
    def _alert(self, subject: str, message: str) -> None:
        for handler in self.alert_handlers:
            try:
                handler(subject, message)
            except AutoGPTException as err:
                log_and_format_exception(err)
            except SMTPException as err:
                log_and_format_exception(err)
            except Exception as err:  # pragma: no cover - unexpected
                log_and_format_exception(err)


def email_alert(
    to_address: str,
    smtp_server: str = "localhost",
    smtp_port: int = 25,
    from_address: str = "noreply@example.com",
) -> AlertHandler:
    """Return an alert handler that sends email notifications."""

    def send(subject: str, message: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_address
        msg["To"] = to_address
        msg.set_content(message)
        with smtplib.SMTP(smtp_server, smtp_port) as smtp:
            smtp.send_message(msg)

    return send


def dashboard_alert(logger: Callable[[str], None] = print) -> AlertHandler:
    """Return an alert handler that logs notifications for dashboards."""

    def send(subject: str, message: str) -> None:
        logger(f"{subject}: {message}")

    return send
