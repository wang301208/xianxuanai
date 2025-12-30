from __future__ import annotations

"""Real-time metrics collection utilities."""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Dict, List, Optional
import time

from backend.monitoring import PerformanceMonitor

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil optional
    psutil = None  # type: ignore

from .bottleneck import BottleneckDetector


@dataclass
class MetricEvent:
    """Container for metrics from a single operation."""

    module: str
    latency: float
    energy: float
    throughput: float
    timestamp: float
    status: Optional[str] = None
    prediction: Any | None = None
    actual: Any | None = None
    confidence: Optional[float] = None
    stage: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealTimeMetricsCollector:
    """Collects latency, energy consumption, and throughput in real time.

    The collector can be used as::

        collector.start("module")
        ... do work ...
        event = collector.end("module")
    """

    def __init__(
        self,
        detector: Optional[BottleneckDetector] = None,
        monitor: PerformanceMonitor | None = None,
    ) -> None:
        self._detector = detector
        self._monitor = monitor
        self._events: List[MetricEvent] = []
        self._starts: Dict[str, tuple[float, float]] = {}
        self._counts: Dict[str, int] = defaultdict(int)
        self._process = psutil.Process() if psutil else None

    # ------------------------------------------------------------------
    def start(self, module: str) -> None:
        """Mark the start of an operation for ``module``."""
        start_time = time.perf_counter()
        start_cpu = 0.0
        if self._process is not None:
            cpu = self._process.cpu_times()
            start_cpu = cpu.user + cpu.system
        self._starts[module] = (start_time, start_cpu)

    # ------------------------------------------------------------------
    def end(
        self,
        module: str,
        items: int = 1,
        *,
        status: Optional[str] = None,
        prediction: Any | None = None,
        actual: Any | None = None,
        confidence: Optional[float] = None,
        stage: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MetricEvent:
        """Finish an operation for ``module`` and record metrics."""
        start_time, start_cpu = self._starts.pop(module, (time.perf_counter(), 0.0))
        end_time = time.perf_counter()
        latency = end_time - start_time

        energy = 0.0
        cpu_percent = 0.0
        mem_percent = 0.0
        if self._process is not None:
            cpu = self._process.cpu_times()
            end_cpu = cpu.user + cpu.system
            energy = max(end_cpu - start_cpu, 0.0)
            if latency > 0:
                cpu_percent = max((end_cpu - start_cpu) / latency * 100.0, 0.0)
            mem_percent = self._process.memory_percent()

        self._counts[module] += items
        throughput = 0.0
        if latency > 0:
            throughput = items / latency

        if status is None and prediction is not None and actual is not None:
            status = "success" if prediction == actual else "failure"

        event = MetricEvent(
            module=module,
            latency=latency,
            energy=energy,
            throughput=throughput,
            timestamp=end_time,
            status=status,
            prediction=prediction,
            actual=actual,
            confidence=confidence,
            stage=stage or module,
            metadata=dict(metadata or {}),
        )
        self._events.append(event)

        if self._detector is not None:
            self._detector.record(module, latency)

        if self._monitor is not None:
            self._monitor.log_resource_usage(module, cpu_percent, mem_percent)
            if event.status is not None or event.prediction is not None or event.actual is not None:
                self._log_outcome(event)
            else:
                self._monitor.log_task_completion(module)

        return event

    # ------------------------------------------------------------------
    def record_outcome(
        self,
        event: MetricEvent,
        *,
        status: Optional[str] = None,
        prediction: Any | None = None,
        actual: Any | None = None,
        confidence: Optional[float] = None,
        stage: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MetricEvent:
        """Attach outcome data to an existing ``event`` and notify monitors."""

        if status is not None:
            event.status = status
        if prediction is not None:
            event.prediction = prediction
        if actual is not None:
            event.actual = actual
        if confidence is not None:
            event.confidence = confidence
        if stage is not None:
            event.stage = stage
        if metadata:
            try:
                event.metadata.update({str(k): v for k, v in dict(metadata).items()})
            except Exception:
                pass

        if event.status is None and event.prediction is not None and event.actual is not None:
            event.status = "success" if event.prediction == event.actual else "failure"
        if event.stage is None:
            event.stage = event.module

        if self._monitor is not None and (
            event.status is not None
            or event.prediction is not None
            or event.actual is not None
            or event.confidence is not None
        ):
            self._log_outcome(event)

        return event

    # ------------------------------------------------------------------
    def emit_event(
        self,
        module: str,
        *,
        latency: float = 0.0,
        energy: float = 0.0,
        throughput: float = 0.0,
        status: Optional[str] = None,
        prediction: Any | None = None,
        actual: Any | None = None,
        confidence: Optional[float] = None,
        stage: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MetricEvent:
        """Append a synthetic metric event (e.g., user feedback, benchmark result).

        This helper is useful when you already have metrics computed externally
        and want them to participate in the same evolution/diagnostics pipeline
        as `start()` / `end()` events.
        """

        timestamp = time.perf_counter()
        event = MetricEvent(
            module=str(module),
            latency=float(latency),
            energy=float(energy),
            throughput=float(throughput),
            timestamp=timestamp,
            status=status,
            prediction=prediction,
            actual=actual,
            confidence=confidence,
            stage=stage or str(module),
            metadata=dict(metadata or {}),
        )
        self._events.append(event)
        if self._detector is not None:
            try:
                self._detector.record(str(module), float(latency))
            except Exception:
                pass

        if self._monitor is not None:
            if event.status is not None or event.prediction is not None or event.actual is not None or event.confidence is not None:
                self._log_outcome(event)
            else:
                self._monitor.log_task_completion(event.module)
        return event

    # ------------------------------------------------------------------
    def _log_outcome(self, event: MetricEvent) -> None:
        if self._monitor is None:
            return

        status = event.status
        if status is None and event.prediction is not None and event.actual is not None:
            status = "success" if event.prediction == event.actual else "failure"
            event.status = status

        stage = event.stage or event.module

        if status is not None and hasattr(self._monitor, "log_task_result"):
            self._monitor.log_task_result(
                event.module,
                status=status,
                stage=stage,
                prediction=event.prediction,
                actual=event.actual,
                confidence=event.confidence,
            )
        elif event.prediction is not None and event.actual is not None:
            # Fallback to prediction logging when the monitor lacks task result support
            self._monitor.log_prediction(event.prediction, event.actual)
        else:
            self._monitor.log_task_completion(event.module)

    # ------------------------------------------------------------------
    def events(self) -> List[MetricEvent]:
        """Return all recorded metric events."""
        return list(self._events)

    # ------------------------------------------------------------------
    def print_dashboard(self) -> None:
        """Print a simple dashboard with average metrics per module."""
        if not self._events:
            print("No metrics collected yet")
            return

        stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: Dict[str, int] = defaultdict(int)
        for event in self._events:
            stats[event.module]["latency"] += event.latency
            stats[event.module]["energy"] += event.energy
            stats[event.module]["throughput"] += event.throughput
            counts[event.module] += 1

        header = f"{'Module':<20}{'Avg Latency':>15}{'Avg Energy':>15}{'Avg Thpt':>15}"
        print(header)
        print("-" * len(header))
        for module, s in stats.items():
            n = counts[module]
            line = (
                f"{module:<20}"
                f"{s['latency']/n:>15.4f}"
                f"{s['energy']/n:>15.4f}"
                f"{s['throughput']/n:>15.4f}"
            )
            print(line)
