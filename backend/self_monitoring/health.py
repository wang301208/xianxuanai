from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING, Any

from .monitor import SelfMonitoringSystem, StepReport, RecoveryDecision
from .remediation import RemediationManager, RemediationResult

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .diagnostics import DiagnosticReport, SystemDiagnostics


SensorFn = Callable[[], "SensorReading"]


@dataclass
class SensorReading:
    """Measurement produced by a health sensor."""

    name: str
    value: Optional[float] = None
    threshold: Optional[float | Tuple[float, float]] = None
    status: str = "ok"
    category: str = "generic"
    units: Optional[str] = None
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    comparison: str = "gt"
    timestamp: float = field(default_factory=lambda: time.time())

    def is_anomalous(self) -> bool:
        """Return ``True`` if the reading indicates an anomaly."""

        status_flag = self.status.lower()
        if status_flag not in {"ok", "nominal", "healthy"}:
            return True
        if self.value is None or self.threshold is None:
            return False

        if isinstance(self.threshold, tuple):
            lo, hi = self.threshold
            if self.comparison == "between":
                return not (lo <= self.value <= hi)
            return self.value < lo or self.value > hi

        if self.comparison == "lt":
            return self.value < float(self.threshold)
        if self.comparison == "between":
            lo, hi = float(self.threshold), float(self.metadata.get("upper", self.threshold))
            return not (lo <= self.value <= hi)
        return self.value > float(self.threshold)

    def summary(self) -> str:
        """Return a short human readable summary."""

        parts = [f"{self.name}={self.value!r}"]
        if self.units:
            parts.append(self.units)
        if self.threshold is not None:
            parts.append(f"threshold={self.threshold!r}")
        if self.status:
            parts.append(f"status={self.status}")
        if self.message:
            parts.append(f"msg={self.message}")
        return "; ".join(parts)


@dataclass
class HealthAlert:
    """Outcome of an anomalous reading handled by :class:`HealthMonitor`."""

    sensor: str
    reading: SensorReading
    decision: Optional[RecoveryDecision] = None
    diagnostic: Optional["DiagnosticReport"] = None
    recorded_at: float = field(default_factory=lambda: time.time())
    remediation: Optional[RemediationResult] = None


class HealthMonitor:
    """Monitor critical internal metrics and trigger recovery workflows."""

    def __init__(
        self,
        *,
        self_monitor: Optional[SelfMonitoringSystem] = None,
        diagnostics: Optional["SystemDiagnostics"] = None,
        remediator: Optional[RemediationManager] = None,
    ) -> None:
        self._sensors: Dict[str, SensorFn] = {}
        self._history: List[HealthAlert] = []
        self._self_monitor = self_monitor
        self._diagnostics = diagnostics
        self._remediator = remediator

    # ------------------------------------------------------------------
    def register_sensor(self, name: str, sensor: SensorFn) -> None:
        """Register a callable returning :class:`SensorReading`."""

        if not callable(sensor):
            raise TypeError("sensor must be callable")
        self._sensors[name] = sensor

    def remove_sensor(self, name: str) -> None:
        """Remove a previously registered sensor."""

        self._sensors.pop(name, None)

    def sensors(self) -> List[str]:
        """Return the list of registered sensor names."""

        return list(self._sensors)

    # ------------------------------------------------------------------
    def evaluate(self, *, names: Optional[Iterable[str]] = None) -> List[HealthAlert]:
        """Collect readings, run self-monitoring, and trigger diagnostics."""

        alerts: List[HealthAlert] = []
        targets = list(names) if names is not None else list(self._sensors)

        for name in targets:
            sensor = self._sensors.get(name)
            if sensor is None:
                continue
            reading = sensor()
            if reading.name != name:
                reading.name = name
            if not reading.is_anomalous():
                continue

            decision: Optional[RecoveryDecision] = None
            if self._self_monitor is not None:
                metrics: Dict[str, float] = {}
                if isinstance(reading.value, (int, float)):
                    metrics["value"] = float(reading.value)
                if isinstance(reading.threshold, (int, float)):
                    metrics["threshold"] = float(reading.threshold)

                report = StepReport(
                    action=f"sensor:{name}",
                    observation=reading.message or reading.summary(),
                    status=reading.status or "alert",
                    metrics=metrics,
                    error=reading.message if reading.status.lower() in {"error", "critical"} else None,
                    metadata={
                        "category": reading.category,
                        "units": reading.units,
                        "sensor_metadata": reading.metadata,
                    },
                )
                decision = self._self_monitor.assess_step(report)

            diagnostic = None
            if self._diagnostics is not None:
                try:
                    diagnostic = self._diagnostics.diagnose(name, reading, decision)
                except Exception as exc:  # pragma: no cover - defensive
                    diagnostic = None
                    metrics = {
                        "sensor": name,
                        "error": str(exc),
                    }
                    if self._self_monitor is not None:
                        report = StepReport(
                            action=f"sensor:{name}:diagnostics",
                            observation=str(exc),
                            status="error",
                            metrics=metrics,
                            error=str(exc),
                        )
                        self._self_monitor.assess_step(report)

            remediation = None
            if self._remediator is not None:
                module_name = (
                    reading.metadata.get("module")
                    or reading.metadata.get("component")
                    or reading.metadata.get("service")
                    or name
                )
                allow_patch = bool(reading.metadata.get("allow_patch", False))
                remediation = self._remediator.attempt(
                    module_name,
                    {
                        "sensor": name,
                        "reading": reading,
                        "decision": decision,
                        "diagnostic": diagnostic,
                    },
                    allow_code_patch=allow_patch,
                )

            alert = HealthAlert(
                sensor=name,
                reading=reading,
                decision=decision,
                diagnostic=diagnostic,
                remediation=remediation,
            )
            alerts.append(alert)
            self._history.append(alert)

        return alerts

    def history(self) -> List[HealthAlert]:
        """Return previous alerts."""

        return list(self._history)
