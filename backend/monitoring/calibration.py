from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, Optional


ErrorFn = Callable[[Any, Any, Optional[Dict[str, Any]]], float]
AdjustFn = Callable[[float, "CalibrationProfile"], None]


@dataclass
class CalibrationRecord:
    """Captures a single prediction/observation pair."""

    prediction: Any
    actual: Any
    error: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationProfile:
    """Configuration and rolling statistics for a module under calibration."""

    name: str
    tolerance: float
    adjust_fn: AdjustFn
    error_fn: ErrorFn
    ema_beta: float
    max_history: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    ema_error: float = 0.0
    history: Deque[CalibrationRecord] = field(default_factory=deque)
    adjustments: int = 0
    last_error: float = 0.0
    last_update: float = 0.0

    def append(self, record: CalibrationRecord) -> None:
        self.history.append(record)
        while len(self.history) > self.max_history:
            self.history.popleft()
        self.last_error = record.error
        self.last_update = record.timestamp


def _default_error_fn(prediction: Any, actual: Any, _: Optional[Dict[str, Any]]) -> float:
    """Fallback error function supporting numeric values."""

    try:
        pred = float(prediction)
        act = float(actual)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError("prediction and actual must be numeric or provide custom error_fn") from exc
    return act - pred


class PerformanceCalibrator:
    """Track prediction drift and trigger module specific adjustments."""

    def __init__(self) -> None:
        self._profiles: Dict[str, CalibrationProfile] = {}

    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        *,
        adjust_fn: AdjustFn,
        tolerance: float = 0.1,
        error_fn: Optional[ErrorFn] = None,
        ema_beta: float = 0.7,
        max_history: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CalibrationProfile:
        """Register a module for calibration."""

        profile = CalibrationProfile(
            name=name,
            tolerance=float(tolerance),
            adjust_fn=adjust_fn,
            error_fn=error_fn or _default_error_fn,
            ema_beta=float(ema_beta),
            max_history=int(max_history),
            metadata=dict(metadata or {}),
        )
        self._profiles[name] = profile
        return profile

    def unregister(self, name: str) -> None:
        """Remove a previously registered profile."""

        self._profiles.pop(name, None)

    def profiles(self) -> Iterable[str]:
        """Return the names of all registered profiles."""

        return tuple(self._profiles)

    # ------------------------------------------------------------------
    def record(
        self,
        name: str,
        *,
        prediction: Any,
        actual: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record a new sample for ``name`` and trigger adjustment if needed."""

        if name not in self._profiles:
            raise KeyError(f"Unknown calibration profile '{name}'")
        profile = self._profiles[name]
        info = dict(metadata or {})

        error = profile.error_fn(prediction, actual, info)
        record = CalibrationRecord(
            prediction=prediction,
            actual=actual,
            error=error,
            timestamp=time.time(),
            metadata=info,
        )
        profile.append(record)

        if profile.adjustments == 0 and len(profile.history) == 1:
            profile.ema_error = error
        else:
            profile.ema_error = profile.ema_beta * profile.ema_error + (1 - profile.ema_beta) * error

        should_adjust = abs(profile.ema_error) > profile.tolerance
        if should_adjust:
            profile.adjust_fn(profile.ema_error, profile)
            profile.adjustments += 1
        return should_adjust

    # ------------------------------------------------------------------
    def bias(self, name: str) -> float:
        """Return the current exponential moving average error."""

        profile = self._profiles[name]
        return profile.ema_error

    def history(self, name: str) -> Deque[CalibrationRecord]:
        """Return the recorded history for ``name``."""

        profile = self._profiles[name]
        return profile.history
