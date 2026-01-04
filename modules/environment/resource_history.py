from __future__ import annotations

"""Helpers for aggregating multi-node resource signals over a short time window."""

import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Iterable, Optional, Tuple

from modules.events import EventBus

from .registry import get_hardware_registry, subscribe_resource_signals

Sample = Tuple[float, Dict[str, float]]


class ResourceSignalHistory:
    """Maintain a rolling window of `resource.signal` events for trend analysis.

    This component is intended to run on the coordinator/scheduler node:

    - Subscribe to `resource.signal` (published by workers via `report_resource_signal`)
    - Update the local `HardwareEnvironmentRegistry` with the latest metrics
    - Keep a short sliding window (e.g., 60s/300s) for sustained-load detection
    """

    def __init__(
        self,
        *,
        event_bus: EventBus,
        window_seconds: float = 300.0,
        max_samples_per_worker: int = 600,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus is required")
        self._bus = event_bus
        self._window = max(0.0, float(window_seconds))
        self._max_samples = max(1, int(max_samples_per_worker))
        self._now = time_source or time.time

        self._lock = threading.RLock()
        self._history: Dict[str, Deque[Sample]] = {}
        self._cancel = subscribe_resource_signals(self._bus, self._on_resource_signal)

    def close(self) -> None:
        cancel = getattr(self, "_cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass
        with self._lock:
            self._history.clear()

    # ------------------------------------------------------------------ aggregation
    def worker_samples(self, worker_id: str, *, window_seconds: float | None = None) -> list[Sample]:
        """Return samples for *worker_id* within the configured (or provided) window."""

        wid = str(worker_id or "")
        if not wid:
            return []
        window = self._window if window_seconds is None else max(0.0, float(window_seconds))
        cutoff = self._now() - window
        with self._lock:
            items = list(self._history.get(wid, ()))
        return [(ts, metrics) for ts, metrics in items if ts >= cutoff]

    def rolling_mean(
        self,
        worker_id: str,
        *,
        keys: Iterable[str] | None = None,
        window_seconds: float | None = None,
    ) -> Dict[str, float]:
        """Return the rolling mean for the selected metric keys."""

        samples = self.worker_samples(worker_id, window_seconds=window_seconds)
        if not samples:
            return {}
        if keys is None:
            key_set = set()
            for _, metrics in samples:
                key_set.update(metrics.keys())
            keys = sorted(key_set)

        totals: Dict[str, float] = {str(k): 0.0 for k in keys}
        counts: Dict[str, int] = {str(k): 0 for k in keys}
        for _, metrics in samples:
            for key in totals:
                if key in metrics:
                    try:
                        totals[key] += float(metrics[key])
                        counts[key] += 1
                    except Exception:
                        continue

        return {k: (totals[k] / counts[k]) for k in totals if counts[k] > 0}

    def sustained_high(
        self,
        worker_id: str,
        *,
        cpu_threshold: float = 90.0,
        memory_threshold: float = 85.0,
        sustain_seconds: float = 30.0,
        min_samples: int = 3,
    ) -> bool:
        """Return True if load is sustained above thresholds within *sustain_seconds*."""

        window = max(0.0, float(sustain_seconds))
        samples = self.worker_samples(worker_id, window_seconds=window)
        if len(samples) < max(1, int(min_samples)):
            return False
        oldest = samples[0][0]
        newest = samples[-1][0]
        if newest - oldest < window:
            return False

        means = self.rolling_mean(
            worker_id,
            keys=("cpu_percent", "memory_percent"),
            window_seconds=window,
        )
        cpu = float(means.get("cpu_percent", 0.0))
        mem = float(means.get("memory_percent", 0.0))
        return cpu >= float(cpu_threshold) or mem >= float(memory_threshold)

    # ------------------------------------------------------------------ event handler
    async def _on_resource_signal(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        worker_id = str(event.get("worker_id") or "")
        if not worker_id:
            return
        signal = event.get("resource_signal")
        if not isinstance(signal, dict):
            return

        metrics: Dict[str, float] = {}
        for key, value in signal.items():
            try:
                metrics[str(key)] = float(value)
            except Exception:
                continue
        if not metrics:
            return

        metadata = event.get("metadata")
        meta_dict = dict(metadata) if isinstance(metadata, dict) else None

        try:
            get_hardware_registry().update_metrics(worker_id, metrics, metadata=meta_dict)
        except Exception:
            pass

        ts = float(self._now())
        with self._lock:
            bucket = self._history.get(worker_id)
            if bucket is None:
                bucket = deque(maxlen=self._max_samples)
                self._history[worker_id] = bucket
            bucket.append((ts, metrics))
            self._prune_locked(worker_id, now=ts)

    def _prune_locked(self, worker_id: str, *, now: float) -> None:
        bucket = self._history.get(worker_id)
        if not bucket:
            return
        if self._window <= 0.0:
            return
        cutoff = now - self._window
        while bucket and bucket[0][0] < cutoff:
            bucket.popleft()


__all__ = ["ResourceSignalHistory"]

