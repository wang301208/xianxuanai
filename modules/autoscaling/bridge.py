from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

from modules.environment import (
    subscribe_resource_signals,
    subscribe_service_signals,
)


class AutoScalerBridge:
    """Translate resource/service signals into autoscaling callbacks."""

    def __init__(
        self,
        event_bus,
        *,
        scale_up: Callable[[str, Dict[str, float]], None],
        scale_down: Optional[Callable[[str, Dict[str, float]], None]] = None,
        high_threshold: float = 0.85,
        low_threshold: float = 0.25,
        evaluation_window: int = 5,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._bus = event_bus
        self._scale_up = scale_up
        self._scale_down = scale_down
        self._high = high_threshold
        self._low = low_threshold
        self._window = max(1, evaluation_window)
        self._logger = logger or logging.getLogger(__name__)
        self._history: Dict[str, Deque[Dict[str, float]]] = {}
        self._subscriptions = [
            subscribe_resource_signals(self._bus, self._on_resource_signal),
            subscribe_service_signals(self._bus, self._on_service_signal),
        ]

    def stop(self) -> None:
        for cancel in self._subscriptions:
            try:
                cancel()
            except Exception:  # pragma: no cover - defensive cleanup
                self._logger.debug("Autoscaler bridge cancellation failed", exc_info=True)
        self._subscriptions.clear()

    async def _on_resource_signal(self, event: Dict[str, Any]) -> None:
        worker_id = event.get("worker_id")
        metrics = event.get("resource_signal") or {}
        if worker_id and metrics:
            self._record_metrics(worker_id, metrics)

    async def _on_service_signal(self, event: Dict[str, Any]) -> None:
        service_id = event.get("service_id")
        metrics = event.get("metrics") or {}
        if service_id and metrics:
            self._record_metrics(service_id, metrics)

    def _record_metrics(self, identifier: str, metrics: Dict[str, Any]) -> None:
        numeric = {
            str(k): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        if not numeric:
            return
        history = self._history.setdefault(identifier, deque(maxlen=self._window))
        history.appendleft(numeric)
        self._evaluate(identifier, list(history))

    def _evaluate(self, identifier: str, samples: list[Dict[str, float]]) -> None:
        if not samples:
            return
        aggregate: Dict[str, float] = {}
        for sample in samples:
            for key, value in sample.items():
                aggregate[key] = aggregate.get(key, 0.0) + value
        for key in aggregate:
            aggregate[key] /= len(samples)
        load = max(
            aggregate.get("gpu_utilization", 0.0),
            aggregate.get("cpu_utilization", 0.0),
            aggregate.get("queue_depth", 0.0),
        )
        if load >= self._high:
            self._logger.debug("Triggering scale up for %s (avg load %.2f)", identifier, load)
            self._scale_up(identifier, aggregate)
        elif load <= self._low and self._scale_down is not None:
            self._logger.debug("Triggering scale down for %s (avg load %.2f)", identifier, load)
            self._scale_down(identifier, aggregate)
