"""Utilities for streaming data ingestion using an event queue.

The :class:`StreamingDataIngestor` subscribes to an event bus topic and
buffers incoming events in a local queue. A consumer can periodically drain
this queue and feed the resulting samples into an online learning routine.

The ingestor optionally accepts an active learning sampler which can be used
to prioritise which samples from the queue are returned.
"""

from __future__ import annotations

from queue import Empty, Queue
from typing import Any, Callable, Dict, List

import psutil

from events import EventBus, create_event_bus
from backend.ml.active_sampler import ActiveLearningSampler
from backend.monitoring import PerformanceMonitor


class StreamingDataIngestor:
    """Subscribe to events and expose them as a stream of training samples."""

    def __init__(
        self,
        bus: EventBus | None = None,
        *,
        topic: str = "training.sample",
        sampler: ActiveLearningSampler | None = None,
        monitor: PerformanceMonitor | None = None,
    ) -> None:
        self.bus = bus or create_event_bus()
        self.topic = topic
        self.queue: Queue[Dict[str, Any]] = Queue()

        async def _enqueue(event: Dict[str, Any]) -> None:
            self.queue.put(event)

        self.bus.subscribe(topic, _enqueue)
        self.sampler = sampler
        self._monitor = monitor
        self._process = psutil.Process()

    def drain(self) -> List[Dict[str, Any]]:
        """Return all queued events, optionally prioritised by the sampler."""
        batch: List[Dict[str, Any]] = []
        while True:
            try:
                batch.append(self.queue.get_nowait())
            except Empty:
                break
        if self.sampler and batch:
            features = [e.get("features") for e in batch]
            k = len(batch)
            idx = self.sampler.select(features=features, k=k)
            batch = [batch[i] for i in idx]
        return batch

    def stream(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Process all queued events using ``handler`` and log metrics."""
        for event in self.drain():
            handler(event)
            if self._monitor is not None:
                cpu = self._process.cpu_percent(interval=None)
                mem = self._process.memory_percent()
                self._monitor.log_resource_usage("runner.streaming", cpu, mem)
                self._monitor.log_task_completion("runner.streaming")
