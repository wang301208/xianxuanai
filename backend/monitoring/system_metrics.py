from __future__ import annotations

import threading
import time
from typing import Dict

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

from events import EventBus


class SystemMetricsCollector:
    """Periodically publish CPU and memory usage for registered processes."""

    def __init__(self, event_bus: EventBus, interval: float = 5.0) -> None:
        self._bus = event_bus
        self._interval = interval
        self._agents: Dict[str, int] = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def register(self, name: str, pid: int) -> None:
        """Start tracking metrics for *pid* under *name*."""
        self._agents[name] = pid

    def unregister(self, name: str) -> None:
        self._agents.pop(name, None)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            for name, pid in list(self._agents.items()):
                cpu = 0.0
                mem = 0.0
                if psutil is not None:
                    try:
                        proc = psutil.Process(pid)
                        cpu = proc.cpu_percent(interval=None)
                        mem = proc.memory_percent()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        cpu = 0.0
                        mem = 0.0
                self._bus.publish(
                    "agent.resource",
                    {"agent": name, "cpu": cpu, "memory": mem},
                )
            time.sleep(self._interval)
