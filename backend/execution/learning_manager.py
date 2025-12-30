from __future__ import annotations

"""Event-driven scheduling for background learning cycles.

This module coordinates *when* to run low-priority learning work (e.g. online
self-supervised updates, imitation updates) without interfering with foreground
task execution.
"""

import logging
import os
import threading
import time
from collections import Counter
from typing import Any, Callable, Dict, Mapping, Optional

from events import EventBus

from .task_manager import TaskHandle, TaskManager, TaskPriority

logger = logging.getLogger(__name__)


class LearningManager:
    """Schedule background learning via the event bus and TaskManager."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        task_manager: TaskManager,
        run_learning_cycle: Callable[[], Mapping[str, Any] | Dict[str, Any]],
        tick_topic: str = "learning.tick",
        request_topic: str = "learning.request",
        completed_topic: str = "learning.cycle_completed",
        state_topic: str = "agent.state",
        task_completed_topic: str = "coordinator.task_completed",
        min_interval: float | None = None,
        event_min_interval: float | None = None,
        cpu_ceiling: float | None = None,
        mem_ceiling: float | None = None,
        token_capacity: float | None = None,
        token_refill_per_sec: float | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self._bus = event_bus
        self._task_manager = task_manager
        self._run_learning_cycle = run_learning_cycle
        self._logger = logger_ or logger

        self._tick_topic = tick_topic
        self._request_topic = request_topic
        self._completed_topic = completed_topic

        self._min_interval = float(
            os.getenv("LEARNING_MIN_INTERVAL", "120") if min_interval is None else min_interval
        )
        self._event_min_interval = float(
            os.getenv("LEARNING_EVENT_MIN_INTERVAL", "30")
            if event_min_interval is None
            else event_min_interval
        )
        self._cpu_ceiling = float(
            os.getenv("LEARNING_CPU_MAX", "60") if cpu_ceiling is None else cpu_ceiling
        )
        self._mem_ceiling = float(
            os.getenv("LEARNING_MEM_MAX", "70") if mem_ceiling is None else mem_ceiling
        )

        cap = float(os.getenv("LEARNING_TOKENS_CAPACITY", "30") if token_capacity is None else token_capacity)
        self._token_capacity = max(0.0, cap)
        self._tokens = float(self._token_capacity)
        self._token_refill_per_sec = float(
            os.getenv("LEARNING_TOKENS_REFILL_PER_SEC", "0.5")
            if token_refill_per_sec is None
            else token_refill_per_sec
        )

        self._lock = threading.RLock()
        self._inflight = threading.Event()
        self._handle: TaskHandle | None = None
        self._pending: Counter[str] = Counter()
        self._states: Dict[str, str] = {}
        self._last_run_ts = 0.0
        self._last_refill_ts = time.time()
        self._pause_until_ts = 0.0

        self._subscriptions = [
            self._bus.subscribe(self._tick_topic, self._on_tick),
            self._bus.subscribe(self._request_topic, self._on_request),
            self._bus.subscribe(state_topic, self._on_agent_state),
            self._bus.subscribe(task_completed_topic, self._on_task_completed),
        ]

    def close(self) -> None:
        with self._lock:
            subs = list(self._subscriptions)
            self._subscriptions.clear()
        for cancel in subs:
            try:
                cancel()
            except Exception:
                continue

    def request_learning(self, reason: str) -> None:
        token = str(reason or "manual").strip() or "manual"
        with self._lock:
            self._pending[token] += 1

    def pause(self, seconds: float, *, reason: str = "paused") -> None:
        """Pause scheduling of learning cycles for *seconds*."""

        try:
            duration = max(0.0, float(seconds))
        except Exception:
            duration = 0.0
        if duration <= 0.0:
            return
        until = time.time() + duration
        with self._lock:
            self._pause_until_ts = max(float(self._pause_until_ts), float(until))
        try:
            self._bus.publish(
                "learning.paused",
                {"time": time.time(), "until": float(self._pause_until_ts), "reason": str(reason)},
            )
        except Exception:
            pass

    def resume(self) -> None:
        """Resume scheduling immediately."""

        with self._lock:
            self._pause_until_ts = 0.0

    def throttle(self, *, max_tokens: float = 0.5) -> None:
        """Reduce available learning tokens during high-load periods."""

        with self._lock:
            try:
                ceiling = float(max_tokens)
            except Exception:
                ceiling = 0.5
            self._tokens = min(float(self._tokens), max(0.0, ceiling))

    def status(self) -> Dict[str, Any]:
        with self._lock:
            pending = dict(self._pending)
            return {
                "inflight": bool(self._inflight.is_set()),
                "tokens": float(self._tokens),
                "token_capacity": float(self._token_capacity),
                "pending": pending,
                "last_run_ts": float(self._last_run_ts),
                "pause_until": float(self._pause_until_ts),
            }

    def tick(
        self,
        *,
        avg_cpu: float,
        avg_memory: float,
        backlog: int,
        now: float | None = None,
    ) -> bool:
        """Evaluate whether to schedule a learning cycle."""

        now_ts = time.time() if now is None else float(now)
        with self._lock:
            self._refill_tokens(now_ts)
            if now_ts < float(self._pause_until_ts):
                return False
            if self._inflight.is_set():
                return False
            if int(backlog) > 0:
                return False
            if float(avg_cpu) > self._cpu_ceiling or float(avg_memory) > self._mem_ceiling:
                return False

            requested = sum(self._pending.values()) > 0
            min_interval = self._event_min_interval if requested else self._min_interval
            if (now_ts - self._last_run_ts) < float(min_interval):
                return False
            if float(self._tokens) < 1.0:
                return False

            metadata = self._build_metadata(requested=requested)
            try:
                handle = self._task_manager.submit(
                    self._run_learning_cycle,
                    priority=TaskPriority.LOW,
                    category="learning",
                    deadline=now_ts + 300.0,
                    metadata=metadata,
                )
            except Exception:
                self._logger.debug("Failed to submit learning cycle", exc_info=True)
                return False

            self._handle = handle
            self._inflight.set()
            self._last_run_ts = now_ts
            self._tokens = max(0.0, float(self._tokens) - 1.0)
            self._pending.clear()

            try:
                handle.add_done_callback(self._on_learning_done)
            except Exception:
                pass
            return True

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    async def _on_tick(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, Mapping):
            return
        try:
            avg_cpu = float(event.get("avg_cpu", 0.0) or 0.0)
            avg_memory = float(event.get("avg_memory", 0.0) or 0.0)
            backlog = int(event.get("backlog", 0) or 0)
            now = float(event.get("time", time.time()) or time.time())
        except Exception:
            return
        self.tick(avg_cpu=avg_cpu, avg_memory=avg_memory, backlog=backlog, now=now)

    async def _on_request(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, Mapping):
            return
        reason = str(event.get("reason", "manual"))
        self.request_learning(reason)

    async def _on_task_completed(self, _: Dict[str, Any]) -> None:
        self.request_learning("task_completed")

    async def _on_agent_state(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, Mapping):
            return
        agent = event.get("agent")
        state = event.get("state")
        if not agent or not state:
            return
        agent_id = str(agent)
        state_value = str(state)
        with self._lock:
            self._states[agent_id] = state_value
        if state_value == "idle":
            self.request_learning("agent_idle")

    # ------------------------------------------------------------------
    def _refill_tokens(self, now: float) -> None:
        elapsed = max(0.0, float(now) - float(self._last_refill_ts))
        if elapsed <= 0.0:
            return
        refill_rate = max(0.0, float(self._token_refill_per_sec))
        if refill_rate > 0.0 and self._token_capacity > 0.0:
            self._tokens = min(self._token_capacity, float(self._tokens) + elapsed * refill_rate)
        self._last_refill_ts = float(now)

    def _build_metadata(self, *, requested: bool) -> Dict[str, Any]:
        reasons = dict(self._pending)
        if not reasons:
            reasons = {"maintenance": 1}
        return {
            "reason": "event" if requested else "maintenance",
            "reasons": reasons,
            "tokens": float(self._tokens),
        }

    def _on_learning_done(self, handle: TaskHandle) -> None:
        payload: Dict[str, Any] = {
            "time": time.time(),
            "ok": True,
            "task_id": getattr(handle, "task_id", None),
            "metadata": getattr(handle, "metadata", None),
        }
        try:
            result = handle.result()
            if isinstance(result, Mapping):
                payload["stats"] = dict(result)
            else:
                payload["stats"] = {"result": str(result)}
        except Exception as exc:  # pragma: no cover - defensive
            payload["ok"] = False
            payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
            payload["stats"] = {}

        with self._lock:
            self._handle = None
            self._inflight.clear()

        try:
            self._bus.publish(self._completed_topic, payload)
        except Exception:
            pass


__all__ = ["LearningManager"]
