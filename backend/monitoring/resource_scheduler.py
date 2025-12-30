"""Adaptive resource and attention scheduling utilities."""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from events import EventBus

from .global_workspace import GlobalWorkspace, WorkspaceMessage


@dataclass
class ModulePolicy:
    """Configuration and state for dynamically throttled modules."""

    adjust: Callable[[float], None]
    base_interval: float
    min_interval: float
    max_interval: float
    slowdown_factor: float = 1.0
    boost_factor: float = 1.0
    last_interval: float = field(init=False)

    def __post_init__(self) -> None:
        self.base_interval = max(self.min_interval, float(self.base_interval))
        self.last_interval = self.base_interval


class ResourceScheduler:
    """Coordinate compute allocation using live system feedback.

    The scheduler monitors resource usage events and dynamically adjusts the
    global workspace attention threshold, execution weights, and registered
    module polling intervals. High-load conditions trigger conservative
    scheduling while high-severity alerts temporarily boost critical sensing
    components to maximise responsiveness.
    """

    def __init__(
        self,
        global_workspace: GlobalWorkspace,
        event_bus: EventBus,
        *,
        scheduler: Optional[object] = None,
        attention_bounds: Tuple[float, float] = (0.1, 0.9),
        smoothing: float = 0.3,
        backlog_target: int = 5,
        alert_decay: float = 0.15,
    ) -> None:
        self._workspace = global_workspace
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._attention_bounds = (
            min(attention_bounds),
            max(attention_bounds),
        )
        self._smoothing = max(0.0, min(1.0, smoothing))
        self._backlog_target = max(1, backlog_target)
        self._alert_decay = max(0.01, alert_decay)

        self._agent_metrics: Dict[str, Tuple[float, float]] = {}
        self._modules: Dict[str, ModulePolicy] = {}
        self._current_load = 0.0
        self._backlog = 0
        self._alert_level = 0.0
        self._last_update = time.time()
        self._current_threshold = self._attention_bounds[0]
        self._workspace.set_attention_threshold(self._current_threshold)

        self._lock = threading.RLock()
        self._subscriptions: List[Callable[[], None]] = [
            self._event_bus.subscribe("agent.resource", self._on_resource_event),
            self._event_bus.subscribe("environment.alert", self._on_attention_signal),
            self._event_bus.subscribe("environment.update", self._on_attention_signal),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Unsubscribe from event streams and release resources."""

        with self._lock:
            for cancel in self._subscriptions:
                try:
                    cancel()
                except Exception:
                    pass
            self._subscriptions.clear()

    def register_module(
        self,
        name: str,
        adjust_callback: Callable[[float], None],
        *,
        base_interval: float,
        min_interval: float,
        max_interval: float,
        slowdown_factor: float = 1.0,
        boost_factor: float = 1.0,
    ) -> None:
        """Register a module whose cadence should adapt to system load."""

        policy = ModulePolicy(
            adjust=adjust_callback,
            base_interval=base_interval,
            min_interval=min_interval,
            max_interval=max_interval,
            slowdown_factor=max(0.0, slowdown_factor),
            boost_factor=max(0.0, boost_factor),
        )
        with self._lock:
            self._modules[name] = policy

    def unregister_module(self, name: str) -> None:
        """Remove a previously registered adaptive module."""

        with self._lock:
            self._modules.pop(name, None)

    def update_backlog(self, pending_tasks: int) -> None:
        """Inform the scheduler of the current task backlog size."""

        with self._lock:
            self._backlog = max(0, int(pending_tasks))
        self._rebalance()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    async def _on_resource_event(self, event: Dict[str, object]) -> None:
        agent = str(event.get("agent") or "anonymous")
        cpu = float(event.get("cpu", 0.0)) if "cpu" in event else None
        mem = float(event.get("memory", 0.0)) if "memory" in event else None
        action = event.get("action")
        with self._lock:
            if cpu is not None and mem is not None:
                self._agent_metrics[agent] = (max(0.0, cpu), max(0.0, mem))
                self._current_load = self._smooth_load(self._agent_metrics.values())
            if action == "throttle":
                self._alert_level = max(self._alert_level, 0.4)
        self._rebalance()

    async def _on_attention_signal(self, event: Dict[str, object]) -> None:
        severity = event.get("severity")
        if severity is None:
            return
        try:
            sev = max(0.0, min(1.0, float(severity)))
        except (TypeError, ValueError):
            return
        with self._lock:
            self._alert_level = max(self._alert_level, sev)
        self._rebalance()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _smooth_load(self, samples: Iterable[Tuple[float, float]]) -> float:
        cpu_avg = 0.0
        mem_avg = 0.0
        size = 0
        for cpu, mem in samples:
            cpu_avg += cpu
            mem_avg += mem
            size += 1
        if size == 0:
            return self._current_load
        cpu_norm = (cpu_avg / size) / 100.0
        mem_norm = (mem_avg / size) / 100.0
        raw = max(cpu_norm, mem_norm)
        return (self._smoothing * raw) + (1 - self._smoothing) * self._current_load

    def _rebalance(self) -> None:
        with self._lock:
            now = time.time()
            elapsed = max(0.0, now - self._last_update)
            self._last_update = now
            if self._alert_level > 0.0:
                decay = elapsed * self._alert_decay
                self._alert_level = max(0.0, self._alert_level - decay)

            load = max(0.0, min(1.0, self._current_load))
            backlog = max(0, self._backlog)
            alert = max(0.0, min(1.0, self._alert_level))
            module_items = list(self._modules.items())

        modules = [policy for _, policy in module_items]
        threshold = self._compute_threshold(load, backlog, alert)
        if abs(threshold - self._current_threshold) > 1e-3:
            self._workspace.set_attention_threshold(threshold)
            self._current_threshold = threshold

        self._adjust_scheduler(load, backlog)
        self._adjust_modules(modules, load, alert)

        module_snapshot = {name: policy.last_interval for name, policy in module_items}
        summary = (
            f"threshold={threshold:.2f} load={load:.2f} backlog={backlog} alert={alert:.2f}"
        )
        try:
            self._workspace.publish_message(
                WorkspaceMessage(
                    type="monitoring.scheduler",
                    source="resource_scheduler",
                    payload={
                        "threshold": threshold,
                        "load": load,
                        "backlog": backlog,
                        "alert": alert,
                        "module_intervals": module_snapshot,
                    },
                    summary=summary,
                    tags=("monitoring", "scheduler"),
                    importance=max(load, alert),
                ),
                propagate=False,
            )
        except Exception:
            pass

    def _compute_threshold(self, load: float, backlog: int, alert: float) -> float:
        base_low, base_high = self._attention_bounds
        range_width = base_high - base_low
        backlog_pressure = min(1.0, backlog / self._backlog_target)
        pressure = min(1.0, max(load, backlog_pressure))
        baseline = base_low + pressure * range_width
        return max(
            base_low,
            min(base_high, baseline * (1.0 - alert) + base_low * alert),
        )

    def _adjust_scheduler(self, load: float, backlog: int) -> None:
        if not self._scheduler or not hasattr(self._scheduler, "set_weights"):
            return
        backlog_pressure = min(1.0, backlog / self._backlog_target)
        cpu_weight = 1.0 + load * 1.5
        task_weight = 1.0 + backlog_pressure * 2.0
        try:
            self._scheduler.set_weights(cpu=cpu_weight, memory=cpu_weight, tasks=task_weight)
        except Exception:
            # Defensive: avoid propagating scheduler failures
            pass

    def _adjust_modules(
        self,
        modules: List[ModulePolicy],
        load: float,
        alert: float,
    ) -> None:
        adjustments: List[Tuple[ModulePolicy, float]] = []
        for policy in modules:
            interval = policy.base_interval * (1.0 + load * policy.slowdown_factor)
            interval /= 1.0 + alert * policy.boost_factor
            interval = max(policy.min_interval, min(policy.max_interval, interval))
            adjustments.append((policy, interval))

        for policy, interval in adjustments:
            if abs(interval - policy.last_interval) / max(policy.last_interval, 1e-6) < 0.05:
                continue
            maybe_coro = None
            try:
                maybe_coro = policy.adjust(interval)
                if asyncio.iscoroutine(maybe_coro):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop and loop.is_running():
                        loop.create_task(maybe_coro)
                    else:
                        asyncio.run(maybe_coro)
            finally:
                policy.last_interval = interval


__all__ = ["ResourceScheduler"]
