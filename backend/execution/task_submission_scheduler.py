from __future__ import annotations

"""Task submission wrapper (admission control) for TaskManager.

The goal is to keep TaskManager generic while allowing a separate "scheduler"
layer to inject decisions (priority/device/tags) at submission time.

This wrapper is intentionally conservative: it only applies heuristics when
explicitly requested (e.g. `priority=None` / `priority="auto"`).
"""

import logging
import os
import time
from typing import Any, Callable, Dict, Mapping, Optional

from .task_manager import TaskHandle, TaskManager, TaskPriority

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalise_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class TaskSubmissionScheduler:
    """Inject scheduling decisions before tasks enter TaskManager's queue."""

    def __init__(
        self,
        *,
        task_manager: TaskManager,
        event_bus: Any | None = None,
        enabled: bool | None = None,
        emit_status_on_submit: bool | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        if task_manager is None:
            raise ValueError("task_manager is required")
        self._task_manager = task_manager
        self._bus = event_bus
        self._logger = logger_ or logger

        self.enabled = _env_bool("TASK_SUBMISSION_SCHEDULER_ENABLED", True) if enabled is None else bool(enabled)
        self._emit_status = (
            _env_bool("SCHEDULER_STATUS_ON_SUBMIT", False)
            if emit_status_on_submit is None
            else bool(emit_status_on_submit)
        )

    def submit_task(
        self,
        func: Callable[..., Any],
        *args: Any,
        priority: int | TaskPriority | str | None = TaskPriority.NORMAL,
        deadline: float | None = None,
        category: str = "general",
        name: str | None = None,
        device: str | None = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TaskHandle:
        meta = dict(metadata or {})
        resolved_priority = self._resolve_priority(priority, category=category, metadata=meta)
        resolved_device = self._resolve_device(device, metadata=meta)

        if self.enabled:
            self._attach_decision(meta, resolved_priority, resolved_device)

        handle = self._task_manager.submit(
            func,
            *args,
            priority=resolved_priority,
            deadline=deadline,
            category=category,
            name=name,
            device=resolved_device,
            metadata=meta,
            **kwargs,
        )

        if self._emit_status and self._bus is not None and hasattr(self._bus, "publish"):
            self._emit_status_event(handle, resolved_priority, resolved_device)
        return handle

    # Drop-in alias for callers that expect `.submit(...)`.
    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> TaskHandle:
        return self.submit_task(func, *args, **kwargs)

    # ------------------------------------------------------------------ heuristics
    def _resolve_priority(
        self,
        priority: int | TaskPriority | str | None,
        *,
        category: str,
        metadata: Mapping[str, Any],
    ) -> int:
        if priority is None or (isinstance(priority, str) and priority.strip().lower() in {"auto", "dynamic"}):
            return self._infer_priority(category=category, metadata=metadata)
        if isinstance(priority, TaskPriority):
            return int(priority)
        if isinstance(priority, str):
            value = priority.strip().lower()
            mapping = {
                "low": TaskPriority.LOW,
                "normal": TaskPriority.NORMAL,
                "high": TaskPriority.HIGH,
                "critical": TaskPriority.CRITICAL,
            }
            if value in mapping:
                return int(mapping[value])
            try:
                return int(float(value))
            except Exception:
                return int(TaskPriority.NORMAL)
        try:
            return int(priority)
        except Exception:
            return int(TaskPriority.NORMAL)

    def _infer_priority(self, *, category: str, metadata: Mapping[str, Any]) -> int:
        # Explicit flags win.
        if _normalise_bool(metadata.get("critical")):
            return int(TaskPriority.CRITICAL)
        if _normalise_bool(metadata.get("interactive")) or _normalise_bool(metadata.get("user_request")):
            return int(TaskPriority.HIGH)
        if _normalise_bool(metadata.get("background")):
            return int(TaskPriority.LOW)

        cat = (category or "").strip().lower()
        if cat in {"learning", "automl", "maintenance", "background"}:
            return int(TaskPriority.LOW)
        if cat in {"planning", "control", "orchestration"}:
            return int(TaskPriority.HIGH)
        return int(TaskPriority.NORMAL)

    def _resolve_device(self, device: str | None, *, metadata: Mapping[str, Any]) -> str:
        requested = (device or "").strip()
        if requested:
            return requested
        # Default to "auto" so TaskManager can route GPU-capable tasks when tagged.
        return "auto" if self.enabled else "cpu"

    def _attach_decision(self, metadata: Dict[str, Any], priority: int, device: str) -> None:
        now = time.time()
        decision = {
            "time": float(now),
            "priority": int(priority),
            "device": str(device),
            "policy": "heuristic_v1",
        }
        existing = metadata.get("scheduler")
        if isinstance(existing, dict):
            existing.update(decision)
            return
        metadata["scheduler"] = decision

    # ------------------------------------------------------------------ status
    def _emit_status_event(self, handle: TaskHandle, priority: int, device: str) -> None:
        if self._bus is None or not hasattr(self._bus, "publish"):
            return
        try:
            payload = {
                "time": float(time.time()),
                "trigger": "task_scheduled",
                "task_id": handle.task_id,
                "name": handle.name,
                "category": handle.category,
                "priority": int(priority),
                "device": str(device),
            }
            self._bus.publish("scheduler.status", payload)
        except Exception:
            self._logger.debug("Failed to publish scheduler.status on submit", exc_info=True)


__all__ = ["TaskSubmissionScheduler"]

