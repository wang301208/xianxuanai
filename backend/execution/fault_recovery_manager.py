from __future__ import annotations

"""Failure-aware recovery hooks for capability modules.

This component is intentionally lightweight and conservative:
- it only reacts when failed tasks explicitly identify a module in metadata
- it applies a simple "reload module" action with a cooldown
- it publishes recovery events for observability

It complements (but does not replace) deeper self-healing components such as
`SelfDiagnoser` (semantic diagnosis) and `ModuleLifecycleManager` (idle pruning).

Inputs:
- `task_manager.task_completed` (failed tasks)

Outputs:
- `fault_recovery.module_reload_attempted`
- `fault_recovery.module_reloaded` (when reload succeeds)
"""

import logging
import os
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Mapping, Optional

try:  # optional in some deployments
    from events import EventBus  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EventBus = None  # type: ignore

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return int(default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


class FaultRecoveryManager:
    """Reload unstable capability modules when failures burst."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        module_manager: Any,
        enabled: bool | None = None,
        window_secs: float | None = None,
        max_failures: int | None = None,
        cooldown_secs: float | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus is required")
        self._bus = event_bus
        self._modules = module_manager
        self._logger = logger_ or logger

        self.enabled = _env_bool("FAULT_RECOVERY_ENABLED", False) if enabled is None else bool(enabled)
        self._window_secs = _env_float("FAULT_RECOVERY_WINDOW_SECS", 180.0) if window_secs is None else float(window_secs)
        self._max_failures = _env_int("FAULT_RECOVERY_MAX_FAILURES", 3) if max_failures is None else int(max_failures)
        self._cooldown_secs = _env_float("FAULT_RECOVERY_COOLDOWN_SECS", 600.0) if cooldown_secs is None else float(cooldown_secs)

        self._failures: Dict[str, Deque[float]] = {}
        self._last_reload_ts: Dict[str, float] = {}

        self._subscriptions: list[Callable[[], None]] = [
            self._bus.subscribe("task_manager.task_completed", self._on_task_completed),
        ]

    def close(self) -> None:
        subs = list(self._subscriptions)
        self._subscriptions.clear()
        for cancel in subs:
            try:
                cancel()
            except Exception:
                continue

    async def _on_task_completed(self, event: Dict[str, Any]) -> None:
        if not self.enabled or not isinstance(event, Mapping):
            return
        status = str(event.get("status") or "").strip().lower()
        if status in {"completed", "success"}:
            return
        module = self._extract_module(event)
        if not module:
            return
        now = float(event.get("time", time.time()) or time.time())
        window = max(0.0, float(self._window_secs))
        max_failures = max(1, int(self._max_failures))

        history = self._failures.get(module)
        if history is None:
            history = deque(maxlen=max(8, max_failures * 4))
            self._failures[module] = history
        history.append(now)
        if window > 0:
            cutoff = now - window
            while history and float(history[0]) < cutoff:
                history.popleft()

        if len(history) < max_failures:
            return

        last = float(self._last_reload_ts.get(module, 0.0) or 0.0)
        if self._cooldown_secs > 0 and (now - last) < float(self._cooldown_secs):
            return

        self._attempt_reload(module, now, trigger_event=dict(event))

    def _extract_module(self, event: Mapping[str, Any]) -> str | None:
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        for key in ("module", "capability", "capability_module"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _attempt_reload(self, module: str, now: float, *, trigger_event: Dict[str, Any]) -> None:
        module = str(module or "").strip()
        if not module:
            return

        payload = {
            "time": float(now),
            "module": module,
            "action": "reload",
            "trigger": {
                "task_id": trigger_event.get("task_id"),
                "name": trigger_event.get("name"),
                "category": trigger_event.get("category"),
                "error": trigger_event.get("error"),
                "status": trigger_event.get("status"),
            },
        }
        try:
            self._bus.publish("fault_recovery.module_reload_attempted", dict(payload))
        except Exception:
            pass

        ok = False
        error = None
        try:
            unload = getattr(self._modules, "unload", None)
            load = getattr(self._modules, "load", None)
            if callable(unload):
                try:
                    unload(module)
                except KeyError:
                    pass
            if callable(load):
                load(module)
            ok = True
        except Exception as exc:  # pragma: no cover - best effort
            error = repr(exc)
            ok = False

        self._last_reload_ts[module] = float(now)
        if ok:
            self._failures.pop(module, None)
        payload["status"] = "ok" if ok else "failed"
        if error:
            payload["error"] = error
        try:
            self._bus.publish("fault_recovery.module_reloaded", dict(payload))
        except Exception:
            pass


__all__ = ["FaultRecoveryManager"]

