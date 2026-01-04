from __future__ import annotations

"""Event-driven scheduler control plane.

This module keeps "policy decisions" and "actuation" decoupled:

- Producers publish `scheduler.control` commands (e.g. from EnvironmentAdapter,
  external orchestration, or a higher-level scheduler).
- This manager applies commands by calling local actuators (TaskManager, agent
  Scheduler, etc.) and publishes `scheduler.status` for observability.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, Mapping, Optional

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


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except Exception:
        return None


class SchedulerControlManager:
    """Consume `scheduler.control` events and apply local scheduler actions."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        task_manager: Any | None = None,
        scheduler: Any | None = None,
        module_manager: Any | None = None,
        enabled: bool | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus is required")
        self._bus = event_bus
        self._task_manager = task_manager
        self._scheduler = scheduler
        self._module_manager = module_manager
        self._logger = logger_ or logger

        self.enabled = _env_bool("SCHEDULER_CONTROL_ENABLED", True) if enabled is None else bool(enabled)

        self._subscriptions: list[Callable[[], None]] = [
            self._bus.subscribe("scheduler.control", self._on_control),
        ]

    def attach_module_manager(self, module_manager: Any | None) -> None:
        self._module_manager = module_manager

    def close(self) -> None:
        subs = list(self._subscriptions)
        self._subscriptions.clear()
        for cancel in subs:
            try:
                cancel()
            except Exception:
                continue

    async def _on_control(self, event: Dict[str, Any]) -> None:
        if not self.enabled or not isinstance(event, Mapping):
            return
        action = str(event.get("action") or "").strip().lower()
        if not action:
            return

        if action in {"throttle", "set_concurrency", "set_device_concurrency"}:
            self._apply_throttle(event)
            return

        if action in {"publish_status", "status"}:
            self._emit_status(trigger=action, event=dict(event))
            return

        if action in {"set_weights", "set_scheduler_weights"}:
            self._apply_weights(event)
            return

        if action in {"module.update_config", "module.config", "module.update"}:
            self._apply_module_update_config(event)
            return

    # ------------------------------------------------------------------ actions
    def _apply_throttle(self, event: Mapping[str, Any]) -> None:
        if self._task_manager is None:
            return
        device = str(event.get("device") or "cpu").strip().lower() or "cpu"
        concurrency = _safe_int(event.get("concurrency"))
        if concurrency is None:
            concurrency = _safe_int(event.get("max_workers"))
        if concurrency is None:
            return

        reason = event.get("reason")
        source = event.get("source") or "scheduler.control"
        try:
            limiter = getattr(self._task_manager, "set_device_concurrency_limit", None)
            if callable(limiter):
                limiter(
                    device,
                    int(concurrency),
                    reason=str(reason) if reason else None,
                    source=str(source) if source else None,
                )
        except Exception:
            self._logger.debug("Failed to apply scheduler throttle", exc_info=True)
        self._emit_status(trigger="throttle", event=dict(event))

    def _apply_weights(self, event: Mapping[str, Any]) -> None:
        if self._scheduler is None:
            return
        weights = event.get("weights")
        if not isinstance(weights, Mapping):
            return
        update = getattr(self._scheduler, "set_weights", None)
        if not callable(update):
            return
        clean: Dict[str, float] = {}
        for key, value in weights.items():
            try:
                clean[str(key)] = float(value)
            except Exception:
                continue
        if not clean:
            return
        try:
            update(**clean)
        except Exception:
            self._logger.debug("Failed to apply scheduler weights", exc_info=True)
        self._emit_status(trigger="set_weights", event=dict(event))

    def _apply_module_update_config(self, event: Mapping[str, Any]) -> None:
        manager = self._module_manager
        if manager is None:
            return
        module_name = event.get("module")
        if module_name is None:
            module_name = event.get("name")
        if not isinstance(module_name, str) or not module_name.strip():
            return

        load_if_missing = bool(event.get("load") or event.get("ensure") or event.get("load_if_missing"))
        runtime_config = event.get("runtime_config")
        if runtime_config is None:
            runtime_config = event.get("config")
        overrides = event.get("overrides")
        if overrides is not None and not isinstance(overrides, Mapping):
            overrides = None

        module = None
        get_loaded = getattr(manager, "get_loaded", None)
        if callable(get_loaded):
            try:
                module = get_loaded(module_name)
            except Exception:
                module = None
        if module is None and load_if_missing:
            load = getattr(manager, "load", None)
            if callable(load):
                try:
                    module = load(module_name)
                except Exception:
                    module = None
        if module is None:
            return

        update = getattr(module, "update_config", None)
        if not callable(update):
            return

        try:
            if overrides is not None:
                update(runtime_config, overrides=dict(overrides))
            elif runtime_config is not None:
                update(runtime_config)
            else:
                update()
        except TypeError:
            try:
                if overrides is not None:
                    update(dict(overrides))
                elif runtime_config is not None:
                    update(runtime_config)
                else:
                    update()
            except Exception:
                self._logger.debug("Failed to apply module update_config", exc_info=True)
                return
        except Exception:
            self._logger.debug("Failed to apply module update_config", exc_info=True)
            return

        self._emit_status(trigger="module.update_config", event=dict(event))

    # ------------------------------------------------------------------ status
    def _emit_status(self, *, trigger: str, event: Dict[str, Any]) -> None:
        now = time.time()
        payload: Dict[str, Any] = {
            "time": float(now),
            "trigger": str(trigger),
        }
        action = event.get("action")
        if action:
            payload["action"] = str(action)
        source = event.get("source")
        if source:
            payload["source"] = str(source)
        module_name = event.get("module")
        if module_name:
            payload["module"] = str(module_name)

        if self._task_manager is not None:
            try:
                depth = getattr(self._task_manager, "queue_depth", None)
                payload["queue_depth"] = int(depth()) if callable(depth) else None
            except Exception:
                payload["queue_depth"] = None
            limits: Dict[str, int] = {}
            try:
                limit_fn = getattr(self._task_manager, "device_concurrency_limit", None)
                if callable(limit_fn):
                    for name in ("cpu", "gpu"):
                        value = int(limit_fn(name) or 0)
                        if value > 0:
                            limits[name] = value
            except Exception:
                limits = {}
            if limits:
                payload["device_concurrency_limits"] = limits

        try:
            self._bus.publish("scheduler.status", payload)
        except Exception:
            pass


__all__ = ["SchedulerControlManager"]
