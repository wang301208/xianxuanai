from __future__ import annotations

"""Proactive self-debugging hooks for the agent runtime.

This module addresses a common gap in purely threshold-based self-optimisation:
many failures are *logical* (bad plans, invalid assumptions, repeated action
loops) or *functional* (exceptions) and benefit from explicit diagnosis and
repair actions.

The :class:`SelfDebugManager` is event-driven and intentionally lightweight. It:
- listens to decision/action/task lifecycle events on the event bus
- detects failure bursts / replan storms / repeated-action loops
- publishes structured diagnostic cases and a self-debug plan request

It does not apply code patches by default; instead it publishes evidence so a
dedicated agent/human loop (or an opt-in autopatcher) can act on it.
"""

import logging
import os
import time
from collections import Counter, deque
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


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


class SelfDebugManager:
    """Detect complex failure patterns and request explicit self-debug actions."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        memory_router: Any | None = None,
        performance_monitor: Any | None = None,
        enabled: bool | None = None,
        window_secs: float | None = None,
        cooldown_secs: float | None = None,
        max_failures_window: int | None = None,
        max_replans_window: int | None = None,
        max_same_action_failures: int | None = None,
        max_bad_plans_window: int | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus is required")
        self._bus = event_bus
        self._memory_router = memory_router
        self._performance_monitor = performance_monitor
        self._logger = logger_ or logger

        self._enabled = _env_bool("SELF_DEBUG_ENABLED", True) if enabled is None else bool(enabled)
        self._window_secs = _env_float("SELF_DEBUG_WINDOW_SECS", 180.0) if window_secs is None else float(window_secs)
        self._cooldown_secs = (
            _env_float("SELF_DEBUG_COOLDOWN_SECS", 300.0)
            if cooldown_secs is None
            else float(cooldown_secs)
        )
        self._max_failures_window = (
            _env_int("SELF_DEBUG_MAX_FAILURES", 4)
            if max_failures_window is None
            else int(max_failures_window)
        )
        self._max_replans_window = (
            _env_int("SELF_DEBUG_MAX_REPLANS", 3)
            if max_replans_window is None
            else int(max_replans_window)
        )
        self._max_same_action_failures = (
            _env_int("SELF_DEBUG_MAX_SAME_ACTION_FAILURES", 3)
            if max_same_action_failures is None
            else int(max_same_action_failures)
        )
        self._max_bad_plans_window = (
            _env_int("SELF_DEBUG_MAX_BAD_PLANS", 2)
            if max_bad_plans_window is None
            else int(max_bad_plans_window)
        )

        self._failures: Deque[Dict[str, Any]] = deque(maxlen=128)
        self._replans: Deque[float] = deque(maxlen=128)
        self._bad_plans: Deque[float] = deque(maxlen=64)
        self._last_plan: Dict[str, Any] = {}
        self._last_trigger_ts: float | None = None

        self._subscriptions: list[Callable[[], None]] = [
            self._bus.subscribe("agent.action.outcome", self._on_action_outcome),
            self._bus.subscribe("agent.conductor.directive", self._on_conductor_directive),
            self._bus.subscribe("task_manager.task_completed", self._on_task_manager_completed),
            self._bus.subscribe("planner.plan_ready", self._on_plan_ready),
        ]

    def close(self) -> None:
        subs = list(self._subscriptions)
        self._subscriptions.clear()
        for cancel in subs:
            try:
                cancel()
            except Exception:
                continue

    # ------------------------------------------------------------------
    async def _on_action_outcome(self, event: Dict[str, Any]) -> None:
        if not self._enabled or not isinstance(event, Mapping):
            return
        now = float(event.get("time", time.time()) or time.time())
        status = str(event.get("status") or "").lower()
        reward = _safe_float(event.get("reward"))
        is_failure = status == "error" or (reward is not None and reward < 0)
        if not is_failure:
            return

        action = event.get("command") or event.get("action")
        reason = event.get("error_reason") or event.get("error") or status
        self._failures.append(
            {
                "time": now,
                "source": "action_outcome",
                "agent": event.get("agent"),
                "action": str(action) if action is not None else None,
                "reason": str(reason) if reason is not None else None,
                "reward": reward,
            }
        )
        self._maybe_trigger(now, hint="action_outcome")

    async def _on_conductor_directive(self, event: Dict[str, Any]) -> None:
        if not self._enabled or not isinstance(event, Mapping):
            return
        directive = event.get("directive")
        if not isinstance(directive, Mapping):
            return
        if not bool(directive.get("requires_replan")):
            # Only count replans as a signal of decision-chain problems.
            return
        now = float(event.get("time", time.time()) or time.time())
        self._replans.append(now)
        self._maybe_trigger(now, hint="replan")

    async def _on_task_manager_completed(self, event: Dict[str, Any]) -> None:
        if not self._enabled or not isinstance(event, Mapping):
            return
        status = str(event.get("status") or "").lower()
        if status == "completed":
            return
        now = float(event.get("time", time.time()) or time.time())
        self._failures.append(
            {
                "time": now,
                "source": "task_manager",
                "task_id": event.get("task_id"),
                "name": event.get("name"),
                "category": event.get("category"),
                "reason": str(event.get("error") or status),
                "autofix": event.get("autofix"),
            }
        )
        self._maybe_trigger(now, hint="task_failure")

    async def _on_plan_ready(self, event: Dict[str, Any]) -> None:
        if not self._enabled or not isinstance(event, Mapping):
            return
        self._last_plan = dict(event)
        issues = self._validate_plan(event)
        if issues:
            self._bad_plans.append(time.time())
            try:
                self._bus.publish(
                    "diagnostics.plan_issue",
                    {"time": time.time(), "issues": issues, "goal": event.get("goal"), "source": "self_debug"},
                )
            except Exception:
                pass
            self._maybe_trigger(time.time(), hint="bad_plan")

    # ------------------------------------------------------------------
    def _validate_plan(self, plan: Mapping[str, Any]) -> list[str]:
        tasks = plan.get("tasks")
        if tasks is None:
            return []
        if not isinstance(tasks, list):
            return ["tasks_not_list"]
        cleaned: list[str] = [str(t or "").strip() for t in tasks if str(t or "").strip()]
        if not cleaned and tasks:
            return ["tasks_empty"]
        issues: list[str] = []
        if len(cleaned) > 25:
            issues.append("tasks_too_many")
        lower = [t.lower() for t in cleaned]
        dup = len(lower) - len(set(lower))
        if dup >= max(1, len(lower) // 3):
            issues.append("tasks_many_duplicates")
        return issues

    def _prune(self, now: float) -> None:
        window = max(1e-3, float(self._window_secs))
        cutoff = float(now) - window
        while self._failures and float(self._failures[0].get("time", 0.0) or 0.0) < cutoff:
            self._failures.popleft()
        while self._replans and float(self._replans[0]) < cutoff:
            self._replans.popleft()
        while self._bad_plans and float(self._bad_plans[0]) < cutoff:
            self._bad_plans.popleft()

    def _maybe_trigger(self, now: float, *, hint: str) -> None:
        if not self._enabled:
            return
        now = float(now)
        self._prune(now)

        if (
            self._cooldown_secs > 0
            and self._last_trigger_ts is not None
            and (now - float(self._last_trigger_ts)) < float(self._cooldown_secs)
        ):
            return

        failures = list(self._failures)
        replans = list(self._replans)
        bad_plans = list(self._bad_plans)

        trigger_reason = None
        if self._max_failures_window > 0 and len(failures) >= int(self._max_failures_window):
            trigger_reason = "failure_burst"
        elif self._max_replans_window > 0 and len(replans) >= int(self._max_replans_window):
            trigger_reason = "replan_storm"
        elif self._max_bad_plans_window > 0 and len(bad_plans) >= int(self._max_bad_plans_window):
            trigger_reason = "bad_plan_burst"

        if trigger_reason is None and self._max_same_action_failures > 0 and failures:
            counts = Counter(
                f.get("action")
                for f in failures
                if f.get("source") == "action_outcome" and f.get("action")
            )
            action, count = counts.most_common(1)[0] if counts else (None, 0)
            if action and count >= int(self._max_same_action_failures):
                trigger_reason = "action_loop"

        if trigger_reason is None:
            return

        self._last_trigger_ts = now
        case = self._build_case(now, trigger_reason=trigger_reason, hint=hint)
        self._persist_case(case)
        self._publish_case(case)
        self._publish_plan(case)
        self._emit_metrics()

    def _build_case(self, now: float, *, trigger_reason: str, hint: str) -> Dict[str, Any]:
        failures = list(self._failures)
        last_failure = failures[-1] if failures else {}
        plan = dict(self._last_plan or {})
        return {
            "time": float(now),
            "trigger": str(trigger_reason),
            "hint": str(hint),
            "window_secs": float(self._window_secs),
            "counts": {
                "failures": len(failures),
                "replans": len(self._replans),
                "bad_plans": len(self._bad_plans),
            },
            "last_failure": dict(last_failure) if isinstance(last_failure, Mapping) else {},
            "last_plan": {
                "goal": plan.get("goal"),
                "tasks": plan.get("tasks"),
                "source": plan.get("source"),
            },
            "evidence": failures[-10:],
        }

    def _persist_case(self, case: Mapping[str, Any]) -> None:
        router = self._memory_router
        if router is None or not hasattr(router, "add_observation"):
            return
        try:
            summary = f"self_debug trigger={case.get('trigger')} failures={case.get('counts', {}).get('failures')}"
            router.add_observation(summary, source="self_debug", metadata=dict(case))
        except Exception:
            self._logger.debug("Failed to persist self-debug case", exc_info=True)

    def _publish_case(self, case: Mapping[str, Any]) -> None:
        try:
            self._bus.publish("diagnostics.self_debug", dict(case))
        except Exception:
            pass

    def _publish_plan(self, case: Mapping[str, Any]) -> None:
        goal = (
            f"Self-debug ({case.get('trigger')}): analyze recent failures, validate assumptions, "
            "and propose a concrete remediation (plan fix / constraint / code patch + test)."
        )
        tasks = [
            "Summarize last failures and cluster by cause.",
            "Check plan->action alignment and identify invalid assumptions.",
            "If a code bug is suspected, create a minimal repro + patch + tests, then rerun.",
            "If a planning/logic error is suspected, add constraints/guardrails and replan.",
        ]
        try:
            self._bus.publish(
                "planner.plan_ready",
                {
                    "goal": goal,
                    "tasks": tasks,
                    "source": "self_debug",
                    "metadata": {"case": dict(case)},
                },
            )
        except Exception:
            pass

    def _emit_metrics(self) -> None:
        monitor = self._performance_monitor
        if monitor is None or not hasattr(monitor, "log_snapshot"):
            return
        try:
            monitor.log_snapshot({"self_debug_trigger": 1.0})
        except Exception:
            pass


__all__ = ["SelfDebugManager"]
