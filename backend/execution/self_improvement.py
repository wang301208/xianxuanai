from __future__ import annotations

"""Self-improvement goal manager driven by performance metrics.

Historically this module only maintained moving targets and generated a generic
"improve X" goal. The runtime now needs a unified improvement scheduling centre
that can:

- monitor metric gaps over time
- queue concrete improvement items with priorities
- apply a change (or schedule a retrain/knowledge update)
- observe subsequent metrics and decide success vs rollback

The design is intentionally lightweight and defensive: it never assumes specific
planner/brain implementations exist. All integrations are optional and injected
via callbacks from the runtime (AgentLifecycleManager / AdaptiveResourceController).
"""

import heapq
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .improvement_experience import ImprovementExperienceStore, JsonlExperienceStore, metric_group

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return int(default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass
class ImprovementGoal:
    """Target thresholds for a given metric."""

    name: str
    target: float
    direction: str = "decrease"  # or "increase"
    history: List[float] = field(default_factory=list)
    patience: int = 4
    cooldown_secs: float = 300.0
    min_delta: float = 0.0
    failure_streak: int = 0
    last_trigger_ts: float = 0.0

    def update_target(self, value: float, *, tightening: float = 0.9) -> None:
        """Tighten target based on achieved ``value``."""

        self.history.append(value)
        if self.direction == "decrease":
            self.target = min(self.target, value * tightening)
        else:
            self.target = max(self.target, value * (2 - tightening))

    def satisfied(self, value: float) -> bool:
        if self.direction == "decrease":
            return value <= self.target
        return value >= self.target


@dataclass
class ImprovementItem:
    """A concrete improvement attempt derived from metric gaps."""

    item_id: str
    metric: str
    kind: str
    priority: int = 50
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending|active|completed|failed
    reason: str = ""

    attempts: int = 0
    max_attempts: int = 2

    baseline: float | None = None
    observations: List[float] = field(default_factory=list)
    eval_window: int = 5
    min_improvement: float = 0.02

    applied: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SelfImprovementManager:
    """Unified self-improvement scheduler.

    Public surface kept backwards-compatible with the previous implementation:
    - `ensure_goal`, `observe_metrics`, `generate_goal` remain available.

    New capabilities:
    - maintains a priority queue of `ImprovementItem`
    - executes one improvement at a time (closed-loop evaluation + rollback)
    """

    def __init__(
        self,
        *,
        initial_goals: Optional[Mapping[str, ImprovementGoal]] = None,
        tightening: float = 0.95,
        enabled: bool | None = None,
        default_patience: int | None = None,
        default_cooldown_secs: float | None = None,
        eval_window: int | None = None,
        min_improvement: float | None = None,
        max_queue: int | None = None,
        experience_store: ImprovementExperienceStore | None = None,
    ) -> None:
        self.goals: Dict[str, ImprovementGoal] = dict(initial_goals or {})
        self.tightening = tightening

        self.enabled = bool(int(os.getenv("SELF_IMPROVEMENT_ENABLED", "1"))) if enabled is None else bool(enabled)
        self._default_patience = int(os.getenv("SELF_IMPROVEMENT_PATIENCE", "4")) if default_patience is None else int(default_patience)
        self._default_cooldown_secs = float(os.getenv("SELF_IMPROVEMENT_COOLDOWN_SECS", "300")) if default_cooldown_secs is None else float(default_cooldown_secs)
        self._eval_window = int(os.getenv("SELF_IMPROVEMENT_EVAL_WINDOW", "5")) if eval_window is None else int(eval_window)
        self._min_improvement = float(os.getenv("SELF_IMPROVEMENT_MIN_IMPROVEMENT", "0.02")) if min_improvement is None else float(min_improvement)
        self._max_queue = int(os.getenv("SELF_IMPROVEMENT_MAX_QUEUE", "32")) if max_queue is None else int(max_queue)

        self._automl_request_enabled = _env_bool("SELF_IMPROVEMENT_AUTOML_REQUEST_ENABLED", True)
        self._automl_request_cooldown_secs = _env_float("SELF_IMPROVEMENT_AUTOML_COOLDOWN_SECS", 1800.0)
        self._automl_stagnation_window = max(2, _env_int("SELF_IMPROVEMENT_AUTOML_STAGNATION_WINDOW", 8))
        self._automl_stagnation_min_delta = _env_float("SELF_IMPROVEMENT_AUTOML_STAGNATION_MIN_DELTA", 0.01)
        self._automl_target_only = _env_bool("SELF_IMPROVEMENT_AUTOML_TARGET_ONLY", True)

        self._lock = threading.RLock()
        self._experience_store = experience_store if experience_store is not None else JsonlExperienceStore.from_env()
        self._experience_promote = _env_bool("SELF_IMPROVEMENT_EXPERIENCE_PROMOTE", False)
        self._seq = 0
        self._queue: list[tuple[int, float, int, ImprovementItem]] = []
        self._active: ImprovementItem | None = None
        self._latest_metrics: Dict[str, Any] = {}
        self._recently_enqueued: Dict[tuple[str, str], float] = {}
        self._automl_last_request_ts: Dict[str, float] = {}

    def observe_metrics(self, metrics: Mapping[str, float]) -> List[Dict[str, float]]:
        """Update goals based on current metrics and return achieved goals."""

        now = time.time()
        achieved: List[Dict[str, float]] = []
        with self._lock:
            if isinstance(metrics, Mapping):
                for key, value in metrics.items():
                    self._latest_metrics[str(key)] = value  # keep non-float context too

            for name, goal in list(self.goals.items()):
                raw = metrics.get(name)
                if raw is None:
                    continue
                try:
                    value = float(raw)
                except Exception:
                    continue

                goal.history.append(value)
                if goal.satisfied(value):
                    achieved.append({"goal": name, "value": value, "target": goal.target})
                    goal.failure_streak = 0
                    goal.update_target(value, tightening=self.tightening)
                else:
                    goal.failure_streak = int(goal.failure_streak) + 1
                    self._maybe_queue_improvement(goal, value, now=now)

            # Feed active improvement evaluation with the newest metric values.
            if self._active is not None:
                metric = self._active.metric
                raw = metrics.get(metric)
                if raw is not None:
                    try:
                        self._active.observations.append(float(raw))
                    except Exception:
                        pass
        return achieved

    def ensure_goal(
        self,
        name: str,
        target: float,
        direction: str = "decrease",
        *,
        patience: int | None = None,
        cooldown_secs: float | None = None,
    ) -> None:
        """Create or update a goal definition."""

        with self._lock:
            existing = self.goals.get(name)
            if existing is None:
                self.goals[name] = ImprovementGoal(
                    name=name,
                    target=float(target),
                    direction=direction,
                    patience=int(patience) if patience is not None else int(self._default_patience),
                    cooldown_secs=float(cooldown_secs) if cooldown_secs is not None else float(self._default_cooldown_secs),
                )
            else:
                existing.target = float(target)
                existing.direction = direction
                if patience is not None:
                    existing.patience = int(patience)
                if cooldown_secs is not None:
                    existing.cooldown_secs = float(cooldown_secs)

    def generate_goal(self) -> Optional[Dict[str, float]]:
        """Suggest the next internal goal to pursue based on existing targets."""

        if not self.goals:
            return None
        candidate = None
        for goal in self.goals.values():
            if not goal.history:
                candidate = goal
                break
            if goal.direction == "decrease" and goal.history[-1] > goal.target:
                candidate = goal
                break
            if goal.direction == "increase" and goal.history[-1] < goal.target:
                candidate = goal
                break
        if candidate is None:
            candidate = next(iter(self.goals.values()))
        return {"name": candidate.name, "target": candidate.target, "direction": candidate.direction}

    # ------------------------------------------------------------------
    # New scheduling API
    # ------------------------------------------------------------------
    def pending_items(self) -> List[ImprovementItem]:
        with self._lock:
            return [entry[-1] for entry in list(self._queue)]

    def active_item(self) -> ImprovementItem | None:
        with self._lock:
            return self._active

    def enqueue_automl_suggestion(self, suggestion: Mapping[str, Any]) -> bool:
        """Queue an AutoML suggestion for evaluation via the standard loop."""

        if not self.enabled or not isinstance(suggestion, Mapping):
            return False

        suggestion_id = str(suggestion.get("suggestion_id") or "").strip()
        metric = str(suggestion.get("metric") or "").strip()
        params = suggestion.get("params")
        if not suggestion_id or not metric or not isinstance(params, Mapping):
            return False

        now = time.time()
        try:
            now = float(suggestion.get("time") or now)
        except Exception:
            now = time.time()

        try:
            priority = int(suggestion.get("priority") or 80)
        except Exception:
            priority = 80

        eval_window = self._eval_window
        min_improvement = self._min_improvement
        try:
            if suggestion.get("eval_window") is not None:
                eval_window = max(1, int(float(suggestion.get("eval_window") or eval_window)))
        except Exception:
            eval_window = self._eval_window
        try:
            if suggestion.get("min_improvement") is not None:
                min_improvement = max(0.0, float(suggestion.get("min_improvement") or min_improvement))
        except Exception:
            min_improvement = self._min_improvement

        with self._lock:
            if self._active is not None and self._active.metadata.get("suggestion_id") == suggestion_id:
                return False
            for entry in self._queue:
                item = entry[-1]
                if item.metadata.get("suggestion_id") == suggestion_id:
                    return False
            if len(self._queue) >= max(1, int(self._max_queue)):
                return False

            item = ImprovementItem(
                item_id=uuid.uuid4().hex,
                metric=metric,
                kind="automl_apply",
                priority=priority,
                reason=str(suggestion.get("source") or "automl.suggestion"),
                eval_window=eval_window,
                min_improvement=min_improvement,
                metadata={
                    "suggestion_id": suggestion_id,
                    "params": dict(params),
                    "source": suggestion.get("source"),
                    "direction": suggestion.get("direction"),
                    "target": suggestion.get("target"),
                },
            )
            self._seq += 1
            heapq.heappush(self._queue, (int(priority), float(now), int(self._seq), item))
        return True

    def run_next(
        self,
        *,
        event_bus: Any | None = None,
        retrain_callback: Callable[[Mapping[str, Any]], Any] | None = None,
        knowledge_pipeline: Any | None = None,
        memory_router: Any | None = None,
        imitation_policy: Any | None = None,
        predictive_model: Any | None = None,
        runtime_config: Any | None = None,
        now: float | None = None,
    ) -> Dict[str, float]:
        """Execute at most one improvement step (apply/evaluate/rollback)."""

        if not self.enabled:
            return {}
        now_ts = time.time() if now is None else float(now)

        with self._lock:
            active = self._active

        # 1) Evaluate active experiment if enough observations arrived.
        if active is not None and active.status == "active":
            return self._maybe_finalize_active(
                active,
                event_bus=event_bus,
                imitation_policy=imitation_policy,
                predictive_model=predictive_model,
                runtime_config=runtime_config,
                memory_router=memory_router,
                now=now_ts,
            )

        # 2) No active: apply next pending item.
        item = None
        with self._lock:
            while self._queue:
                _prio, _ts, _seq, candidate = heapq.heappop(self._queue)
                if candidate.status == "pending":
                    item = candidate
                    break
            if item is None:
                return self._maybe_request_automl(event_bus=event_bus, now=now_ts)
            self._active = item if self._requires_evaluation(item) else None

        return self._apply_item(
            item,
            event_bus=event_bus,
            retrain_callback=retrain_callback,
            knowledge_pipeline=knowledge_pipeline,
            memory_router=memory_router,
            imitation_policy=imitation_policy,
            predictive_model=predictive_model,
            runtime_config=runtime_config,
            now=now_ts,
        )

    # ------------------------------------------------------------------ internals
    def _maybe_request_automl(self, *, event_bus: Any | None, now: float) -> Dict[str, float]:
        if not self._automl_request_enabled:
            return {}
        if event_bus is None or not hasattr(event_bus, "publish"):
            return {}

        candidate: tuple[str, str, float, float] | None = None
        with self._lock:
            for metric, goal in self.goals.items():
                history = goal.history
                window = int(self._automl_stagnation_window)
                if len(history) < window:
                    continue
                try:
                    last_value = float(history[-1])
                except Exception:
                    continue
                if self._automl_target_only and goal.satisfied(last_value):
                    continue
                try:
                    start_value = float(history[-window])
                except Exception:
                    start_value = float(history[0])
                delta = last_value - start_value if goal.direction == "increase" else start_value - last_value
                if float(delta) >= float(self._automl_stagnation_min_delta):
                    continue
                last_req = float(self._automl_last_request_ts.get(metric, 0.0) or 0.0)
                if self._automl_request_cooldown_secs > 0 and (now - last_req) < float(self._automl_request_cooldown_secs):
                    continue
                gap = (float(goal.target) - last_value) if goal.direction == "increase" else (last_value - float(goal.target))
                if candidate is None or float(gap) > float(candidate[2]):
                    candidate = (metric, goal.direction, float(gap), float(goal.target))

            if candidate is None:
                return {}
            metric, direction, gap, target = candidate
            self._automl_last_request_ts[metric] = float(now)

        payload = {
            "time": float(now),
            "metric": metric,
            "direction": direction,
            "target": float(target),
            "gap": float(gap),
            "eval_window": int(self._eval_window),
            "min_improvement": float(self._min_improvement),
            "source": "self_improvement",
        }
        try:
            event_bus.publish("automl.request", payload)
            return {"self_improvement_automl_requested": 1.0}
        except Exception:
            return {}

    def _maybe_queue_improvement(self, goal: ImprovementGoal, value: float, *, now: float) -> None:
        if not self.enabled:
            return
        if int(goal.failure_streak) < max(1, int(goal.patience)):
            return
        if float(goal.cooldown_secs) > 0 and (now - float(goal.last_trigger_ts)) < float(goal.cooldown_secs):
            return

        # Derive a concrete improvement item from the metric name.
        metric = goal.name
        if metric.startswith("perception_"):
            self._enqueue(
                metric=metric,
                kind="perception_retrain",
                priority=10,
                reason=f"{metric}={value:.3f} below target {goal.target:.3f}",
                metadata={"target": goal.target, "direction": goal.direction},
                now=now,
            )
        elif metric.startswith("decision_") and metric == "decision_success_rate":
            reason = f"{metric}={value:.3f} below target {goal.target:.3f}"
            candidates = [
                {"kind": "decision_exploration_boost", "priority": 20},
                {"kind": "decision_big_brain", "priority": 30},
            ]
            kinds = [c["kind"] for c in candidates]
            ranked = self._experience_store.rank_kinds(metric=metric, kinds=kinds)
            if ranked and ranked != kinds:
                base_priorities = sorted(int(c["priority"]) for c in candidates)
                by_kind = {c["kind"]: dict(c) for c in candidates}
                candidates = []
                for idx, kind in enumerate(ranked):
                    entry = by_kind.get(kind)
                    if entry is None:
                        continue
                    entry["priority"] = base_priorities[min(idx, len(base_priorities) - 1)]
                    candidates.append(entry)
            for c in candidates:
                self._enqueue(
                    metric=metric,
                    kind=str(c["kind"]),
                    priority=int(c["priority"]),
                    reason=reason,
                    metadata={"target": goal.target, "direction": goal.direction},
                    now=now,
                )
        elif metric == "knowledge_gaps":
            domains = self._latest_metrics.get("knowledge_gap_domains")
            query = ""
            if isinstance(domains, (list, tuple, set)):
                query = ", ".join(str(d) for d in domains if str(d))
            self._enqueue(
                metric=metric,
                kind="knowledge_update",
                priority=25,
                reason=f"knowledge gaps={value:.0f}",
                metadata={"query": query, "domains": list(domains) if isinstance(domains, (list, tuple, set)) else None},
                now=now,
            )
        else:
            # Default: publish a plan request so agents/humans can act.
            self._enqueue(
                metric=metric,
                kind="plan_request",
                priority=60,
                reason=f"{metric}={value:.3f} not meeting target {goal.target:.3f}",
                metadata={"target": goal.target, "direction": goal.direction},
                now=now,
            )

        goal.last_trigger_ts = float(now)
        goal.failure_streak = 0  # prevent immediate requeue spam

    def _enqueue(self, *, metric: str, kind: str, priority: int, reason: str, metadata: Dict[str, Any], now: float) -> None:
        key = (str(metric), str(kind))
        recent_ts = float(self._recently_enqueued.get(key, 0.0) or 0.0)
        if recent_ts and (now - recent_ts) < 1.0:
            return
        if len(self._queue) >= max(1, int(self._max_queue)):
            return
        item = ImprovementItem(
            item_id=uuid.uuid4().hex,
            metric=str(metric),
            kind=str(kind),
            priority=int(priority),
            reason=str(reason),
            eval_window=max(1, int(self._eval_window)),
            min_improvement=max(0.0, float(self._min_improvement)),
            metadata=dict(metadata or {}),
        )
        self._seq += 1
        heapq.heappush(self._queue, (int(priority), float(now), int(self._seq), item))
        self._recently_enqueued[key] = float(now)

    def _requires_evaluation(self, item: ImprovementItem) -> bool:
        return item.kind in {"decision_exploration_boost", "decision_big_brain", "automl_apply"}

    def _apply_item(
        self,
        item: ImprovementItem,
        *,
        event_bus: Any | None,
        retrain_callback: Callable[[Mapping[str, Any]], Any] | None,
        knowledge_pipeline: Any | None,
        memory_router: Any | None,
        imitation_policy: Any | None,
        predictive_model: Any | None,
        runtime_config: Any | None,
        now: float,
    ) -> Dict[str, float]:
        stats: Dict[str, float] = {"self_improvement_action": 1.0}
        item.status = "active" if self._requires_evaluation(item) else "completed"

        latest_raw = self._latest_metrics.get(item.metric)
        try:
            item.baseline = float(latest_raw) if latest_raw is not None else None
        except Exception:
            item.baseline = None

        # ------------------------------------------------------------------
        if item.kind == "perception_retrain":
            ok = False
            if retrain_callback is not None:
                try:
                    retrain_callback({"module": "perception", "metric": item.metric, "reason": item.reason})
                    stats["self_improvement_retrain_scheduled"] = 1.0
                    ok = True
                except Exception:
                    stats["self_improvement_retrain_failed"] = 1.0
            self._publish_event(event_bus, "self_improvement.item_completed", item, now=now, success=ok)
            self._record_experience(
                item,
                success=ok,
                evaluated=False,
                baseline=item.baseline,
                average=None,
                memory_router=memory_router,
                event_bus=event_bus,
                now=now,
            )
            return stats

        if item.kind == "decision_exploration_boost":
            applied = self._apply_exploration_boost(imitation_policy, magnitude=0.08)
            item.applied.update(applied)
            stats["self_improvement_exploration_boost"] = 1.0 if applied else 0.0
            self._publish_event(event_bus, "self_improvement.item_started", item, now=now, success=True)
            return stats

        if item.kind == "decision_big_brain":
            applied = self._apply_flag(runtime_config, "big_brain", True)
            item.applied.update(applied)
            stats["self_improvement_big_brain"] = 1.0 if applied else 0.0
            self._publish_event(event_bus, "self_improvement.item_started", item, now=now, success=True)
            return stats

        if item.kind == "automl_apply":
            applied = self._apply_automl_params(
                item,
                imitation_policy=imitation_policy,
                predictive_model=predictive_model,
                runtime_config=runtime_config,
            )
            item.applied.update(applied)
            changes = applied.get("automl_changes")
            applied_count = float(len(changes)) if isinstance(changes, Mapping) else 0.0
            stats["self_improvement_automl_apply"] = applied_count
            if applied_count <= 0:
                item.status = "failed"
                objective_value = self._objective_value(item, value=item.baseline)
                self._publish_automl_feedback(
                    event_bus,
                    item,
                    now=now,
                    baseline=item.baseline,
                    average=item.baseline,
                    objective_value=objective_value,
                    success=False,
                    note="no_op",
                )
                self._publish_event(event_bus, "self_improvement.item_completed", item, now=now, success=False)
                self._record_experience(
                    item,
                    success=False,
                    evaluated=False,
                    baseline=item.baseline,
                    average=item.baseline,
                    memory_router=memory_router,
                    event_bus=event_bus,
                    now=now,
                    extra={"note": "no_op"},
                )
                with self._lock:
                    self._active = None
                return stats
            self._publish_event(event_bus, "self_improvement.item_started", item, now=now, success=True)
            return stats

        if item.kind == "knowledge_update":
            ok = self._apply_knowledge_update(
                item,
                knowledge_pipeline=knowledge_pipeline,
                memory_router=memory_router,
            )
            stats["self_improvement_knowledge_update"] = 1.0 if ok else 0.0
            self._publish_event(event_bus, "self_improvement.item_completed", item, now=now, success=ok)
            self._record_experience(
                item,
                success=ok,
                evaluated=False,
                baseline=item.baseline,
                average=None,
                memory_router=memory_router,
                event_bus=event_bus,
                now=now,
            )
            return stats

        if item.kind == "plan_request":
            ok = self._publish_plan_request(event_bus, item)
            stats["self_improvement_plan_request"] = 1.0 if ok else 0.0
            self._publish_event(event_bus, "self_improvement.item_completed", item, now=now, success=ok)
            self._record_experience(
                item,
                success=ok,
                evaluated=False,
                baseline=item.baseline,
                average=None,
                memory_router=memory_router,
                event_bus=event_bus,
                now=now,
            )
            return stats

        # Unknown action kind.
        self._publish_event(event_bus, "self_improvement.item_completed", item, now=now, success=False)
        return stats

    def _maybe_finalize_active(
        self,
        item: ImprovementItem,
        *,
        event_bus: Any | None,
        imitation_policy: Any | None,
        predictive_model: Any | None,
        runtime_config: Any | None,
        memory_router: Any | None,
        now: float,
    ) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        window = max(1, int(item.eval_window))
        if len(item.observations) < window:
            return stats

        baseline = item.baseline
        if baseline is None:
            baseline = item.observations[0]
        average = sum(item.observations[-window:]) / max(window, 1)
        improved = False
        if item.metric and item.metric in self.goals:
            goal = self.goals[item.metric]
            if goal.direction == "decrease":
                improved = average <= float(baseline) - float(item.min_improvement)
            else:
                improved = average >= float(baseline) + float(item.min_improvement)
        else:
            improved = average >= float(baseline) + float(item.min_improvement)

        if item.kind == "automl_apply":
            objective_value = self._objective_value(item, value=average)
            self._publish_automl_feedback(
                event_bus,
                item,
                now=now,
                baseline=baseline,
                average=average,
                objective_value=objective_value,
                success=bool(improved),
            )

        if improved:
            item.status = "completed"
            stats["self_improvement_success"] = 1.0
            self._publish_event(event_bus, "self_improvement.item_completed", item, now=now, success=True)
            self._record_experience(
                item,
                success=True,
                evaluated=True,
                baseline=baseline,
                average=average,
                memory_router=memory_router,
                event_bus=event_bus,
                now=now,
            )
            with self._lock:
                self._active = None
            return stats

        # Not improved -> rollback if possible
        rolled_back = self._rollback(
            item,
            imitation_policy=imitation_policy,
            predictive_model=predictive_model,
            runtime_config=runtime_config,
        )
        item.status = "failed"
        stats["self_improvement_rollback"] = 1.0 if rolled_back else 0.0
        self._publish_event(event_bus, "self_improvement.item_completed", item, now=now, success=False)
        self._record_experience(
            item,
            success=False,
            evaluated=True,
            baseline=baseline,
            average=average,
            memory_router=memory_router,
            event_bus=event_bus,
            now=now,
            extra={"rolled_back": bool(rolled_back)},
        )
        with self._lock:
            self._active = None
        return stats

    def _record_experience(
        self,
        item: ImprovementItem,
        *,
        success: bool,
        evaluated: bool,
        baseline: float | None,
        average: float | None,
        memory_router: Any | None,
        event_bus: Any | None,
        now: float,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(item, ImprovementItem):
            return
        record: Dict[str, Any] = {
            "time": float(now),
            "metric": str(item.metric),
            "metric_group": metric_group(str(item.metric)),
            "kind": str(item.kind),
            "item_id": str(item.item_id),
            "success": bool(success),
            "evaluated": bool(evaluated),
            "baseline": baseline,
            "average": average,
            "direction": str(item.metadata.get("direction") or (self.goals.get(item.metric).direction if item.metric in self.goals else "") or "increase"),
            "target": item.metadata.get("target") if item.metadata else None,
            "reason": str(item.reason or ""),
            "eval_window": int(item.eval_window),
            "min_improvement": float(item.min_improvement),
            "applied": dict(item.applied),
            "metadata": dict(item.metadata),
        }
        if isinstance(extra, Mapping):
            record.update(dict(extra))

        try:
            self._experience_store.record(record)
        except Exception:
            logger.debug("Failed to record improvement experience", exc_info=True)

        if memory_router is not None and hasattr(memory_router, "add_observation"):
            summary = (
                f"self_improvement experience metric={item.metric} kind={item.kind} "
                f"success={int(bool(success))} evaluated={int(bool(evaluated))}"
            )
            try:
                memory_router.add_observation(
                    summary,
                    source="self_improvement_experience",
                    metadata={"experience": record},
                    promote=bool(self._experience_promote),
                )
            except TypeError:
                try:
                    memory_router.add_observation(
                        summary,
                        source="self_improvement_experience",
                        metadata={"experience": record},
                    )
                except Exception:
                    pass
            except Exception:
                pass

        if event_bus is not None and hasattr(event_bus, "publish"):
            try:
                event_bus.publish(
                    "self_improvement.experience_recorded",
                    {
                        "time": float(now),
                        "metric": str(item.metric),
                        "kind": str(item.kind),
                        "success": bool(success),
                        "evaluated": bool(evaluated),
                    },
                )
            except Exception:
                pass

    @staticmethod
    def _publish_event(bus: Any | None, topic: str, item: ImprovementItem, *, now: float, success: bool) -> None:
        if bus is None or not hasattr(bus, "publish"):
            return
        try:
            bus.publish(
                topic,
                {
                    "time": float(now),
                    "item_id": item.item_id,
                    "metric": item.metric,
                    "kind": item.kind,
                    "priority": int(item.priority),
                    "status": item.status,
                    "success": bool(success),
                    "reason": item.reason,
                    "metadata": dict(item.metadata),
                },
            )
        except Exception:
            pass

    @staticmethod
    def _apply_exploration_boost(imitation_policy: Any | None, *, magnitude: float) -> Dict[str, Any]:
        if imitation_policy is None:
            return {}
        cfg = getattr(imitation_policy, "config", None)
        if cfg is None or not hasattr(cfg, "inference_uniform_mix"):
            return {}
        try:
            before = float(getattr(cfg, "inference_uniform_mix", 0.0) or 0.0)
            after = min(1.0, max(0.0, before + float(magnitude)))
            setattr(cfg, "inference_uniform_mix", after)
            return {"imitation_uniform_mix_before": before, "imitation_uniform_mix_after": after}
        except Exception:
            return {}

    @staticmethod
    def _apply_flag(config: Any | None, attr: str, value: Any) -> Dict[str, Any]:
        if config is None or not hasattr(config, attr):
            return {}
        try:
            before = getattr(config, attr)
        except Exception:
            before = None
        try:
            setattr(config, attr, value)
            return {"attr": attr, "before": before, "after": value}
        except Exception:
            return {}

    def _rollback(
        self,
        item: ImprovementItem,
        *,
        imitation_policy: Any | None,
        predictive_model: Any | None,
        runtime_config: Any | None,
    ) -> bool:
        if not item.applied:
            return False
        if item.kind == "decision_exploration_boost":
            cfg = getattr(imitation_policy, "config", None) if imitation_policy is not None else None
            if cfg is None or not hasattr(cfg, "inference_uniform_mix"):
                return False
            before = item.applied.get("imitation_uniform_mix_before")
            if before is None:
                return False
            try:
                setattr(cfg, "inference_uniform_mix", float(before))
                return True
            except Exception:
                return False
        if item.kind == "decision_big_brain":
            attr = str(item.applied.get("attr") or "big_brain")
            before = item.applied.get("before")
            if runtime_config is None or not hasattr(runtime_config, attr):
                return False
            try:
                setattr(runtime_config, attr, before)
                return True
            except Exception:
                return False
        if item.kind == "automl_apply":
            changes = item.applied.get("automl_changes")
            if not isinstance(changes, Mapping):
                return False
            ok = False
            for name, change in changes.items():
                if not isinstance(change, Mapping):
                    continue
                before = change.get("before")
                if name == "imitation.lr":
                    cfg = getattr(imitation_policy, "config", None) if imitation_policy is not None else None
                    if cfg is None or not hasattr(cfg, "lr") or before is None:
                        continue
                    try:
                        setattr(cfg, "lr", float(before))
                        ok = True
                    except Exception:
                        continue
                elif name == "imitation.inference_uniform_mix":
                    cfg = getattr(imitation_policy, "config", None) if imitation_policy is not None else None
                    if cfg is None or not hasattr(cfg, "inference_uniform_mix") or before is None:
                        continue
                    try:
                        setattr(cfg, "inference_uniform_mix", float(before))
                        ok = True
                    except Exception:
                        continue
                elif name == "runtime.big_brain":
                    if runtime_config is None or not hasattr(runtime_config, "big_brain"):
                        continue
                    try:
                        setattr(runtime_config, "big_brain", before)
                        ok = True
                    except Exception:
                        continue
                elif name == "predictive.reconstruction_lr":
                    if predictive_model is None or before is None:
                        continue
                    if hasattr(predictive_model, "set_learning_rates"):
                        try:
                            predictive_model.set_learning_rates(reconstruction_lr=float(before))  # type: ignore[call-arg]
                            ok = True
                        except Exception:
                            continue
                elif name == "predictive.prediction_lr":
                    if predictive_model is None or before is None:
                        continue
                    if hasattr(predictive_model, "set_learning_rates"):
                        try:
                            predictive_model.set_learning_rates(prediction_lr=float(before))  # type: ignore[call-arg]
                            ok = True
                        except Exception:
                            continue
            return ok
        return False

    def _apply_automl_params(
        self,
        item: ImprovementItem,
        *,
        imitation_policy: Any | None,
        predictive_model: Any | None,
        runtime_config: Any | None,
    ) -> Dict[str, Any]:
        params = item.metadata.get("params")
        if not isinstance(params, Mapping):
            return {}

        changes: Dict[str, Dict[str, Any]] = {}
        rejected: List[str] = []

        for raw_name, raw_value in params.items():
            name = str(raw_name).strip()
            if not name:
                continue

            if name == "imitation.lr":
                cfg = getattr(imitation_policy, "config", None) if imitation_policy is not None else None
                if cfg is None or not hasattr(cfg, "lr"):
                    rejected.append(name)
                    continue
                try:
                    before = float(getattr(cfg, "lr"))
                    after = float(raw_value)
                except Exception:
                    rejected.append(name)
                    continue
                after = max(1e-6, min(1.0, after))
                try:
                    setattr(cfg, "lr", after)
                    changes[name] = {"before": before, "after": after}
                except Exception:
                    rejected.append(name)
                continue

            if name == "imitation.inference_uniform_mix":
                cfg = getattr(imitation_policy, "config", None) if imitation_policy is not None else None
                if cfg is None or not hasattr(cfg, "inference_uniform_mix"):
                    rejected.append(name)
                    continue
                try:
                    before = float(getattr(cfg, "inference_uniform_mix", 0.0) or 0.0)
                    after = float(raw_value)
                except Exception:
                    rejected.append(name)
                    continue
                after = max(0.0, min(1.0, after))
                try:
                    setattr(cfg, "inference_uniform_mix", after)
                    changes[name] = {"before": before, "after": after}
                except Exception:
                    rejected.append(name)
                continue

            if name == "runtime.big_brain":
                if runtime_config is None or not hasattr(runtime_config, "big_brain"):
                    rejected.append(name)
                    continue
                before = getattr(runtime_config, "big_brain", None)
                after = bool(raw_value)
                try:
                    setattr(runtime_config, "big_brain", after)
                    changes[name] = {"before": before, "after": after}
                except Exception:
                    rejected.append(name)
                continue

            if name in {"predictive.reconstruction_lr", "predictive.prediction_lr"}:
                if predictive_model is None or not hasattr(predictive_model, "set_learning_rates"):
                    rejected.append(name)
                    continue
                current = {}
                if hasattr(predictive_model, "learning_rates"):
                    try:
                        current = predictive_model.learning_rates()  # type: ignore[call-arg]
                    except Exception:
                        current = {}
                before = None
                if name == "predictive.reconstruction_lr":
                    before = current.get("learning_rate")
                else:
                    before = current.get("prediction_learning_rate")
                try:
                    after = float(raw_value)
                except Exception:
                    rejected.append(name)
                    continue
                after = max(1e-6, min(1.0, after))
                try:
                    if name == "predictive.reconstruction_lr":
                        predictive_model.set_learning_rates(reconstruction_lr=after)  # type: ignore[call-arg]
                    else:
                        predictive_model.set_learning_rates(prediction_lr=after)  # type: ignore[call-arg]
                    changes[name] = {"before": before, "after": after}
                except Exception:
                    rejected.append(name)
                continue

            rejected.append(name)

        payload: Dict[str, Any] = {}
        if changes:
            payload["automl_changes"] = changes
        if rejected:
            payload["automl_rejected"] = rejected
        return payload

    def _objective_value(self, item: ImprovementItem, *, value: float | None) -> float | None:
        if value is None:
            return None
        direction = str(item.metadata.get("direction") or "").strip().lower()
        if not direction and item.metric in self.goals:
            direction = str(self.goals[item.metric].direction).strip().lower()
        direction = direction or "increase"
        try:
            numeric = float(value)
        except Exception:
            return None
        return -numeric if direction == "decrease" else numeric

    @staticmethod
    def _publish_automl_feedback(
        bus: Any | None,
        item: ImprovementItem,
        *,
        now: float,
        baseline: float | None,
        average: float | None,
        objective_value: float | None,
        success: bool,
        note: str | None = None,
    ) -> None:
        if bus is None or not hasattr(bus, "publish"):
            return
        suggestion_id = str(item.metadata.get("suggestion_id") or "").strip()
        if not suggestion_id or not item.metric:
            return
        score = 0.0
        try:
            if objective_value is not None:
                score = float(objective_value)
        except Exception:
            score = 0.0

        payload: Dict[str, Any] = {
            "time": float(now),
            "suggestion_id": suggestion_id,
            "metric": str(item.metric),
            "objective_value": score,
            "baseline": baseline,
            "average": average,
            "success": bool(success),
            "params": dict(item.metadata.get("params") or {}),
            "source": "self_improvement",
        }
        if note:
            payload["note"] = str(note)
        try:
            bus.publish("automl.feedback", payload)
        except Exception:
            pass

    def _apply_knowledge_update(self, item: ImprovementItem, *, knowledge_pipeline: Any | None, memory_router: Any | None) -> bool:
        query = str(item.metadata.get("query") or "").strip()
        if not query:
            domains = item.metadata.get("domains")
            if isinstance(domains, list) and domains:
                query = ", ".join(str(d) for d in domains if str(d).strip())
        if not query:
            query = item.metric

        statements: list[str] = []
        hits: list[dict] = []
        try:
            from modules.knowledge.research_tool import ResearchTool  # type: ignore

            tool = ResearchTool(workspace_root=os.getenv("WORKSPACE_ROOT") or os.getcwd())
            for kw in [token.strip() for token in re.split(r"[,;\\s]+", query) if token.strip()][:4]:
                for hit in tool.query_docs(kw, max_results=2):
                    payload = hit.to_dict() if hasattr(hit, "to_dict") else {"path": getattr(hit, "path", ""), "snippet": getattr(hit, "snippet", "")}
                    hits.append(payload)
        except Exception:
            hits = []

        for hit in hits[:6]:
            path = str(hit.get("path") or "").strip()
            snippet = str(hit.get("snippet") or "").strip()
            if not snippet:
                continue
            statements.append(f"[doc]{path}: {snippet}")

        if memory_router is not None and hasattr(memory_router, "add_observation"):
            try:
                memory_router.add_observation(
                    f"self_improvement knowledge_update query={query} hits={len(hits)}",
                    source="self_improvement",
                    metadata={"query": query, "hits": hits, "time": time.time()},
                )
            except Exception:
                pass

        if knowledge_pipeline is not None and hasattr(knowledge_pipeline, "process_task_event") and statements:
            try:
                knowledge_pipeline.process_task_event(
                    {
                        "task_id": f"self_improvement:{item.item_id}",
                        "summary": f"Knowledge update for query: {query}",
                        "knowledge_statements": statements,
                        "metadata": {"source": "self_improvement", "query": query, "hits": hits},
                    }
                )
                return True
            except Exception:
                return False
        return bool(statements)

    @staticmethod
    def _publish_plan_request(event_bus: Any | None, item: ImprovementItem) -> bool:
        if event_bus is None or not hasattr(event_bus, "publish"):
            return False
        goal = f"Self-improvement: address metric '{item.metric}' ({item.reason})"
        tasks = [
            "Inspect recent metrics and failure traces for this metric.",
            "Choose one concrete remediation and define a quick validation step.",
        ]
        try:
            event_bus.publish("planner.plan_ready", {"goal": goal, "tasks": tasks, "source": "self_improvement"})
            return True
        except Exception:
            return False
