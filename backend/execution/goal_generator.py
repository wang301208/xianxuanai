"""Goal generation module.

This module expands the original outcome-driven :class:`GoalGenerator` with
environment-aware goal discovery. The generator now listens for signals from
the :class:`~backend.world_model.WorldModel` and optional external event feeds
to surface new high-level goals whenever opportunities or risks are detected.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional

from ..memory import LongTermMemory
from ..reflection import ReflectionModule, ReflectionResult
from ..self_monitoring import SelfMonitoringSystem, StepReport
from ..world_model import WorldModel

logger = logging.getLogger(__name__)


EnvironmentAnalyzer = Callable[
    [List["EnvironmentSignal"], Dict[str, Any]],
    Iterable[str],
]


@dataclass
class EnvironmentSignal:
    """Structured representation of an environment observation."""

    source: str
    description: str
    severity: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        """Stable key used for deduplication."""

        origin = self.metadata.get("id") or self.metadata.get("uid")
        if origin:
            return str(origin)
        return f"{self.source}:{hash(self.description)}"


class GoalListener:
    """Continuously monitor environment sources and feed signals to the generator."""

    def __init__(
        self,
        world_model: WorldModel,
        generator: "GoalGenerator",
        *,
        event_bus: Any | None = None,
        poll_interval: float = 30.0,
        topics: Iterable[str] | None = None,
        start: bool = True,
    ) -> None:
        self._world_model = world_model
        self._generator = generator
        self._event_bus = event_bus
        self._poll_interval = max(5.0, poll_interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._topics = tuple(topics or ("environment.update", "environment.alert"))
        self._subscriptions: List[Callable[[], None]] = []
        if self._event_bus:
            for topic in self._topics:
                cancel = self._event_bus.subscribe(topic, self._handle_event)
                if cancel:
                    self._subscriptions.append(cancel)
        if start:
            self.start()

    @property
    def poll_interval(self) -> float:
        return self._poll_interval

    def set_poll_interval(self, interval: float) -> None:
        """Dynamically adjust the polling cadence."""

        self._poll_interval = max(5.0, float(interval))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
        for cancel in self._subscriptions:
            try:
                cancel()
            except Exception:  # pragma: no cover - defensive unsubscribe
                logger.debug("Failed to cancel goal listener subscription", exc_info=True)
        self._subscriptions.clear()

    async def _handle_event(self, event: Dict[str, Any]) -> None:
        description = self._extract_description(event)
        if not description:
            return
        severity = self._safe_float(event.get("severity"), default=0.5)
        source = str(event.get("source") or event.get("topic") or "environment")
        self._generator.observe_environment(
            description,
            source=source,
            severity=severity,
            metadata=event,
        )

    def evaluate_once(self) -> None:
        """Public helper for tests to run a single evaluation cycle."""

        self._collect_world_state()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._collect_world_state()
            self._stop.wait(self._poll_interval)

    def _collect_world_state(self) -> None:
        try:
            state = self._world_model.get_state()
        except Exception:  # pragma: no cover - defensive
            logger.warning("GoalListener failed to read world state", exc_info=True)
            return
        self._generator.process_world_state(state)

    @staticmethod
    def _extract_description(event: Dict[str, Any]) -> str | None:
        for key in ("detail", "description", "summary", "message"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _safe_float(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default


@dataclass
class GoalGenerator:
    """Generate new goals from past outcomes and environment signals."""

    reflection: ReflectionModule
    memory: LongTermMemory
    world_model: WorldModel | None
    event_bus: Any | None
    environment_analyzer: EnvironmentAnalyzer | None = None
    listener_poll_interval: float = 30.0
    start_listener: bool = True
    max_signal_history: int = 50
    dedupe_window: float = 900.0
    intrinsic_motivation: bool = True
    simulation_horizon: int = 3

    def __init__(
        self,
        reflection: Optional[ReflectionModule] = None,
        memory: Optional[LongTermMemory] = None,
        *,
        world_model: Optional[WorldModel] = None,
        event_bus: Any | None = None,
        environment_analyzer: EnvironmentAnalyzer | None = None,
        listener_poll_interval: float = 30.0,
        start_listener: bool = True,
        max_signal_history: int = 50,
        dedupe_window: float = 900.0,
        intrinsic_motivation: bool = True,
        simulation_horizon: int = 3,
    ) -> None:
        self.reflection = reflection or ReflectionModule()
        self.memory = memory or LongTermMemory(":memory:")
        self.world_model = world_model
        self.event_bus = event_bus
        self.environment_analyzer = environment_analyzer
        self.listener_poll_interval = listener_poll_interval
        self.start_listener = start_listener
        self.max_signal_history = max_signal_history
        self.dedupe_window = dedupe_window
        self.intrinsic_motivation = intrinsic_motivation
        self.simulation_horizon = max(1, int(simulation_horizon))
        monitor_threshold = self.reflection.quality_threshold or 0.6
        self.monitor = SelfMonitoringSystem(
            reflection=self.reflection,
            memory=self.memory,
            quality_threshold=monitor_threshold,
            adjust_strategy=self._monitor_adjustments,
            recovery_hook=self._monitor_recovery,
        )
        self._pending_goals: Deque[str] = deque()
        self._signals: Deque[EnvironmentSignal] = deque()
        self._last_seen: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._listener: GoalListener | None = None
        self._latest_world_state: Dict[str, Any] = {}
        if self.world_model and self.start_listener:
            self._listener = GoalListener(
                self.world_model,
                self,
                event_bus=self.event_bus,
                poll_interval=self.listener_poll_interval,
            )

    @property
    def listener(self) -> GoalListener | None:
        """Return the active goal listener if one is running."""

        return self._listener

    def set_listener_poll_interval(self, interval: float) -> None:
        """Update the listener polling interval and apply it if active."""

        self.listener_poll_interval = max(5.0, float(interval))
        if self._listener:
            self._listener.set_poll_interval(self.listener_poll_interval)

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Stop background listeners and release resources."""

        if self._listener:
            self._listener.stop()
        self._listener = None

    # ------------------------------------------------------------------
    def observe_environment(
        self,
        description: str,
        *,
        source: str = "environment",
        severity: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an external observation that may lead to a new goal."""

        metadata = dict(metadata or {})
        signal = EnvironmentSignal(
            source=source,
            description=description.strip(),
            severity=max(0.0, min(1.0, severity)),
            metadata=metadata,
        )
        if not signal.description:
            return

        with self._lock:
            self._signals.append(signal)
            if len(self._signals) > self.max_signal_history:
                self._signals.popleft()

        self._memory_event(signal)
        if self.world_model:
            combined_meta = {"source": signal.source}
            if metadata:
                combined_meta.update(metadata)
            self.world_model.record_opportunity(
                signal.description,
                weight=signal.severity,
                metadata=combined_meta,
            )
        goal = self._analyze_signal(signal)
        if goal:
            self._schedule_goal(goal, key=f"signal:{signal.key()}")

    def process_world_state(self, state: Dict[str, Any]) -> None:
        """Inspect world model state and enqueue any relevant goals."""

        self._latest_world_state = state
        now = time.time()

        tasks = state.get("tasks") or {}
        if isinstance(tasks, dict):
            for task_id, metadata in tasks.items():
                if not isinstance(metadata, dict):
                    continue
                self._evaluate_task(task_id, metadata, now)

        resources = state.get("resources") or {}
        if isinstance(resources, dict):
            self._evaluate_resources(resources, now)

        actions = state.get("actions") or []
        if isinstance(actions, list):
            self._evaluate_actions(actions, now)

    def generate(self) -> Optional[str]:
        """Produce the next goal from pending signals or memory reflections."""

        with self._lock:
            if self._pending_goals:
                return self._pending_goals.popleft()

        goal = self._generate_from_signals()
        if goal:
            return goal

        intrinsic = self._generate_intrinsic_goal()
        if intrinsic:
            return intrinsic

        outcomes = list(self.memory.get("outcome"))
        if not outcomes:
            return None
        last = outcomes[-1]
        _, revised = self.reflection.reflect(last)
        return f"Build on: {revised}"

    # ------------------------------------------------------------------
    def _generate_from_signals(self) -> Optional[str]:
        signal = self._pop_signal()
        if not signal:
            return None
        goal = self._analyze_signal(signal)
        if goal:
            return goal
        return None

    def _generate_intrinsic_goal(self) -> Optional[str]:
        if not self.intrinsic_motivation or not self.world_model:
            return None

        targets = self.world_model.suggest_learning_targets(limit=1)
        if targets:
            domain = targets[0]
            projection = []
            try:
                projection = self.world_model.simulate(
                    [
                        {
                            "domain": domain,
                            "agent_id": "self",
                            "estimated_load": 6.0,
                            "learning_rate": 0.12,
                            "discover": f"advance {domain}",
                        }
                    ],
                    horizon=min(self.simulation_horizon, 3),
                )
            except Exception:
                logger.debug("World model simulation failed for intrinsic goal", exc_info=True)
            expected: Optional[float] = None
            if projection:
                last_state = projection[-1]
                comp_block = last_state.get("competence", {}).get(domain)
                if isinstance(comp_block, dict):
                    expected = float(comp_block.get("score", 0.0))
            if expected is not None:
                return (
                    f"Self-improve in {domain} to reach competence≈{expected:.2f} "
                    "via focused practice informed by simulations."
                )
            return (
                f"Self-improve in {domain} through deliberate practice sessions "
                "guided by world model projections."
            )

        opportunities = self.world_model.suggest_opportunities(limit=1)
        if opportunities:
            topic = opportunities[0]["topic"]
            return f"Explore emerging opportunity: {topic}"

        return None

    def _pop_signal(self) -> EnvironmentSignal | None:
        with self._lock:
            if not self._signals:
                return None
            highest_index = max(
                range(len(self._signals)),
                key=lambda idx: (
                    self._signals[idx].severity,
                    self._signals[idx].timestamp,
                ),
            )
            signal = self._signals[highest_index]
            del self._signals[highest_index]
            return signal

    def _schedule_goal(self, goal: str, *, key: Optional[str] = None) -> None:
        if not goal:
            return
        with self._lock:
            if key:
                last = self._last_seen.get(key)
                if last and time.time() - last < self.dedupe_window:
                    return
                self._last_seen[key] = time.time()
            self._pending_goals.append(goal)

    def _evaluate_task(self, task_id: str, metadata: Dict[str, Any], now: float) -> None:
        status = str(metadata.get("status", "")).lower()
        if status in {"done", "completed", "complete", "success"}:
            return

        description = metadata.get("description") or metadata.get("summary") or ""
        deadline = self._parse_timestamp(
            metadata.get("deadline") or metadata.get("due") or metadata.get("due_at")
        )
        updated_at = self._parse_timestamp(
            metadata.get("updated_at") or metadata.get("timestamp") or metadata.get("created_at")
        )
        blockage = metadata.get("blocked") or metadata.get("is_blocked")

        if blockage or status in {"blocked", "stalled", "failed"}:
            goal = (
                f"Unblock task '{task_id}'. Context: {description}".strip()
                or f"Resolve blockage for task '{task_id}'"
            )
            self._schedule_goal(goal, key=f"task:block:{task_id}")
            return

        if deadline is not None and deadline - now < 1800:
            goal = (
                f"Accelerate task '{task_id}' to meet deadline. Context: {description}".strip()
            )
            self._schedule_goal(goal, key=f"task:deadline:{task_id}")

        if updated_at is not None and now - updated_at > 3600:
            goal = (
                f"Review stalled task '{task_id}' and plan next steps. Context: {description}".strip()
            )
            self._schedule_goal(goal, key=f"task:review:{task_id}")

    def _evaluate_resources(self, resources: Dict[str, Dict[str, Any]], now: float) -> None:
        for agent_id, usage in resources.items():
            if not isinstance(usage, dict):
                continue
            cpu = self._safe_float(usage.get("cpu"))
            memory = self._safe_float(usage.get("memory"))
            cpu_pred = self._safe_float(usage.get("cpu_pred"), default=cpu)
            mem_pred = self._safe_float(usage.get("memory_pred"), default=memory)
            quota = usage.get("quota") or {}
            cpu_quota = self._safe_float(quota.get("cpu"), default=100.0)
            mem_quota = self._safe_float(quota.get("memory"), default=100.0)

            if cpu_pred > cpu_quota or mem_pred > mem_quota:
                goal = (
                    f"Mitigate resource pressure for agent '{agent_id}' "
                    f"(cpu≈{cpu_pred:.1f}%, mem≈{mem_pred:.1f}%)."
                )
                self._schedule_goal(goal, key=f"resource:overload:{agent_id}")
                continue

            if cpu < 20.0 and memory < 30.0:
                goal = (
                    f"Assign high-value work to idle agent '{agent_id}' "
                    "to leverage available capacity."
                )
                self._schedule_goal(goal, key=f"resource:idle:{agent_id}")

    def _evaluate_actions(self, actions: Iterable[Dict[str, Any]], now: float) -> None:
        recent = list(actions)[-5:]
        for record in recent:
            if not isinstance(record, dict):
                continue
            detailed = any(field in record for field in ("status", "result", "error", "metrics"))
            agent_id = record.get("agent_id")
            agent_id_str = str(agent_id or "unknown")
            if detailed:
                report = self._build_step_report(record, agent_id_str)
                decision = self.monitor.assess_step(report)
                if decision.should_retry:
                    recommendation = decision.revision.strip()
                    goal_text: Optional[str] = None
                    if decision.adjustments:
                        retry_goal = decision.adjustments.get("retry_goal")
                        if isinstance(retry_goal, str) and retry_goal.strip():
                            goal_text = retry_goal.strip()
                    if not goal_text:
                        base = f"Retry action '{report.action or 'task'}' for agent '{agent_id_str}'"
                        goal_text = f"{base}. Recommendation: {recommendation}" if recommendation else base
                    dedupe_key = f"action:retry:{agent_id_str}:{abs(hash(report.action or 'unknown'))}"
                    self._schedule_goal(goal_text, key=dedupe_key)
                continue

            action = str(record.get("action") or "").lower()
            if "error" in action or "fail" in action:
                goal = (
                    f"Investigate repeated failures reported by agent '{agent_id_str}' "
                    f"({record.get('action')})."
                )
                self._schedule_goal(goal, key=f"action:failure:{agent_id_str}")

    def _analyze_signal(self, signal: EnvironmentSignal) -> Optional[str]:
        analyzer = self.environment_analyzer or self._default_analyzer
        try:
            goals = list(
                filter(
                    None,
                    analyzer([signal], self._latest_world_state or {}),
                )
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("Environment analyzer failed", exc_info=True)
            return None
        return goals[0] if goals else None

    def _build_step_report(self, record: Dict[str, Any], agent_id: str) -> StepReport:
        metrics: Dict[str, float] = {}
        raw_metrics = record.get("metrics")
        if isinstance(raw_metrics, dict):
            for key, value in raw_metrics.items():
                try:
                    metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        observation = (
            record.get("result")
            or record.get("observation")
            or record.get("error")
            or record.get("action")
            or ""
        )

        metadata: Dict[str, Any] = {}
        if isinstance(record.get("metadata"), dict):
            metadata.update(record["metadata"])
        for key in ("agent_id", "task_id", "goal"):
            if record.get(key) is not None:
                metadata.setdefault(key, record[key])
        metadata.setdefault("agent_id", agent_id)

        return StepReport(
            action=str(record.get("action") or ""),
            observation=str(observation),
            status=str(record.get("status") or "unknown"),
            metrics=metrics,
            error=str(record.get("error")) if record.get("error") else None,
            retries=int(record.get("retries") or 0),
            metadata=metadata,
        )

    def _monitor_adjustments(self, report: StepReport, evaluation: ReflectionResult) -> Dict[str, Any]:
        adjustments: Dict[str, Any] = {}
        severity = max(0.1, round(1.0 - evaluation.confidence, 3))
        adjustments["severity"] = severity
        focus = report.metadata.get("goal") or report.metadata.get("task_id") or report.action or "workstream"
        agent = report.metadata.get("agent_id", "unknown")
        adjustments["retry_goal"] = (
            f"Retry action '{focus}' for agent '{agent}' with updated plan (severity={severity:.2f})."
        )
        if report.error:
            adjustments["diagnostics"] = report.error
        if report.metrics:
            slow_metric = max(report.metrics.items(), key=lambda item: item[1])[0]
            adjustments["focus_metric"] = slow_metric
        if self.world_model:
            self.world_model.update_competence(
                str(report.metadata.get("domain") or focus),
                evaluation.confidence,
                source="reflection",
                metadata={"from_monitor": True, "agent": agent},
            )
            if evaluation.confidence < 0.5:
                self.world_model.record_opportunity(
                    f"Investigate {focus}",
                    weight=max(0.3, 1.0 - evaluation.confidence),
                    metadata={"origin": "monitor", "agent": agent},
                )
        return adjustments

    def _monitor_recovery(self, report: StepReport, evaluation: ReflectionResult) -> None:
        agent = str(report.metadata.get("agent_id", "unknown"))
        focus = report.metadata.get("goal") or report.metadata.get("task_id") or report.action or "task"
        message = (
            f"Diagnose failure for agent '{agent}' on '{focus}'. "
            f"Reflection: {evaluation.sentiment} (conf={evaluation.confidence:.2f})."
        )
        dedupe_key = f"action:diagnose:{agent}:{abs(hash(focus))}"
        self._schedule_goal(message, key=dedupe_key)
        if self.world_model:
            self.world_model.record_opportunity(
                f"Deep-dive on {focus}",
                weight=max(0.3, 1.0 - evaluation.confidence),
                metadata={"origin": "recovery", "agent": agent},
            )

    def _default_analyzer(
        self,
        signals: List[EnvironmentSignal],
        _: Dict[str, Any],
    ) -> Iterable[str]:
        for signal in signals:
            prefix = "Respond to" if signal.severity >= 0.7 else "Investigate"
            yield f"{prefix} {signal.source}: {signal.description}"

    def _memory_event(self, signal: EnvironmentSignal) -> None:
        try:
            tags = [signal.source]
            if signal.severity >= 0.7:
                tags.append("high")
            self.memory.store(
                signal.description,
                metadata={"category": "environment_event", "tags": tags},
            )
        except Exception:
            logger.debug("Failed to persist environment event", exc_info=True)

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                try:
                    return datetime.fromisoformat(value).timestamp()
                except ValueError:
                    return None
        return None

    @staticmethod
    def _safe_float(value: Any, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default


__all__ = ["EnvironmentSignal", "GoalGenerator", "GoalListener"]
