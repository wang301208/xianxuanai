"""Supervisory conductor coordinating agent decisions and task health."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from events import EventBus
from events.coordination import TaskStatus
from reasoning.decision_engine import ActionDirective, DecisionEngine

if TYPE_CHECKING:  # pragma: no cover - hints only
    from third_party.autogpt.autogpt.agents.base import BaseAgent


logger = logging.getLogger(__name__)


class AgentConductor:
    """Monitor agent activity and inject supervisory directives when needed."""

    def __init__(
        self,
        event_bus: EventBus,
        decision_engine_factory: Callable[[], DecisionEngine] | None = None,
        *,
        max_consecutive_errors: int = 3,
    ) -> None:
        self._event_bus = event_bus
        self._decision_engine_factory = decision_engine_factory or DecisionEngine
        self._max_consecutive_errors = max(1, max_consecutive_errors)
        self._agents: Dict[str, BaseAgent] = {}
        self._decision_engines: Dict[str, DecisionEngine] = {}
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._subscriptions = [
            self._event_bus.subscribe("agent.status", self._handle_task_status),
            self._event_bus.subscribe("agent.action.outcome", self._handle_action_outcome),
        ]

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def create_decision_engine(self, agent_id: str) -> DecisionEngine:
        """Return a decision engine instance dedicated to *agent_id*."""

        engine = self._decision_engine_factory()
        self._decision_engines[agent_id] = engine
        return engine

    def register_agent(self, agent: "BaseAgent") -> None:
        """Track *agent* and ensure it is supervised."""

        agent_id = agent.state.agent_id
        self._agents[agent_id] = agent
        self._error_counts[agent_id] = 0
        if agent.decision_engine is None:
            engine = self._decision_engines.get(agent_id) or self._decision_engine_factory()
            agent.decision_engine = engine
            agent.config.use_decision_engine = True
            self._decision_engines[agent_id] = engine
        logger.debug("AgentConductor registered agent %s", agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Stop supervising *agent_id*."""

        self._agents.pop(agent_id, None)
        self._decision_engines.pop(agent_id, None)
        self._error_counts.pop(agent_id, None)

    def close(self) -> None:
        """Detach from the event bus."""

        for cancel in self._subscriptions:
            try:
                cancel()
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to cancel conductor subscription", exc_info=True)
        self._subscriptions.clear()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    async def _handle_task_status(self, event: Dict[str, Any]) -> None:
        agent_id = event.get("agent_id")
        if not agent_id or agent_id not in self._agents:
            return

        status_value = event.get("status")
        try:
            status = TaskStatus(status_value)
        except Exception:
            logger.debug("Unknown task status '%s' received for agent %s", status_value, agent_id)
            return

        if status == TaskStatus.FAILED:
            detail = event.get("detail") or event.get("summary")
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            error_reason = metadata.get("error_reason") or detail
            if error_reason and "rate limit" in error_reason.lower():
                logger.info(
                    "Transient failure reported by %s due to rate limits; allowing retry.",
                    agent_id,
                )
                self._error_counts[agent_id] = 0
                return

            replan_metadata = {"source": "task_status"}
            if metadata:
                for key in ("error_reason", "error", "failing_command", "failing_command_args", "error_context"):
                    if key in metadata:
                        replan_metadata[key] = metadata[key]
            self._request_replan(
                agent_id,
                error_reason
                or "Task execution failed; supervisor requests replanning.",
                metadata=replan_metadata,
            )
            self._error_counts[agent_id] = 0
        elif status == TaskStatus.COMPLETED:
            self._error_counts[agent_id] = 0

        metadata = event.get("metadata") or {}
        if isinstance(metadata, dict) and metadata.get("goal_deviation"):
            reason = metadata.get("goal_deviation_reason") or "Goal deviation detected."
            self._request_replan(
                agent_id,
                reason,
                metadata={"source": "goal_deviation", "details": metadata},
            )

    async def _handle_action_outcome(self, event: Dict[str, Any]) -> None:
        agent_id = event.get("agent")
        if not agent_id or agent_id not in self._agents:
            return

        directive_info = event.get("directive")
        if isinstance(directive_info, dict) and directive_info.get("requires_replan"):
            self._error_counts[agent_id] = 0
            return

        status = (event.get("status") or "").lower()
        reward_value = event.get("reward")
        try:
            reward_value = float(reward_value)
        except Exception:
            reward_value = None
        if status == "error":
            error_reason = event.get("error_reason")
            error_info = event.get("error") if isinstance(event.get("error"), dict) else None
            command_args = event.get("command_args") if isinstance(event.get("command_args"), dict) else None

            if error_reason and "do not execute this command again" in error_reason.lower():
                metadata = {
                    "source": "error_monitor",
                    "error_reason": error_reason,
                    "last_command": event.get("command"),
                }
                if command_args is not None:
                    metadata["last_command_args"] = command_args
                if error_info is not None:
                    metadata["error"] = error_info
                self._request_replan(
                    agent_id,
                    error_reason,
                    metadata=metadata,
                )
                self._error_counts[agent_id] = 0
                return

            fatal_error_types = {"PermissionError", "FileNotFoundError"}
            if error_info and error_info.get("exception_type") in fatal_error_types:
                metadata = {
                    "source": "error_monitor",
                    "error": error_info,
                    "last_command": event.get("command"),
                }
                if command_args is not None:
                    metadata["last_command_args"] = command_args
                reason = error_reason or (
                    f"{error_info['exception_type']} encountered; replanning recommended."
                )
                self._request_replan(agent_id, reason, metadata=metadata)
                self._error_counts[agent_id] = 0
                return

            self._error_counts[agent_id] = self._error_counts.get(agent_id, 0) + 1
            if self._error_counts[agent_id] >= self._max_consecutive_errors:
                metadata = {
                    "source": "error_monitor",
                    "error_count": self._error_counts[agent_id],
                    "last_command": event.get("command"),
                }
                if error_reason:
                    metadata["error_reason"] = error_reason
                if command_args is not None:
                    metadata["last_command_args"] = command_args
                if error_info is not None:
                    metadata["error"] = error_info
                self._request_replan(
                    agent_id,
                    error_reason
                    or "Repeated command errors detected; replanning recommended.",
                    metadata=metadata,
                )
                self._error_counts[agent_id] = 0
                self._record_decision_metric(agent_id, success=False, reward=reward_value)
        else:
            self._error_counts[agent_id] = 0
            self._record_decision_metric(agent_id, success=True, reward=reward_value)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _request_replan(
        self, agent_id: str, reason: str, *, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        agent = self._agents.get(agent_id)
        if not agent:
            return

        directive = ActionDirective.replan(reason, metadata=metadata)
        agent.queue_directive(directive)
        logger.info("Conductor requested replanning for agent %s: %s", agent_id, reason)
        payload = {
            "agent": agent_id,
            "directive": directive.to_dict(),
            "source": "conductor",
        }
        try:
            engine = self._decision_engines.get(agent_id)
            if engine is not None:
                engine.metrics = getattr(engine, "metrics", {})
                engine.metrics["replans"] = engine.metrics.get("replans", 0) + 1
        except Exception:
            pass
        try:
            self._event_bus.publish("agent.conductor.directive", payload)
        except Exception:
            logger.debug("Failed to publish conductor directive", exc_info=True)

    def _record_decision_metric(self, agent_id: str, *, success: bool, reward: float | None = None) -> None:
        try:
            engine = self._decision_engines.get(agent_id)
            if engine is None:
                return
            metrics = getattr(engine, "metrics", {})
            metrics["decisions"] = metrics.get("decisions", 0) + 1
            if success:
                metrics["successes"] = metrics.get("successes", 0) + 1
            metrics["success_rate"] = metrics.get("successes", 0) / max(1, metrics.get("decisions", 0))
            if reward is not None:
                total = metrics.get("reward_total", 0.0) + float(reward)
                metrics["reward_total"] = total
                metrics["reward_avg"] = total / max(1, metrics.get("decisions", 0))
            engine.metrics = metrics
            manager = getattr(self, "_manager", None)
            if manager is not None:
                update_fn = getattr(manager, "_update_self_improvement_from_metrics", None)
                if callable(update_fn):
                    update_fn(
                        {
                            "decision_success_rate": metrics.get("success_rate", 0.0),
                            "decision_reward_avg": metrics.get("reward_avg", 0.0),
                        }
                    )
        except Exception:
            pass


__all__ = ["AgentConductor"]
