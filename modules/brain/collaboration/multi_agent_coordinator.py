from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

try:  # pragma: no cover - optional dependency
    from backend.world_model import WorldModel
except Exception:  # pragma: no cover - fallback when heavy deps missing
    @dataclass
    class WorldModel:  # type: ignore[redefinition]
        """Minimal stub used when the full world model is unavailable."""

        def __init__(self) -> None:
            self.tasks: Dict[str, Dict[str, Any]] = {}
            self.actions: list[Dict[str, str]] = []

        def add_task(self, task_id: str, metadata: Dict[str, Any]) -> None:
            self.tasks[task_id] = metadata

        def record_action(
            self,
            agent_id: str,
            action: str,
            *,
            status: str | None = None,
            result: str | None = None,
            error: str | None = None,
            metrics: Dict[str, float] | None = None,
            retries: int = 0,
            metadata: Dict[str, Any] | None = None,
        ) -> None:
            record: Dict[str, Any] = {"agent_id": agent_id, "action": action}
            if status is not None:
                record["status"] = status
            if result is not None:
                record["result"] = result
            if error is not None:
                record["error"] = error
            if retries:
                record["retries"] = retries
            if metrics:
                record["metrics"] = dict(metrics)
            if metadata:
                record["metadata"] = dict(metadata)
            self.actions.append(record)

        def get_state(self) -> Dict[str, Any]:
            return {"tasks": dict(self.tasks), "actions": list(self.actions)}

from modules.brain.message_bus import (
    publish_neural_event,
    reset_message_bus,
    subscribe_to_brain_region,
)


@dataclass
class NeuralMessageBus:
    """Thin wrapper around the global message bus utilities."""

    def __post_init__(self) -> None:
        # Ensure a clean bus for each coordinator instance
        reset_message_bus()

    def publish(self, event: Dict[str, Any]) -> None:
        publish_neural_event(event)

    def subscribe(self, region: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        subscribe_to_brain_region(region, handler)


@dataclass
class MultiAgentCoordinator:
    """Coordinate task distribution and result aggregation across agents.

    The coordinator exposes a small collaboration protocol consisting of
    task declaration, synchronisation and conflict resolution.  Agents
    communicate through the :class:`NeuralMessageBus` and share a common
    :class:`WorldModel` instance.
    """

    bus: NeuralMessageBus = field(default_factory=NeuralMessageBus)
    world_model: WorldModel = field(default_factory=WorldModel)

    def __post_init__(self) -> None:
        self.agents: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Any] = {}
        # Subscribe to our coordination channel for incoming results
        self.bus.subscribe("coordinator", self._handle_result)

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------
    def register_agent(self, agent_id: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        """Register ``handler`` to execute tasks for ``agent_id``."""

        if not callable(handler):
            raise TypeError("handler must be callable")
        self.agents[agent_id] = handler

        def wrapped(event: Dict[str, Any]) -> None:
            task_id = event.get("task_id")
            payload = event.get("payload")
            result = handler(payload)
            self.bus.publish(
                {
                    "target": "coordinator",
                    "task_id": task_id,
                    "result": result,
                    "agent_id": agent_id,
                }
            )

        self.bus.subscribe(agent_id, wrapped)

    # ------------------------------------------------------------------
    # Collaboration protocol
    # ------------------------------------------------------------------
    def declare_task(self, task_id: str, agent_id: str, payload: Dict[str, Any]) -> None:
        """Declare a new task for ``agent_id`` with ``payload``."""

        self.tasks[task_id] = {"agent_id": agent_id, "payload": payload}
        self.world_model.add_task(task_id, {"agent_id": agent_id, "payload": payload})

    def assign_task(self, task_id: str) -> None:
        """Dispatch ``task_id`` to its designated agent."""

        task = self.tasks.get(task_id)
        if not task:
            raise KeyError(f"unknown task {task_id}")
        self.bus.publish(
            {
                "target": task["agent_id"],
                "task_id": task_id,
                "payload": task["payload"],
            }
        )

    def _handle_result(self, event: Dict[str, Any]) -> None:
        task_id = event.get("task_id")
        result = event.get("result")
        if task_id is None:
            return
        existing = self.results.get(task_id)
        if existing is None:
            self.results[task_id] = result
            self.world_model.record_action(
                event.get("agent_id", ""), f"completed {task_id}"
            )
        else:
            self.resolve_conflict(task_id, existing, result)

    def resolve_conflict(self, task_id: str, existing: Any, incoming: Any) -> Any:
        """Resolve conflicting results for ``task_id``.

        The default strategy keeps the original result and ignores the new one.
        Subclasses may override to implement different policies.
        """

        return existing

    def synchronize(self) -> Dict[str, Any]:
        """Return aggregated results from completed tasks."""

        return dict(self.results)


__all__ = ["MultiAgentCoordinator", "NeuralMessageBus"]
