from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional, List


class TaskStatus(str, Enum):
    """Enumeration of possible task states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskDispatchEvent:
    """Event published by the coordinator to assign a task to an agent.

    ``metadata`` contains supplemental details that allow subscribers to
    reconstruct how and why a task was routed (e.g., selection heuristics or
    scheduling hints). ``routed`` is only included when the event represents a
    deliberate reroute so consumers can differentiate between freshly dispatched
    work and retargeted tasks. Both fields are omitted from the serialized
    payload when they are not provided so legacy consumers do not receive
    unexpected keys.
    """

    task_id: str
    payload: Dict[str, Any]
    assigned_to: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    routed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("metadata") is None:
            data.pop("metadata", None)
        if not data.get("routed"):
            data.pop("routed", None)
        return data


@dataclass
class TaskResultEvent:
    """Event emitted by a worker after completing an RPC task."""

    task_id: str
    status: TaskStatus
    result: Dict[str, Any]
    worker_id: Optional[str] = None
    duration: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        if data.get("metadata") is None:
            data.pop("metadata", None)
        return data


@dataclass
class TaskStatusEvent:
    """Event emitted by an agent to report the status of a task."""

    agent_id: str
    task_id: str
    status: TaskStatus
    detail: Optional[str] = None
    summary: Optional[str] = None
    knowledge_statements: Optional[List[str]] = None
    knowledge_facts: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        # Remove optional fields that were not provided to keep payload lean.
        for key in ("detail", "summary", "knowledge_statements", "knowledge_facts", "metadata"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


@dataclass
class IterationEvent:
    """Event representing a reflexive iteration step."""

    iteration: int
    candidates: List[str]
    selected: str
    scores: Dict[str, float] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceSignalEvent:
    """Event emitted to advertise resource availability or pressure."""

    worker_id: str
    resource_signal: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("metadata") is None:
            data.pop("metadata", None)
        return data


def render_iteration_timeline(events: List[IterationEvent]) -> str:
    """Return a simple textual visualization of iteration events."""

    lines = []
    for ev in events:
        score_part = ""
        if ev.scores:
            formatted = ", ".join(
                f"{name}:{score:.2f}" for name, score in ev.scores.items()
            )
            score_part = f" [{formatted}]"
        lines.append(f"{ev.iteration}: {ev.selected}{score_part}")
    return "\n".join(lines)
