from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class AgentPausedEvent:
    """Event published when an agent is paused."""

    agent: str
    reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.reason is None:
            data.pop("reason")
        return data


@dataclass
class AgentResumedEvent:
    """Event published when an agent is resumed."""

    agent: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentTerminatedEvent:
    """Event published when an agent is terminated."""

    agent: str
    reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.reason is None:
            data.pop("reason")
        return data


__all__ = [
    "AgentPausedEvent",
    "AgentResumedEvent",
    "AgentTerminatedEvent",
]
