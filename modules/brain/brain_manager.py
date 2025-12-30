"""Unified brain manager for coordinating sub-managers.

This module exposes :class:`UnifiedBrainManager`, a thin orchestrator that
coordinates the ``AgentLifecycleManager`` and ``RuntimeModuleManager`` under a
single interface.  The class listens for events on the ``message_bus`` and
delegates work to the respective managers, publishing notifications for resource
allocation or fault isolation.

The implementation favours simplicity so that behaviour is easy to reason
about and test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, TYPE_CHECKING

from .message_bus import publish_neural_event, subscribe_to_brain_region

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from backend.execution.manager import AgentLifecycleManager
    from backend.capability.runtime_loader import RuntimeModuleManager


class LifecycleManager(Protocol):
    """Minimal protocol for lifecycle managers."""

    def pause_agent(self, name: str, reason: str | None = None) -> None:
        ...

    def resume_agent(self, name: str) -> None:
        ...


class ModuleManager(Protocol):
    """Minimal protocol for runtime module managers."""

    def load_module(self, name: str) -> None:
        ...

    def unload_module(self, name: str) -> None:
        ...


@dataclass
class UnifiedBrainManager:
    """Coordinate agent lifecycle and runtime modules via events.

    Parameters
    ----------
    lifecycle : :class:`AgentLifecycleManager`
        Manager responsible for agent creation and control.
    modules : :class:`RuntimeModuleManager`
        Manager responsible for loading runtime capability modules.
    """

    lifecycle: LifecycleManager
    modules: ModuleManager

    def __post_init__(self) -> None:
        # Listen for events addressed to the brain manager itself.
        subscribe_to_brain_region("brain_manager", self._handle_event)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def _handle_event(self, event: Dict[str, Any]) -> None:
        """Dispatch incoming events to the appropriate sub-manager."""

        command = event.get("command")
        if command == "pause_agent":
            self.lifecycle.pause_agent(event.get("agent"), event.get("reason"))
        elif command == "resume_agent":
            self.lifecycle.resume_agent(event.get("agent"))
        elif command == "load_module":
            self.modules.load_module(event.get("module"))
        elif command == "unload_module":
            self.modules.unload_module(event.get("module"))
        elif command == "fault":
            agent = event.get("agent")
            self.lifecycle.pause_agent(agent, "fault")
            publish_neural_event(
                {
                    "target": event.get("notify", "monitor"),
                    "agent": agent,
                    "status": "isolated",
                }
            )
        # Unknown commands are ignored to keep the manager robust.
