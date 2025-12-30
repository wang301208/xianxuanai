"""Shared perception module abstractions."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from BrainSimulationSystem.environment.base import PerceptionPacket


class PerceptionModule(Protocol):
    """Protocol implemented by concrete sensory front-ends."""

    def process(self, packet: PerceptionPacket, *, info: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        ...
