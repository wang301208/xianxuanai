"""Memory storage utilities and shared protocols."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Protocol, runtime_checkable

from .long_term import LongTermMemory
from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory

try:  # Optional dependency (numpy-heavy)
    from .differentiable_neural_computer import DifferentiableNeuralComputer
except Exception:  # pragma: no cover - fallback when numpy is missing
    DifferentiableNeuralComputer = None  # type: ignore


@runtime_checkable
class MemoryProtocol(Protocol):
    """Common interface implemented by memory backends."""

    def store(self, item, *, metadata: Optional[Mapping[str, object]] = None):
        """Persist ``item`` together with optional ``metadata``."""

        ...

    def retrieve(
        self, filters: Optional[Mapping[str, object]] = None
    ) -> Iterable[object]:
        """Return items matching ``filters`` (backend specific semantics)."""

        ...

    def clear(self) -> None:
        """Remove all stored items."""

        ...

    def search(
        self, query: str, *, limit: Optional[int] = None
    ) -> Iterable[object]:  # pragma: no cover - optional implementation
        """Optional text search over stored items."""

        ...


__all__ = [
    "MemoryProtocol",
    "LongTermMemory",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "DifferentiableNeuralComputer",
]
