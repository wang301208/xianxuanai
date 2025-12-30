"""Protocols describing the public interface for brain backends."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from modules.brain.state import BrainCycleResult


@runtime_checkable
class BrainBackendProtocol(Protocol):
    """Minimal surface exposed by brain backends used in AutoGPT agents.

    Implementations must provide :meth:`process_cycle`, which accepts a mapping
    of sensory/context inputs and returns a :class:`BrainCycleResult`. Backends
    may optionally expose lifecycle helpers such as :meth:`shutdown`,
    :meth:`attach_knowledge_base`, or :meth:`update_config`; agents should only
    call those methods after checking for their presence via :func:`hasattr`.
    """

    def process_cycle(self, input_payload: Mapping[str, Any]) -> BrainCycleResult:
        """Advance the simulation by one cognitive cycle."""
        ...


class SupportsBrainShutdown(Protocol):
    """Optional protocol for backends that provide a shutdown hook."""

    def shutdown(self) -> None:
        """Release resources and stop any background processing."""
        ...


class SupportsKnowledgeBaseAttachment(Protocol):
    """Optional protocol for backends that can accept knowledge bases."""

    def attach_knowledge_base(self, knowledge_base: Any) -> None:
        """Attach a knowledge base object for enriched cognition."""
        ...


class SupportsRuntimeConfigUpdate(Protocol):
    """Optional protocol for backends that accept runtime config updates."""

    def update_config(self, runtime_config: Any | None = None, **kwargs: Any) -> None:
        """Apply runtime configuration changes."""
        ...
