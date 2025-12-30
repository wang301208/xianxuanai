"""Re-export ontology enums from the vendored AutoGPT implementation."""

from __future__ import annotations

try:  # pragma: no cover - import-time bridge
    from third_party.autogpt.autogpt.core.knowledge_graph.ontology import *  # type: ignore
except Exception:  # pragma: no cover - minimal fallback
    from enum import Enum

    class EntityType(str, Enum):
        AGENT = "agent"

    class RelationType(str, Enum):
        RELATED_TO = "related_to"

__all__ = ["EntityType", "RelationType"]

