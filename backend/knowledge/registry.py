"""Global registry for knowledge-related components.

The registry offers a lightweight mechanism for sharing the active
``ConceptAligner`` and knowledge graph store between modules.  This is useful
for abilities or background workers that need to access consolidated knowledge
without direct references being threaded through every call site.
"""

from __future__ import annotations

from typing import Optional

try:
    from backend.concept_alignment import ConceptAligner
except Exception:  # pragma: no cover - optional dependency
    ConceptAligner = None  # type: ignore

try:
    from backend.autogpt.autogpt.core.knowledge_graph.graph_store import (
        GraphStore,
        get_graph_store,
    )
except Exception:  # pragma: no cover - optional dependency
    GraphStore = None  # type: ignore

    def get_graph_store():  # type: ignore[no-redef]
        raise RuntimeError("Graph store backend is unavailable")

_DEFAULT_ALIGNER: ConceptAligner | None = None
_DEFAULT_GRAPH: GraphStore | None = None


def set_default_aligner(aligner: ConceptAligner) -> None:
    """Register ``aligner`` as the process-wide default."""

    global _DEFAULT_ALIGNER
    _DEFAULT_ALIGNER = aligner


def get_default_aligner() -> Optional[ConceptAligner]:
    """Return the registered default aligner, if any."""

    return _DEFAULT_ALIGNER


def require_default_aligner() -> ConceptAligner:
    """Return the default aligner or raise ``RuntimeError`` if absent."""

    aligner = get_default_aligner()
    if aligner is None:
        raise RuntimeError(
            "Knowledge aligner not configured. Initialise KnowledgeConsolidator or "
            "register a ConceptAligner via knowledge.registry.set_default_aligner()."
        )
    return aligner


def set_graph_store(store: GraphStore) -> None:
    """Register ``store`` as the default graph store."""

    global _DEFAULT_GRAPH
    _DEFAULT_GRAPH = store


def get_graph_store_instance() -> GraphStore:
    """Return the registered graph store or fall back to :func:`get_graph_store`."""

    global _DEFAULT_GRAPH
    if _DEFAULT_GRAPH is None:
        _DEFAULT_GRAPH = get_graph_store()
    return _DEFAULT_GRAPH
