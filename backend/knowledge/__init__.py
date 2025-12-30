"""Knowledge integration utilities.

This package provides components for integrating multiple knowledge sources
into a unified representation that can be queried by other parts of the
system.
"""

from .unified import UnifiedKnowledgeBase
from .vector_store import LocalVectorStore

try:
    from .consolidation import KnowledgeConsolidator
except Exception:  # pragma: no cover - optional dependency
    KnowledgeConsolidator = None  # type: ignore

try:
    from .guard import KnowledgeGuard, ValidationResult
except Exception:  # pragma: no cover - optional dependency
    KnowledgeGuard = None  # type: ignore
    ValidationResult = None  # type: ignore

try:
    from .importer import BulkKnowledgeImporter
except Exception:  # pragma: no cover - optional dependency
    BulkKnowledgeImporter = None  # type: ignore

from .registry import (
    get_default_aligner,
    get_graph_store_instance,
    require_default_aligner,
    set_default_aligner,
    set_graph_store,
)

try:
    from .router import MemoryRouter
    from .ingest import DocumentIngestor, DocumentChunk, ImageIngestor
except Exception:  # pragma: no cover - optional dependency
    MemoryRouter = None  # type: ignore
    DocumentIngestor = None  # type: ignore
    DocumentChunk = None  # type: ignore
    ImageIngestor = None  # type: ignore

__all__ = [
    "UnifiedKnowledgeBase",
    "LocalVectorStore",
    "KnowledgeConsolidator",
    "BulkKnowledgeImporter",
    "DocumentIngestor",
    "DocumentChunk",
    "ImageIngestor",
    "set_default_aligner",
    "get_default_aligner",
    "require_default_aligner",
    "set_graph_store",
    "get_graph_store_instance",
    "MemoryRouter",
    "KnowledgeGuard",
    "ValidationResult",
]
