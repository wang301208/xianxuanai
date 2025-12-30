"""Runtime knowledge integration helpers with graceful fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

try:  # optional dependency chain pulls in heavy numeric stack
    from .runtime_importer import KnowledgeFact, RuntimeKnowledgeImporter
except Exception:  # pragma: no cover - fallback when backend unavailable
    @dataclass
    class KnowledgeFact:  # type: ignore[override]
        subject: str
        predicate: str
        obj: str
        subject_id: Optional[str] = None
        object_id: Optional[str] = None
        subject_description: Optional[str] = None
        object_description: Optional[str] = None
        metadata: Dict[str, Any] | None = None
        confidence: Optional[float] = None
        source: Optional[str] = None
        context: Optional[str] = None
        timestamp: Optional[float] = None

        def __post_init__(self) -> None:  # pragma: no cover - fallback normalisation
            if self.metadata is None:
                self.metadata = {}

    class RuntimeKnowledgeImporter:  # type: ignore[override]
        def ingest_facts(self, _facts: Iterable[KnowledgeFact]) -> Dict[str, Any]:
            raise RuntimeError("Knowledge importer unavailable")

from .long_term_memory import LongTermMemoryCoordinator, ConsolidatedSummary

try:
    from .knowledge_consolidation import ExternalKnowledgeConsolidator, KnowledgeConsolidationConfig
except Exception:  # pragma: no cover
    ExternalKnowledgeConsolidator = None  # type: ignore[assignment]
    KnowledgeConsolidationConfig = None  # type: ignore[assignment]

try:  # optional helpers may rely on broader runtime context
    from .prompt_fusion import collect_knowledge_context, collect_knowledge_context_async
except Exception:  # pragma: no cover - expose no-op fallbacks
    async def collect_knowledge_context_async(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {}

    def collect_knowledge_context(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {}

try:
    from .action_guard import ActionGuard, ActionGuardResult
except Exception:  # pragma: no cover - basic fallbacks
    ActionGuard = None  # type: ignore[assignment]
    ActionGuardResult = None  # type: ignore[assignment]

try:
    from .task_updater import KnowledgeUpdatePipeline
except Exception:  # pragma: no cover
    KnowledgeUpdatePipeline = None  # type: ignore[assignment]

try:
    from .acquisition import KnowledgeAcquisitionManager
except Exception:  # pragma: no cover
    KnowledgeAcquisitionManager = None  # type: ignore[assignment]

try:
    from .self_teacher import SelfTeacher
except Exception:  # pragma: no cover
    SelfTeacher = None  # type: ignore[assignment]

try:
    from .problem_analyzer import ProblemAnalyzer, ProblemBreakdown
except Exception:  # pragma: no cover
    ProblemAnalyzer = None  # type: ignore[assignment]
    ProblemBreakdown = None  # type: ignore[assignment]

try:
    from .research_tool import ResearchTool, WebSearchHit, DocHit
except Exception:  # pragma: no cover
    ResearchTool = None  # type: ignore[assignment]
    WebSearchHit = None  # type: ignore[assignment]
    DocHit = None  # type: ignore[assignment]

try:
    from .knowledge_base import KnowledgeBase, KnowledgeBaseItem
except Exception:  # pragma: no cover
    KnowledgeBase = None  # type: ignore[assignment]
    KnowledgeBaseItem = None  # type: ignore[assignment]

try:
    from .graph_actors import (
        ingest_batch as ingest_graph_batch,
        query_pattern as query_graph_pattern,
        create_ray_graph_ingest_actor,
        create_ray_graph_query_actor,
    )
except Exception:  # pragma: no cover - provide benign fallbacks
    def ingest_graph_batch(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {}

    def query_graph_pattern(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {}

    def create_ray_graph_ingest_actor(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Ray graph ingest actor unavailable")

    def create_ray_graph_query_actor(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Ray graph query actor unavailable")

__all__ = [
    "KnowledgeFact",
    "RuntimeKnowledgeImporter",
    "LongTermMemoryCoordinator",
    "ConsolidatedSummary",
    "ExternalKnowledgeConsolidator",
    "KnowledgeConsolidationConfig",
    "collect_knowledge_context",
    "collect_knowledge_context_async",
    "ActionGuard",
    "ActionGuardResult",
    "KnowledgeUpdatePipeline",
    "ingest_graph_batch",
    "query_graph_pattern",
    "create_ray_graph_ingest_actor",
    "create_ray_graph_query_actor",
    "KnowledgeAcquisitionManager",
    "SelfTeacher",
    "ProblemAnalyzer",
    "ProblemBreakdown",
    "ResearchTool",
    "WebSearchHit",
    "DocHit",
    "KnowledgeBase",
    "KnowledgeBaseItem",
]
