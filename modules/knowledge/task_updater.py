"""Orchestrate knowledge updates when tasks complete."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.knowledge.consolidation import KnowledgeConsolidator
from backend.knowledge.router import MemoryRouter

from .runtime_importer import KnowledgeFact, RuntimeKnowledgeImporter

LOGGER = logging.getLogger(__name__)


class KnowledgeUpdatePipeline:
    """Pipeline that promotes task outcomes into the knowledge graph."""

    def __init__(
        self,
        *,
        consolidator: KnowledgeConsolidator,
        importer: RuntimeKnowledgeImporter,
        memory_router: MemoryRouter | None = None,
        hot_limit: int = 128,
    ) -> None:
        self._consolidator = consolidator
        self._importer = importer
        self._memory_router = memory_router
        self._hot_limit = max(1, hot_limit)

    # ------------------------------------------------------------------
    def process_task_event(self, event: Dict[str, Any]) -> None:
        """Ingest knowledge emitted by a completed task."""

        statements = self._extract_statements(event)
        for text, metadata in statements:
            try:
                self._consolidator.record_statement(
                    text,
                    source=metadata.pop("source", "task"),
                    metadata=metadata,
                )
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.debug("Failed to queue statement for consolidation.", exc_info=True)

        if statements:
            try:
                self._consolidator.wait_idle()
            except Exception:
                LOGGER.debug("Knowledge consolidator wait failed.", exc_info=True)

        facts = self._extract_facts(event)
        if facts:
            try:
                self._importer.ingest_facts(facts)
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.debug("Runtime knowledge import failed.", exc_info=True)

        if self._memory_router is not None:
            try:
                promoted = self._memory_router.review()
                if promoted:
                    LOGGER.debug("Promoted %d memory entries to long-term knowledge.", len(promoted))
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.debug("Memory router review failed.", exc_info=True)

        self._enforce_hot_limit()

    # ------------------------------------------------------------------
    def _enforce_hot_limit(self) -> None:
        try:
            hot = list(self._consolidator.hot_concepts)
        except Exception:
            return
        if len(hot) <= self._hot_limit:
            return
        for concept_id in hot[self._hot_limit :]:
            try:
                self._consolidator.demote_concept(concept_id)
            except Exception:
                LOGGER.debug("Failed to demote concept %s during cooling.", concept_id, exc_info=True)

    # ------------------------------------------------------------------
    def _extract_statements(self, event: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        task_id = str(event.get("task_id") or "task")
        agent_id = event.get("agent_id")
        base_metadata: Dict[str, Any] = {"task_id": task_id}
        if agent_id:
            base_metadata["agent_id"] = agent_id
        if isinstance(event.get("metadata"), dict):
            base_metadata.update(event["metadata"])

        statements: List[Tuple[str, Dict[str, Any]]] = []

        for key, flag in (("summary", "summary"), ("detail", "detail")):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                metadata = dict(base_metadata)
                metadata["source"] = f"task:{task_id}"
                metadata[f"is_{flag}"] = True
                statements.append((value.strip(), metadata))

        supplementary = event.get("knowledge_statements")
        for entry in _ensure_iterable(supplementary):
            if isinstance(entry, str) and entry.strip():
                metadata = dict(base_metadata)
                metadata["source"] = f"knowledge:{task_id}"
                statements.append((entry.strip(), metadata))

        notes = event.get("notes")
        for entry in _ensure_iterable(notes):
            if isinstance(entry, str) and entry.strip():
                metadata = dict(base_metadata)
                metadata["source"] = f"note:{task_id}"
                statements.append((entry.strip(), metadata))

        supplementary_info = event.get("supplementary_info")
        for entry in _ensure_iterable(supplementary_info):
            text = _stringify(entry)
            if text:
                metadata = dict(base_metadata)
                metadata["source"] = f"supplement:{task_id}"
                statements.append((text, metadata))

        return statements

    def _extract_facts(self, event: Dict[str, Any]) -> List[KnowledgeFact]:
        raw_facts = event.get("knowledge_facts")
        facts: List[KnowledgeFact] = []
        for raw in _ensure_iterable(raw_facts):
            fact = _normalize_fact(raw)
            if fact is not None:
                facts.append(fact)
        return facts


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ensure_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return value
    return [value]


def _normalize_fact(raw: Any) -> KnowledgeFact | None:
    if isinstance(raw, KnowledgeFact):
        return raw
    if not isinstance(raw, dict):
        return None
    subject = str(raw.get("subject") or "").strip()
    predicate = str(raw.get("predicate") or "").strip()
    obj = str(raw.get("object") or raw.get("obj") or "").strip()
    if not subject or not predicate or not obj:
        return None
    return KnowledgeFact(
        subject=subject,
        predicate=predicate,
        obj=obj,
        subject_id=_optional_str(raw.get("subject_id")),
        object_id=_optional_str(raw.get("object_id")),
        subject_description=_optional_str(raw.get("subject_description")),
        object_description=_optional_str(raw.get("object_description")),
        metadata=dict(raw.get("metadata") or {}),
        confidence=_optional_float(raw.get("confidence")),
        source=_optional_str(raw.get("source")),
        context=_optional_str(raw.get("context")),
        timestamp=_optional_float(raw.get("timestamp")),
    )


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)
