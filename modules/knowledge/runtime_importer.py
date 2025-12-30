from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from backend.knowledge.importer import BulkKnowledgeImporter, RawConcept, RawRelation
from backend.knowledge.registry import (
    get_default_aligner,
    get_graph_store_instance,
    require_default_aligner,
    set_default_aligner,
)


_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass
class KnowledgeFact:
    """Structured fact collected during agent execution."""

    subject: str
    predicate: str
    obj: str
    subject_id: Optional[str] = None
    object_id: Optional[str] = None
    subject_description: str | None = None
    object_description: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    source: str | None = None
    context: str | None = None
    timestamp: float | None = None


class RuntimeKnowledgeImporter:
    """Convert runtime facts into knowledge graph updates via BulkKnowledgeImporter."""

    def __init__(
        self,
        *,
        importer: BulkKnowledgeImporter | None = None,
        auto_register_aligner: bool = True,
    ) -> None:
        aligner = get_default_aligner()
        if aligner is None:
            if auto_register_aligner:
                try:
                    from backend.concept_alignment import ConceptAligner
                    from capability.librarian import Librarian

                    aligner = ConceptAligner(librarian=Librarian(), entities={})
                    set_default_aligner(aligner)
                except Exception:
                    aligner = require_default_aligner()
            else:
                raise RuntimeError(
                    "ConceptAligner is not initialised. Register one via "
                    "backend.knowledge.registry.set_default_aligner() or "
                    "initialise KnowledgeConsolidator."
                )

        graph_store = get_graph_store_instance()
        if importer is None:
            importer = BulkKnowledgeImporter(aligner, graph_store=graph_store)
        self._importer = importer

    # ------------------------------------------------------------------
    def ingest_facts(self, facts: Iterable[KnowledgeFact]) -> Dict[str, Any]:
        concepts: Dict[str, RawConcept] = {}
        relations: List[RawRelation] = []

        for fact in facts:
            subject_id = fact.subject_id or _slugify(fact.subject)
            object_id = fact.object_id or _slugify(fact.obj)
            subject = concepts.get(subject_id)
            if subject is None:
                subject = RawConcept(
                    id=subject_id,
                    label=fact.subject.strip() or subject_id,
                    description=fact.subject_description or "",
                    metadata=_concept_metadata(fact, role="subject"),
                )
                concepts[subject_id] = subject

            obj_concept = concepts.get(object_id)
            if obj_concept is None:
                obj_concept = RawConcept(
                    id=object_id,
                    label=fact.obj.strip() or object_id,
                    description=fact.object_description or "",
                    metadata=_concept_metadata(fact, role="object"),
                )
                concepts[object_id] = obj_concept

            relations.append(
                RawRelation(
                    source=subject_id,
                    relation=fact.predicate.strip() or "related_to",
                    target=object_id,
                    weight=fact.confidence if fact.confidence is not None else 1.0,
                    metadata=_relation_metadata(fact),
                )
            )

        if not relations:
            return {"imported": 0, "concepts": 0}
        return self._importer.ingest_records(concepts.values(), relations)

    # ------------------------------------------------------------------
    def ingest_event(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        """Extract facts from an agent event payload."""

        facts_payload = event.get("knowledge_facts") if isinstance(event, Mapping) else None
        if not isinstance(facts_payload, Sequence):
            return {"imported": 0, "concepts": 0}

        facts: List[KnowledgeFact] = []
        for raw in facts_payload:
            if not isinstance(raw, Mapping):
                continue
            subject = str(raw.get("subject") or "").strip()
            predicate = str(raw.get("predicate") or "").strip()
            obj = str(raw.get("object") or raw.get("obj") or "").strip()
            if not subject or not predicate or not obj:
                continue
            facts.append(
                KnowledgeFact(
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    subject_id=_optional_str(raw.get("subject_id")),
                    object_id=_optional_str(raw.get("object_id")),
                    subject_description=_optional_str(raw.get("subject_description")),
                    object_description=_optional_str(raw.get("object_description")),
                    confidence=_optional_float(raw.get("confidence")),
                    source=_optional_str(raw.get("source")),
                    metadata=dict(raw.get("metadata") or {}),
                )
            )
        return self.ingest_facts(facts)

    # ------------------------------------------------------------------
    def ingest_triples(self, triples: Iterable[Tuple[str, str, str]]) -> Dict[str, Any]:
        return self.ingest_facts(
            KnowledgeFact(subject=s, predicate=p, obj=o) for s, p, o in triples
        )


def _slugify(text: str) -> str:
    slug = _SLUG_PATTERN.sub("_", text.lower()).strip("_")
    return slug or uuid.uuid4().hex[:12]


def _concept_metadata(fact: KnowledgeFact, *, role: str) -> Dict[str, Any]:
    metadata = {"role": role}
    if fact.source:
        metadata["source"] = fact.source
    if fact.context:
        metadata.setdefault("context", fact.context)
    if fact.timestamp is not None:
        metadata.setdefault("timestamp", fact.timestamp)
    metadata.update(fact.metadata)
    return metadata


def _relation_metadata(fact: KnowledgeFact) -> Dict[str, Any]:
    metadata = dict(fact.metadata)
    if fact.source:
        metadata.setdefault("source", fact.source)
    if fact.confidence is not None:
        metadata.setdefault("confidence", fact.confidence)
    if fact.context:
        metadata.setdefault("context", fact.context)
    if fact.timestamp is not None:
        metadata.setdefault("timestamp", fact.timestamp)
    return metadata


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
