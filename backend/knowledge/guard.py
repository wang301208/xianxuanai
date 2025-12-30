"""Confidence guard for newly consolidated knowledge.

This module defines :class:`KnowledgeGuard`, a lightweight sentinel that monitors
incoming memory statements and validates them before they are trusted.
The guard can invoke pluggable validators (API checks, rule engines, etc.) and
then update the confidence and status metadata in both the long-term memory
store and the knowledge graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Protocol, Sequence

from backend.memory.long_term import LongTermMemory

try:  # Optional knowledge graph dependency for offline tests
    from third_party.autogpt.autogpt.core.knowledge_graph.graph_store import GraphStore, get_graph_store
except Exception:  # pragma: no cover
    GraphStore = Any  # type: ignore

    def get_graph_store() -> Any:  # type: ignore
        return None


class ValidationResult(Mapping[str, Any]):
    """Result returned by validators.

    Attributes
    ----------
    confidence: float
        Suggested confidence value in range [0.0, 1.0].
    status: str
        Status marker (e.g. ``"verified"``, ``"needs_review"``).
    metadata: Mapping[str, Any]
        Arbitrary metadata snapshot for auditing.
    reason: str
        Optional explanation for logs.
    """

    def __init__(
        self,
        confidence: float,
        status: str,
        metadata: Optional[Mapping[str, Any]] = None,
        reason: str | None = None,
    ) -> None:
        self._confidence = max(0.0, min(1.0, float(confidence)))
        self._status = status
        self._metadata = dict(metadata or {})
        self._reason = reason or ""

    def __getitem__(self, key: str) -> Any:
        if key == "confidence":
            return self._confidence
        if key == "status":
            return self._status
        if key == "metadata":
            return dict(self._metadata)
        if key == "reason":
            return self._reason
        raise KeyError(key)

    def __iter__(self):
        yield from ("confidence", "status", "metadata", "reason")

    def __len__(self) -> int:
        return 4

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def status(self) -> str:
        return self._status

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._metadata)

    @property
    def reason(self) -> str:
        return self._reason

    def with_defaults(self, *, default_status: str, base_metadata: Mapping[str, Any]) -> "ValidationResult":
        merged_meta = dict(base_metadata)
        merged_meta.update(self._metadata)
        status = self._status or default_status
        return ValidationResult(self._confidence, status, merged_meta, self._reason)


class Validator(Protocol):
    """Validator interface applied to knowledge entries."""

    def __call__(self, entry: "CandidateKnowledge") -> ValidationResult: ...


@dataclass
class CandidateKnowledge:
    """Context bundle passed to validators."""

    entry_id: Optional[int]
    content: str
    source: str
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


class KnowledgeGuard:
    """Evaluate new memories and maintain confidence metadata."""

    def __init__(
        self,
        *,
        memory: LongTermMemory,
        validators: Sequence[Validator],
        base_status: str = "pending",
        auto_promote_threshold: float = 0.75,
        demote_threshold: float = 0.3,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._memory = memory
        self._validators = list(validators)
        self._base_status = base_status
        self._auto_promote_threshold = auto_promote_threshold
        self._demote_threshold = demote_threshold
        self._logger = logger or logging.getLogger(__name__)
        self._graph: GraphStore | None = get_graph_store()
    @property
    def auto_promote_threshold(self) -> float:
        return self._auto_promote_threshold

    @property
    def demote_threshold(self) -> float:
        return self._demote_threshold

    def evaluate(
        self,
        content: str,
        *,
        source: str,
        entry_id: Optional[int] = None,
        metadata: Optional[MutableMapping[str, Any]] = None,
    ) -> ValidationResult:
        """Run validators and update long-term memory metadata."""

        candidate = CandidateKnowledge(
            entry_id=entry_id,
            content=content,
            source=source,
            metadata=metadata or {},
        )
        best_result: ValidationResult | None = None
        for validator in self._validators:
            try:
                result = validator(candidate)
            except Exception:  # pragma: no cover - defensive logging
                self._logger.exception("Knowledge validator %s failed", validator)
                continue
            if best_result is None or result.confidence > best_result.confidence:
                best_result = result
        if best_result is None:
            best_result = ValidationResult(0.5, self._base_status)
        final_result = best_result.with_defaults(
            default_status=self._base_status,
            base_metadata=candidate.metadata,
        )
        if entry_id is not None:
            self._memory.update_entry(
                entry_id,
                confidence=final_result.confidence,
                status=final_result.status,
                metadata=final_result.metadata,
            )
        self._update_graph_confidence(candidate, final_result)
        if final_result.confidence < self._demote_threshold:
            self._logger.info(
                "Knowledge guard flagged entry %s (confidence=%.2f): %s",
                entry_id,
                final_result.confidence,
                final_result.reason,
            )
        elif final_result.confidence >= self._auto_promote_threshold:
            self._logger.debug(
                "Knowledge guard validated entry %s (confidence=%.2f)",
                entry_id,
                final_result.confidence,
            )
        return final_result

    def _update_graph_confidence(self, entry: CandidateKnowledge, result: ValidationResult) -> None:
        if not self._graph:
            return
        try:
            metadata = dict(result.metadata)
            metadata.update({"confidence": result.confidence, "status": result.status})
            node_id = entry.metadata.get("concept_id")
            if node_id:
                self._graph.update_node(node_id, metadata=metadata)
        except Exception:  # pragma: no cover
            self._logger.debug("Failed to update graph confidence", exc_info=True)
