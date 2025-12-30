"""High level coordination for unified long-term memory."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from modules.memory.vector_store import VectorMemoryStore, VectorRecord
except Exception:  # pragma: no cover - lightweight fallback when memory stack unavailable
    @dataclass
    class VectorRecord:  # type: ignore[override]
        id: str
        text: str
        metadata: Dict[str, Any]
        score: float = 0.0

    class VectorMemoryStore:  # type: ignore[override]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self._records: List[VectorRecord] = []

        def add_text(
            self, text: str, metadata: Optional[Dict[str, Any]] = None, record_id: str | None = None
        ) -> str:
            record_id = record_id or str(uuid.uuid4())
            record = VectorRecord(record_id, text, dict(metadata or {}), score=1.0)
            self._records.append(record)
            return record_id

        def query(self, text: str, top_k: int = 5, **_kwargs: Any) -> List[VectorRecord]:
            scored: List[VectorRecord] = []
            for record in self._records:
                score = 1.0 if text.lower() in record.text.lower() else 0.0
                scored.append(VectorRecord(record.id, record.text, dict(record.metadata), score))
            return sorted(scored, key=lambda r: r.score, reverse=True)[:top_k]

try:  # optional dependency when knowledge stack not fully installed
    from .runtime_importer import KnowledgeFact, RuntimeKnowledgeImporter
except Exception:  # pragma: no cover - provide minimal fallbacks for testing
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class KnowledgeFact:  # type: ignore[override]
        subject: str
        predicate: str
        obj: str
        subject_id: Optional[str] = None
        object_id: Optional[str] = None
        subject_description: Optional[str] = None
        object_description: Optional[str] = None
        metadata: Dict[str, Any] = None  # type: ignore[assignment]
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

try:  # optional dependency during tests or minimal installs
    from .repository import KnowledgeRepository
except Exception:  # pragma: no cover - repository stack may be unavailable
    KnowledgeRepository = None  # type: ignore


@dataclass
class ConsolidatedSummary:
    """Structured summary returned by consolidation routines."""

    text: str
    metadata: Dict[str, Any]
    vector_id: Optional[str]


@dataclass
class HippocampalTrace:
    """Short-term interaction payload awaiting consolidation."""

    id: str
    text: str
    timestamp: float
    facts: List[KnowledgeFact]
    metadata: Dict[str, Any]


class LongTermMemoryCoordinator:
    """Bridge vector memory, knowledge graphs, and episodic summaries.

    The coordinator offers a single entry point for recording new knowledge facts,
    consolidating episodic experience into persistent summaries and performing
    similarity lookups before engaging more expensive reasoning modules.  It uses
    the existing ``RuntimeKnowledgeImporter`` for graph persistence and a
    ``VectorMemoryStore`` for fuzzy recall.
    """

    def __init__(
        self,
        *,
        storage_root: Path | None = None,
        vector_store: VectorMemoryStore | None = None,
        knowledge_importer: RuntimeKnowledgeImporter | None = None,
        knowledge_repository: KnowledgeRepository | None = None,
        hippocampal_max_traces: int = 128,
    ) -> None:
        root = storage_root or Path("data/long_term_memory")
        root.mkdir(parents=True, exist_ok=True)

        self._vector_store = vector_store or VectorMemoryStore(root / "vector_store")
        self._importer = knowledge_importer or RuntimeKnowledgeImporter()
        self._repository = knowledge_repository
        self._hippocampal_traces: List[HippocampalTrace] = []
        self._hippocampal_max = max(1, hippocampal_max_traces)

    # ------------------------------------------------------------------ public API
    @property
    def vector_store(self) -> VectorMemoryStore:
        return self._vector_store

    def record_fact(
        self,
        fact: KnowledgeFact,
        *,
        embed: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConsolidatedSummary:
        """Persist a single fact and optionally embed it for fuzzy recall."""

        self._importer.ingest_facts([fact])
        merged_meta = self._fact_metadata(fact)
        if metadata:
            merged_meta.update(metadata)

        vector_id: Optional[str] = None
        if embed:
            text = self._fact_to_text(fact)
            vector_id = self._vector_store.add_text(text, merged_meta)
        return ConsolidatedSummary(text=self._fact_to_text(fact), metadata=merged_meta, vector_id=vector_id)

    def record_facts(
        self,
        facts: Iterable[KnowledgeFact],
        *,
        embed: bool = True,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ConsolidatedSummary]:
        """Persist and embed a collection of facts."""

        fact_list = list(facts)
        if not fact_list:
            return []
        self._importer.ingest_facts(fact_list)
        summaries: List[ConsolidatedSummary] = []
        for fact in fact_list:
            metadata = self._fact_metadata(fact)
            if base_metadata:
                metadata.update(base_metadata)
            vector_id: Optional[str] = None
            if embed:
                vector_id = self._vector_store.add_text(self._fact_to_text(fact), metadata)
            summaries.append(
                ConsolidatedSummary(text=self._fact_to_text(fact), metadata=metadata, vector_id=vector_id)
            )
        return summaries

    def stage_interaction(
        self,
        text: str,
        *,
        facts: Optional[Iterable[KnowledgeFact]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Capture short-term details in a hippocampal-style buffer.

        The staged traces can later be consolidated into semantic (cortical)
        memory via :meth:`consolidate_hippocampal_traces` without relying on
        immediate graph persistence. This mirrors how the biological
        hippocampus rapidly encodes experiences before replaying them to
        cortex for gradual integration.
        """

        trace_id = str(uuid.uuid4())
        trace = HippocampalTrace(
            id=trace_id,
            text=text.strip(),
            timestamp=time.time(),
            facts=list(facts or []),
            metadata=dict(metadata or {}),
        )
        self._hippocampal_traces.append(trace)
        if len(self._hippocampal_traces) > self._hippocampal_max:
            self._hippocampal_traces.pop(0)
        return trace_id

    def consolidate_hippocampal_traces(
        self,
        *,
        limit: Optional[int] = None,
        embed: bool = True,
    ) -> Optional[ConsolidatedSummary]:
        """Replay staged traces into long-term cortical storage.

        Each hippocampal trace is optionally embedded for future fuzzy recall
        and its associated facts are promoted into the symbolic repository.
        The method also synthesises a lightweight consolidated summary that
        reflects the merged experience payload.
        """

        if not self._hippocampal_traces:
            return None

        batch_size = limit or len(self._hippocampal_traces)
        to_process = self._hippocampal_traces[:batch_size]
        consolidated_lines: List[str] = []
        trace_ids: List[str] = []

        for trace in to_process:
            trace_ids.append(trace.id)
            consolidated_lines.append(
                f"[{int(trace.timestamp)}] {trace.text}" if trace.text else f"[{int(trace.timestamp)}]"
            )
            if embed:
                metadata = {
                    "stage": "hippocampal",
                    "trace_id": trace.id,
                    "timestamp": trace.timestamp,
                    **trace.metadata,
                }
                self._vector_store.add_text(trace.text or "", metadata)
            if trace.facts:
                self.record_facts(
                    trace.facts,
                    embed=embed,
                    base_metadata={
                        "stage": "hippocampal",
                        "trace_id": trace.id,
                        "timestamp": trace.timestamp,
                    },
                )

        consolidated_text = " \n".join(consolidated_lines)
        summary_metadata = {
            "stage": "cortical",
            "trace_count": len(to_process),
            "hippocampal_traces": trace_ids,
        }
        vector_id = None
        if embed:
            vector_id = self._vector_store.add_text(consolidated_text, summary_metadata)

        del self._hippocampal_traces[:batch_size]

        return ConsolidatedSummary(text=consolidated_text, metadata=summary_metadata, vector_id=vector_id)

    def consolidate_episode(
        self,
        *,
        task_id: str,
        policy_version: str,
        total_reward: float,
        steps: int,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConsolidatedSummary:
        """Create a summary entry for an episode and store it in vector memory."""

        summary_text = self._episode_summary_text(
            task_id=task_id,
            policy_version=policy_version,
            total_reward=total_reward,
            steps=steps,
            success=success,
            metadata=metadata or {},
        )
        summary_meta = {
            "task_id": task_id,
            "policy_version": policy_version,
            "total_reward": total_reward,
            "steps": steps,
            "success": success,
        }
        if metadata:
            summary_meta.update(metadata)
        vector_id = self._vector_store.add_text(summary_text, summary_meta)
        return ConsolidatedSummary(text=summary_text, metadata=summary_meta, vector_id=vector_id)

    def query_similar(self, text: str, *, top_k: int = 5) -> List[VectorRecord]:
        """Retrieve similar memories based on semantic similarity."""

        return self._vector_store.query(text, top_k=top_k)

    def known_fact(
        self,
        *,
        subject: str,
        predicate: str,
        obj: Optional[str] = None,
    ) -> bool:
        """Check whether a fact already exists in the knowledge repository."""

        if self._repository is None:
            return False
        try:
            result = self._repository.query(node_id=subject)
        except Exception:  # pragma: no cover - defensive fallback
            return False
        edges = result.get("edges") or result.get("relations") or []
        for edge in edges:
            relation = edge.get("relation_type") or edge.get("relation")
            if relation != predicate:
                continue
            if obj is None:
                return True
            target = edge.get("target") or edge.get("object")
            if target == obj:
                return True
        return False

    # ------------------------------------------------------------------ helpers
    def _fact_to_text(self, fact: KnowledgeFact) -> str:
        return f"{fact.subject} {fact.predicate} {fact.obj}".strip()

    def _fact_metadata(self, fact: KnowledgeFact) -> Dict[str, Any]:
        meta = {
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object": fact.obj,
        }
        if fact.source:
            meta["source"] = fact.source
        if fact.confidence is not None:
            meta["confidence"] = fact.confidence
        if fact.context:
            meta["context"] = fact.context
        if fact.timestamp is not None:
            meta["timestamp"] = fact.timestamp
        meta.update(fact.metadata)
        return meta

    def _episode_summary_text(
        self,
        *,
        task_id: str,
        policy_version: str,
        total_reward: float,
        steps: int,
        success: bool,
        metadata: Dict[str, Any],
    ) -> str:
        base = (
            f"Episode {task_id} with policy {policy_version} completed in {steps} steps "
            f"with reward {total_reward:.2f}. Outcome: {'success' if success else 'failure'}."
        )
        if metadata:
            extras = ", ".join(f"{key}={value}" for key, value in sorted(metadata.items()))
            return f"{base} Details: {extras}."
        return base

