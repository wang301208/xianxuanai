from __future__ import annotations

import json
import math
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, MutableMapping, Sequence

from backend.memory import EpisodicMemory, LongTermMemory, SemanticMemory, WorkingMemory

if TYPE_CHECKING:
    from backend.knowledge.guard import KnowledgeGuard


@dataclass
class MemoryTrace:
    """Representation of information stored across the memory hierarchy."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=lambda: time.time())
    modality: str = "observation"
    text: str = ""
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.trace_id,
            "timestamp": self.timestamp,
            "modality": self.modality,
            "content": self.text,
            "importance": self.importance,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    def to_persistent_payload(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False)


class HierarchicalMemorySystem:
    """Coordinate short-term, episodic, semantic, and long-term memories."""

    def __init__(
        self,
        *,
        working_capacity: int = 7,
        episodic_limit: int = 256,
        consolidation_importance: float = 0.7,
        consolidation_window: float = 600.0,
        consolidation_batch_size: int = 5,
        long_term_path: str | None = ":memory:",
        long_term_max_entries: int | None = 5000,
        decay_half_life: float = 86400.0,
        interference_penalty: float = 0.1,
        semantic_limit: int = 2048,
        knowledge_guard: "KnowledgeGuard | None" = None,
    ) -> None:
        self.working_memory = WorkingMemory(capacity=max(1, working_capacity))
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self._semantic_order: OrderedDict[str, float] = OrderedDict()
        if long_term_path is None:
            self.long_term_memory = None
        else:
            if isinstance(long_term_path, Path):
                storage_path: str | Path = long_term_path
            elif str(long_term_path) in {":memory:", "file::memory:?cache=shared"}:
                storage_path = str(long_term_path)
            else:
                storage_path = Path(str(long_term_path))
            self.long_term_memory = LongTermMemory(storage_path, max_entries=long_term_max_entries)

        self.episodic_limit = max(1, episodic_limit)
        self.consolidation_importance = max(0.0, min(1.0, consolidation_importance))
        self.consolidation_window = max(0.0, consolidation_window)
        self.consolidation_batch_size = max(1, consolidation_batch_size)
        self.decay_half_life = max(0.0, decay_half_life)
        self.interference_penalty = max(0.0, interference_penalty)
        self.semantic_limit = max(1, semantic_limit)
        self._knowledge_guard = knowledge_guard

        self._hippocampal_buffer: list[MemoryTrace] = []
        self._episodic_traces: list[MemoryTrace] = []

    # ------------------------------------------------------------------
    # Encoding paths
    # ------------------------------------------------------------------
    def encode_experience(
        self,
        content: Any,
        *,
        modality: str = "observation",
        importance: float = 0.5,
        tags: Sequence[str] | None = None,
        metadata: MutableMapping[str, Any] | None = None,
    ) -> MemoryTrace:
        """Store an observation in working and episodic memory."""

        trace = MemoryTrace(
            timestamp=time.time(),
            modality=modality,
            text=self._to_text(content),
            importance=self._clip_importance(importance),
            tags=list(tags or []),
            metadata=dict(metadata or {}),
            raw=content,
        )

        self.working_memory.store(trace)
        self._episodic_traces.append(trace)
        self._hippocampal_buffer.append(trace)
        self.episodic_memory.store(
            trace.to_persistent_payload(),
            metadata={"timestamp": trace.timestamp},
        )
        self._apply_interference_penalty()
        self._truncate_episodic()

        if trace.importance >= self.consolidation_importance:
            self.consolidate(force=True)

        return trace

    def add_semantic_fact(
        self,
        key: str,
        value: Any,
        *,
        tags: Sequence[str] | None = None,
        importance: float = 0.6,
        metadata: MutableMapping[str, Any] | None = None,
    ) -> MemoryTrace:
        """Store a semantic fact and optionally consolidate it."""

        text_value = self._to_text(value)
        self.semantic_memory.add(key, text_value)
        self._semantic_order.pop(key, None)
        self._semantic_order[key] = time.time()
        self._truncate_semantic()

        trace = self.encode_experience(
            {"key": key, "value": text_value},
            modality="semantic",
            importance=importance,
            tags=tags,
            metadata=metadata,
        )
        trace.metadata["semantic_key"] = key
        if importance >= self.consolidation_importance:
            self._store_long_term(trace, category="semantic")
        return trace

    # ------------------------------------------------------------------
    # Consolidation & decay
    # ------------------------------------------------------------------
    def consolidate(self, *, force: bool = False) -> list[str]:
        """Move eligible hippocampal traces into long-term memory."""

        if not self._hippocampal_buffer or self.long_term_memory is None:
            return []

        now = time.time()
        consolidated: list[MemoryTrace] = []
        remaining: list[MemoryTrace] = []

        for trace in self._hippocampal_buffer:
            age = now - trace.timestamp
            eligible = (
                force
                or trace.importance >= self.consolidation_importance
                or (self.consolidation_window and age >= self.consolidation_window)
            )
            if eligible:
                consolidated.append(trace)
            else:
                remaining.append(trace)

        if not force:
            # Respect batch size and re-queue leftovers for later consolidation.
            to_store = consolidated[: self.consolidation_batch_size]
            remaining.extend(consolidated[self.consolidation_batch_size :])
        else:
            to_store = consolidated

        stored_ids: list[str] = []
        for trace in to_store:
            self._store_long_term(trace)
            stored_ids.append(trace.trace_id)

        self._hippocampal_buffer = remaining
        return stored_ids

    def apply_decay(self) -> None:
        """Apply time-based forgetting to episodic and long-term memories."""

        if self.decay_half_life <= 0:
            return

        now = time.time()
        cutoff = now - self.decay_half_life

        retained: list[MemoryTrace] = []
        for trace in self._episodic_traces:
            if trace.timestamp >= cutoff or trace.importance >= self.consolidation_importance:
                retained.append(trace)
        self._episodic_traces = retained

        # Working memory is naturally bounded by capacity; no extra decay needed.

        if self.long_term_memory is not None:
            self.long_term_memory.forget(before_ts=cutoff)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str | None = None,
        *,
        limit: int = 5,
        sources: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve memory traces relevant to ``query`` from configured sources."""

        sources_set = {src.lower() for src in (sources or ("working", "episodic", "semantic", "long_term"))}
        max_results = max(1, limit)
        query_lower = query.lower() if query else None
        tag_set = {t.lower() for t in tags} if tags else None

        results: list[dict[str, Any]] = []

        def matches(trace_tags: Iterable[str], text: str) -> bool:
            if query_lower and query_lower not in text.lower():
                return False
            if tag_set and not tag_set.issubset({t.lower() for t in trace_tags}):
                return False
            return True

        if "working" in sources_set:
            for trace in self.working_memory.retrieve({"reverse": True}):
                if isinstance(trace, MemoryTrace) and matches(trace.tags, trace.text):
                    results.append(self._format_trace(trace, source="working"))
                    if len(results) >= max_results:
                        return results

        if "episodic" in sources_set:
            for trace in reversed(self._episodic_traces):
                if matches(trace.tags, trace.text):
                    results.append(self._format_trace(trace, source="episodic"))
                    if len(results) >= max_results:
                        return results

        if "semantic" in sources_set:
            for key in reversed(list(self._semantic_order.keys())):
                value = self.semantic_memory.get(key)
                if value is None:
                    continue
                if matches([key], value):
                    results.append(
                        {
                            "source": "semantic",
                            "id": key,
                            "timestamp": self._semantic_order[key],
                            "content": value,
                            "tags": [key],
                            "metadata": {},
                        }
                    )
                    if len(results) >= max_results:
                        return results

        if "long_term" in sources_set and self.long_term_memory is not None:
            for payload in self.long_term_memory.retrieve({"newest_first": True}):
                try:
                    record = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                record_tags = record.get("tags", [])
                text = record.get("content", "")
                if matches(record_tags, text):
                    record["source"] = "long_term"
                    results.append(record)
                    if len(results) >= max_results:
                        break

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _clip_importance(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _to_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return repr(value)

    def _truncate_episodic(self) -> None:
        overflow = len(self._episodic_traces) - self.episodic_limit
        if overflow <= 0:
            return
        self._episodic_traces = self._episodic_traces[overflow:]

    def _truncate_semantic(self) -> None:
        while len(self._semantic_order) > self.semantic_limit:
            key, _ = self._semantic_order.popitem(last=False)
            # SemanticMemory exposes a dict-like interface internally.
            if hasattr(self.semantic_memory, "remove"):
                self.semantic_memory.remove(key)
            else:  # pragma: no cover - fallback for legacy implementations
                facts = self.semantic_memory.all()
                if key in facts:
                    del facts[key]

    def _apply_interference_penalty(self) -> None:
        if len(self._episodic_traces) <= self.episodic_limit:
            return
        overflow = len(self._episodic_traces) - self.episodic_limit
        for trace in self._episodic_traces[:overflow]:
            trace.importance = max(0.0, trace.importance - self.interference_penalty)

    def _store_long_term(self, trace: MemoryTrace, *, category: str | None = None) -> None:
        if self.long_term_memory is None:
            return
        payload = trace.to_persistent_payload()
        tags = trace.tags
        metadata: MutableMapping[str, Any] = {
            "category": category or trace.modality,
            "tags": list(tags) if tags else None,
            "timestamp": trace.timestamp,
        }
        entry_id = self.long_term_memory.store(payload, metadata=metadata)
        if self._knowledge_guard is not None and trace.importance >= self.consolidation_importance:
            metadata: MutableMapping[str, Any] = {
                "category": category or trace.modality,
                "tags": list(tags) if tags else [],
                "importance": trace.importance,
            }
            self._knowledge_guard.evaluate(
                trace.text or payload,
                source="hierarchical_memory",
                entry_id=entry_id,
                metadata=metadata,
            )

    def _format_trace(self, trace: MemoryTrace, *, source: str) -> dict[str, Any]:
        data = trace.as_dict()
        data["source"] = source
        data["raw"] = trace.raw
        return data

    def shutdown(self) -> None:
        """Close persistent resources."""

        if self.long_term_memory is not None:
            self.long_term_memory.close()
