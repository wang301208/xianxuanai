"""Background knowledge consolidation utilities.

This module implements :class:`KnowledgeConsolidator`, a lightweight manager that
listens for newly acquired statements and distils them into the internal
concept graph.  Incoming observations are queued and processed on a background
thread so the main execution flow is never blocked.

For every statement the consolidator:

* encodes the text into an embedding vector,
* aligns it against existing :class:`ConceptNode` objects via
  :class:`backend.concept_alignment.ConceptAligner`,
* updates the best matching concept when the similarity is high enough,
  otherwise creates a new concept node,
* persists the update in both the in-memory dictionary and the global
  knowledge graph.

The implementation keeps a simple in-memory vector index which is sufficient
for unit tests and small deployments.  It can be replaced with a more
capable index by supplying a custom librarian when needed.
"""

from __future__ import annotations

import hashlib
from collections import deque
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from backend.concept_alignment import ConceptAligner
from backend.knowledge.vector_store import LocalVectorStore
from backend.knowledge.registry import set_default_aligner, set_graph_store
from backend.knowledge.guard import KnowledgeGuard
from modules.common.concepts import ConceptNode

try:  # pragma: no cover - optional dependency during lightweight tests
    from third_party.autogpt.autogpt.core.knowledge_graph.graph_store import get_graph_store
    from third_party.autogpt.autogpt.core.knowledge_graph.ontology import EntityType, RelationType
except Exception:  # pragma: no cover - fallback namespace used in tests
    from backend.autogpt.autogpt.core.knowledge_graph.graph_store import (  # type: ignore
        get_graph_store,
    )
    from backend.autogpt.autogpt.core.knowledge_graph.ontology import (  # type: ignore
        EntityType,
        RelationType,
    )


Encoder = Callable[[str], List[float]]


def _default_text_encoder(text: str, *, dimensions: int = 8) -> List[float]:
    """Return a deterministic pseudo-embedding for ``text``.

    The encoder hashes the lowercase UTF-8 representation of ``text`` and
    normalises the digest into the requested number of floating point values.
    This keeps the module free from external heavy dependencies while still
    enabling similarity checks for tests and lightweight environments.
    """

    digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
    step = len(digest) // dimensions
    if step == 0:
        step = 1
    embedding: List[float] = []
    for idx in range(0, len(digest), step):
        chunk = digest[idx : idx + step]
        if not chunk:
            break
        value = int.from_bytes(chunk, byteorder="big", signed=False)
        embedding.append(value / float(2 ** (8 * len(chunk))))
        if len(embedding) == dimensions:
            break
    while len(embedding) < dimensions:
        embedding.append(0.0)
    return embedding


class _ConceptVectorIndex:
    """Maintain separate hot/cold :class:`LocalVectorStore` instances."""

    def __init__(self) -> None:
        self._vectors: Dict[str, List[float]] = {}
        self._zones: Dict[str, str] = {}
        self._hot_store: Optional[LocalVectorStore] = None
        self._cold_store: Optional[LocalVectorStore] = None

    def add_or_update(
        self,
        concept_id: str,
        embedding: List[float],
        *,
        zone: Optional[str] = None,
    ) -> None:
        previous_zone = self._zones.get(concept_id)
        new_zone = zone or previous_zone or "hot"
        self._vectors[concept_id] = embedding
        self._zones[concept_id] = new_zone
        if previous_zone and previous_zone != new_zone:
            self._rebuild_zone(previous_zone)
        self._rebuild_zone(new_zone)

    def set_zone(self, concept_id: str, zone: str) -> None:
        if concept_id not in self._vectors:
            return
        current_zone = self._zones.get(concept_id, "hot")
        if current_zone == zone:
            return
        self._zones[concept_id] = zone
        self._rebuild_zone(current_zone)
        self._rebuild_zone(zone)

    def zone_of(self, concept_id: str) -> str:
        return self._zones.get(concept_id, "hot")

    def search(self, embedding: List[float], top_k: int) -> List[str]:
        if top_k <= 0:
            return []
        results: List[str] = []
        results.extend(self._search_store(self._hot_store, embedding, top_k))
        if len(results) >= top_k:
            return results[:top_k]
        remaining = top_k - len(results)
        for candidate in self._search_store(self._cold_store, embedding, remaining):
            if candidate not in results:
                results.append(candidate)
            if len(results) == top_k:
                break
        return results

    def hot_ids(self) -> List[str]:
        return [concept_id for concept_id, zone in self._zones.items() if zone == "hot"]

    def clear(self) -> None:
        self._vectors.clear()
        self._zones.clear()
        self._hot_store = None
        self._cold_store = None

    def _rebuild_zone(self, zone: str) -> None:
        vectors = {
            concept_id: vector
            for concept_id, vector in self._vectors.items()
            if self._zones.get(concept_id, "hot") == zone
        }
        if not vectors:
            if zone == "hot":
                self._hot_store = None
            else:
                self._cold_store = None
            return
        dimension = len(next(iter(vectors.values())))
        store = LocalVectorStore(dimension, use_faiss=False)
        for concept_id, vector in vectors.items():
            store.add(vector, {"id": concept_id})
        if zone == "hot":
            self._hot_store = store
        else:
            self._cold_store = store

    @staticmethod
    def _search_store(
        store: Optional[LocalVectorStore],
        embedding: List[float],
        top_k: int,
    ) -> List[str]:
        if not store or top_k <= 0:
            return []
        hits = store.search(embedding, top_k=top_k)
        return [meta.get("id") for meta in hits if meta.get("id")]


class _InMemoryLibrarian:
    """Minimal librarian compatible with :class:`ConceptAligner`."""

    def __init__(self, index: _ConceptVectorIndex) -> None:
        self._index = index

    def search(
        self,
        embedding: List[float],
        n_results: int = 5,
        vector_type: str = "text",
        return_content: bool = False,
    ) -> List[str]:
        _ = vector_type  # This lightweight librarian only supports a single space.
        ids = self._index.search(embedding, top_k=n_results)
        if return_content:
            return ids  # ConceptAligner expects ids regardless of this flag.
        return ids


@dataclass
class _PendingStatement:
    """Container for statements queued for consolidation."""

    text: str
    source: str
    metadata: Dict[str, Any]
    vector_type: str
    timestamp: float


class KnowledgeConsolidator:
    """Manage asynchronous consolidation of newly acquired statements."""

    def __init__(
        self,
        *,
        aligner: ConceptAligner | None = None,
        encoder: Encoder | None = None,
        vector_type: str = "text",
        similarity_threshold: float = 0.82,
        queue_size: int = 256,
        hot_limit: int = 128,
        guard: KnowledgeGuard | None = None,
    ) -> None:
        self._encoder = encoder or _default_text_encoder
        self._vector_type = vector_type
        self._similarity_threshold = similarity_threshold
        self._queue: "queue.Queue[_PendingStatement]" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._index = _ConceptVectorIndex()
        self._graph = get_graph_store()
        self._hot_limit = max(1, int(hot_limit))
        self._hot_order: deque[str] = deque()
        self._guard = guard

        if aligner is None:
            entities: Dict[str, ConceptNode] = {}
            librarian = _InMemoryLibrarian(self._index)
            aligner = ConceptAligner(librarian, entities)
            self._manage_index = True
        else:
            # When a pre-configured aligner is supplied we do not know if its
            # librarian supports live updates.  To keep behaviour predictable
            # we still manage our own index and expect callers to synchronise
            # external stores separately.
            entities = aligner.entities
            self._manage_index = False

        self._aligner = aligner
        self._entities = aligner.entities

        self._worker = threading.Thread(target=self._run, name="knowledge-consolidator")
        self._worker.daemon = True
        self._worker.start()

        # Share aligner and graph store globally for consumers such as abilities.
        set_default_aligner(self._aligner)
        set_graph_store(self._graph)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record_statement(
        self,
        text: str,
        *,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector_type: Optional[str] = None,
    ) -> None:
        """Queue ``text`` for consolidation."""

        if not text.strip():
            return
        try:
            self._queue.put_nowait(
                _PendingStatement(
                    text=text.strip(),
                    source=source,
                    metadata=metadata or {},
                    vector_type=vector_type or self._vector_type,
                    timestamp=time.time(),
                )
            )
        except queue.Full:  # pragma: no cover - defensive backpressure
            # Drop the oldest entry to make room for fresh knowledge.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(
                _PendingStatement(
                    text=text.strip(),
                    source=source,
                    metadata=metadata or {},
                    vector_type=vector_type or self._vector_type,
                    timestamp=time.time(),
                )
            )

    def stop(self, *, timeout: float | None = None) -> None:
        """Request the background worker to stop and wait for it."""

        self._stop.set()
        self._worker.join(timeout=timeout)

    def promote_concept(self, concept_id: str) -> None:
        """Explicitly mark ``concept_id`` as part of the hot tier."""

        if not self._manage_index:
            return
        with self._lock:
            self._register_hot(concept_id)

    def demote_concept(self, concept_id: str) -> None:
        """Move ``concept_id`` into the cold tier."""

        if not self._manage_index:
            return
        with self._lock:
            try:
                self._hot_order.remove(concept_id)
            except ValueError:
                pass
            self._index.set_zone(concept_id, "cold")

    @property
    def hot_concepts(self) -> List[str]:
        """Return hot concepts in most-recent order (best effort)."""

        if not self._manage_index:
            return []
        return list(self._hot_order)

    def wait_idle(self, *, timeout: float = 5.0) -> None:
        """Block until the pending queue has been processed."""

        deadline = time.time() + timeout
        while getattr(self._queue, "unfinished_tasks", 0) and time.time() < deadline:
            time.sleep(0.01)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._process_statement(item)
            finally:
                self._queue.task_done()

    def _process_statement(self, item: _PendingStatement) -> None:
        embedding = self._encoder(item.text)
        if not embedding:
            return

        matches = self._aligner.align(
            embedding,
            n_results=1,
            vector_type=item.vector_type,
        )
        top_match = matches[0] if matches else None
        similarity = 0.0
        if top_match is not None:
            similarity = float(top_match.metadata.get("similarity", 0.0))
        if top_match is None or similarity < self._similarity_threshold:
            self._create_concept(item, embedding, top_match, similarity)
        else:
            self._update_concept(top_match, item, embedding, similarity)

    # ------------------------------------------------------------------
    # Concept bookkeeping helpers
    # ------------------------------------------------------------------
    def _create_concept(
        self,
        item: _PendingStatement,
        embedding: List[float],
        best_match: ConceptNode | None,
        best_similarity: float,
    ) -> None:
        concept_id = uuid.uuid4().hex[:12]
        label = item.text.splitlines()[0]
        if len(label) > 64:
            label = label[:61] + "..."
        node = ConceptNode(
            id=concept_id,
            label=label,
            modalities={item.vector_type: embedding},
            metadata={
                "occurrences": 1,
                "sources": [item.source],
                "last_seen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(item.timestamp)),
                "examples": [item.text],
            },
        )
        with self._lock:
            self._entities[concept_id] = node
            if self._manage_index:
                self._index.add_or_update(concept_id, embedding, zone="hot")
                self._register_hot(concept_id)
            # Clear ConceptAligner cache so future align calls consider the new node.
            if hasattr(self._aligner, "_cached_search"):
                self._aligner._cached_search.cache_clear()  # type: ignore[attr-defined]

        self._graph.add_node(
            concept_id,
            EntityType.CONCEPT,
            label=node.label,
            occurrences=1,
            sources=[item.source],
            last_seen=item.timestamp,
        )
        if best_match is not None and best_similarity > 0.0:
            self._graph.add_edge(
                best_match.id,
                concept_id,
                RelationType.RELATED_TO,
                similarity=best_similarity,
            )
            node.metadata.setdefault("related_to", []).append(
                {"id": best_match.id, "similarity": best_similarity}
            )

        if self._guard is not None:
            result = self._guard.evaluate(
                item.text,
                source=item.source,
                metadata={
                    "concept_id": concept_id,
                    "vector_type": item.vector_type,
                    "similarity": best_similarity,
                    "action": "create",
                },
            )
            if result.confidence < self._guard.demote_threshold:
                self.demote_concept(concept_id)
            elif result.confidence >= self._guard.auto_promote_threshold:
                self.promote_concept(concept_id)

    def _update_concept(
        self,
        node: ConceptNode,
        item: _PendingStatement,
        embedding: List[float],
        similarity: float,
    ) -> None:
        with self._lock:
            occurrences = int(node.metadata.get("occurrences", 0)) + 1
            node.metadata["occurrences"] = occurrences
            sources: List[str] = list(node.metadata.get("sources", []))
            if item.source not in sources:
                sources.append(item.source)
            node.metadata["sources"] = sources
            node.metadata["last_seen"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(item.timestamp)
            )
            examples: List[str] = list(node.metadata.get("examples", []))
            if item.text not in examples:
                examples.append(item.text)
                if len(examples) > 5:
                    examples = examples[-5:]
            node.metadata["examples"] = examples
            # Simple running average to keep embedding representative.
            current = node.modalities.get(item.vector_type)
            if current:
                alpha = 1.0 / occurrences
                updated_vector = [
                    (1.0 - alpha) * old + alpha * new for old, new in zip(current, embedding)
                ]
            else:
                updated_vector = embedding
            node.modalities[item.vector_type] = updated_vector
            if self._manage_index:
                self._index.add_or_update(node.id, updated_vector, zone="hot")
                self._register_hot(node.id)
            if hasattr(self._aligner, "_cached_search"):
                self._aligner._cached_search.cache_clear()  # type: ignore[attr-defined]

        self._graph.add_node(
            node.id,
            EntityType.CONCEPT,
            label=node.label,
            occurrences=node.metadata["occurrences"],
            sources=node.metadata["sources"],
            last_seen=item.timestamp,
        )

        if self._guard is not None:
            result = self._guard.evaluate(
                item.text,
                source=item.source,
                metadata={
                    "concept_id": node.id,
                    "vector_type": item.vector_type,
                    "similarity": similarity,
                    "occurrences": node.metadata["occurrences"],
                    "action": "update",
                },
            )
            if result.confidence < self._guard.demote_threshold:
                self.demote_concept(node.id)
            elif result.confidence >= self._guard.auto_promote_threshold:
                self.promote_concept(node.id)

    # ------------------------------------------------------------------
    # Introspection helpers for tests
    # ------------------------------------------------------------------
    @property
    def concepts(self) -> Iterable[ConceptNode]:
        """Expose consolidated concepts for inspection."""

        return list(self._entities.values())

    # ------------------------------------------------------------------
    # Hot/cold tier helpers
    # ------------------------------------------------------------------
    def _register_hot(self, concept_id: str) -> None:
        if not self._manage_index:
            return
        try:
            self._hot_order.remove(concept_id)
        except ValueError:
            pass
        self._hot_order.append(concept_id)
        self._index.set_zone(concept_id, "hot")
        while len(self._hot_order) > self._hot_limit:
            demote_id = self._hot_order.popleft()
            if demote_id == concept_id:
                continue
            self._index.set_zone(demote_id, "cold")
