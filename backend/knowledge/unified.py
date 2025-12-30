"""Unified knowledge base module.

This module defines :class:`UnifiedKnowledgeBase`, a utility that aggregates
symbolic knowledge from graph backends with vectorised semantic memories.  The
implementation keeps optional dependencies lazy so that unit tests can exercise
lightweight fallbacks while production deployments may attach persistent graph
stores and embedding models.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    class _FakeNDArray(list):
        """Lightweight stand-in for ``numpy.ndarray`` when numpy is unavailable."""

        @property
        def ndim(self) -> int:
            return 1

        def reshape(self, *shape: Any) -> "_FakeNDArray":
            return self

        @property
        def size(self) -> int:
            return len(self)

        @property
        def shape(self) -> tuple[int]:
            return (len(self),)

        def tolist(self) -> List[float]:
            return list(self)

        def __itruediv__(self, other: float) -> "_FakeNDArray":
            divisor = float(other) if other not in (None, 0) else 1.0
            for idx, value in enumerate(self):
                self[idx] = float(value) / divisor
            return self

    def _fake_asarray(data: Any, dtype: Any = float) -> _FakeNDArray:
        try:
            return _FakeNDArray([float(item) for item in data])
        except TypeError:
            return _FakeNDArray([float(data)])
        except ValueError:
            return _FakeNDArray()

    def _fake_zeros(length: int, dtype: Any = float) -> _FakeNDArray:
        return _FakeNDArray([0.0 for _ in range(int(length))])

    np = SimpleNamespace(  # type: ignore
        ndarray=_FakeNDArray,
        asarray=_fake_asarray,
        zeros=_fake_zeros,
        linalg=SimpleNamespace(norm=lambda vec: math.sqrt(sum(float(v) ** 2 for v in vec))),
    )

try:  # Optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    SentenceTransformer = None  # type: ignore

from modules.common import CausalRelation

from ..memory.long_term import LongTermMemory
from .registry import get_graph_store_instance
from .vector_store import LocalVectorStore

if TYPE_CHECKING:  # pragma: no cover - for static analysis only
    from modules.knowledge import KnowledgeFact, RuntimeKnowledgeImporter
else:  # Avoid import-time dependency cycles at runtime
    KnowledgeFact = Any  # type: ignore
    RuntimeKnowledgeImporter = Any  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass
class KnowledgeSource:
    """Simple container for a knowledge source."""

    name: str
    data: Dict[str, str]
    embeddings: Optional[Dict[str, np.ndarray]] = None
    causal_relations: List[CausalRelation] = field(default_factory=list)


@dataclass
class UnifiedKnowledgeBase:
    """Container that stores multiple knowledge sources."""

    sources: Dict[str, KnowledgeSource] = field(default_factory=dict)
    embedder: Any | None = None
    memory: Optional[LongTermMemory] = None
    vector_store: Optional[LocalVectorStore] = None
    knowledge_importer: RuntimeKnowledgeImporter | None = None
    graph_store: Any | None = None

    _fact_cache: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _fallback_dimension: int = field(default=256, init=False)

    def __post_init__(self) -> None:
        if self.graph_store is None:
            try:
                self.graph_store = get_graph_store_instance()
            except Exception:  # pragma: no cover - optional backend
                self.graph_store = None

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------
    def _embed_key(self, source_name: str, concept: str) -> str:
        return f"{source_name}:{concept}"

    def add_source(self, source: KnowledgeSource) -> None:
        """Register a new knowledge source and compute embeddings."""

        self.sources[source.name] = source
        if source.embeddings is None and self.embedder is not None:
            source.embeddings = {}

        if self.embedder is None:
            return

        for concept, description in source.data.items():
            key = self._embed_key(source.name, concept)
            embedding = self._load_or_encode_embedding(key, description)
            if embedding is None:
                continue
            source.embeddings[concept] = embedding
            metadata = {
                "source": source.name,
                "concept": concept,
                "description": description,
                "text": description,
                "key": key,
            }
            self._register_vector(embedding, metadata)

    # ------------------------------------------------------------------
    # Fact ingestion
    # ------------------------------------------------------------------
    def ingest_facts(
        self,
        facts: Iterable[KnowledgeFact],
        *,
        embed: bool = True,
    ) -> Dict[str, Any]:
        """Persist a batch of knowledge facts into the knowledge base."""

        fact_list = [fact for fact in facts if fact is not None]
        if not fact_list:
            return {"imported": 0}

        import_result: Dict[str, Any] = {}
        if self.knowledge_importer is not None:
            try:
                import_result = self.knowledge_importer.ingest_facts(fact_list)
            except Exception:  # pragma: no cover - importer optional
                LOGGER.debug("Runtime knowledge importer failed.", exc_info=True)

        for fact in fact_list:
            self._record_fact(fact, embed=embed)

        if import_result:
            return import_result
        return {"imported": len(fact_list)}

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------
    def query(self, concept: str, *, semantic: bool = False, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve concept descriptions, graph relations and stored facts."""

        concept = (concept or "").strip()
        if not concept:
            return {}

        results: Dict[str, Any] = {}
        causal: List[CausalRelation] = []

        for name, source in self.sources.items():
            if concept in source.data:
                results[name] = source.data[concept]
                causal.extend([r for r in source.causal_relations if r.cause == concept])

        graph_snippets = self._graph_context(concept, top_k=top_k)
        if graph_snippets:
            results["graph"] = graph_snippets

        if not semantic:
            if causal:
                results["causal_relations"] = causal
            return results

        fact_hits = self._search_facts(concept, top_k=top_k)
        if fact_hits:
            results["facts"] = fact_hits

        if causal:
            results["causal_relations"] = causal
        return results

    def concepts(self) -> Iterable[str]:
        """Iterate over all known concept names."""

        for source in self.sources.values():
            for concept in source.data.keys():
                yield concept

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _load_or_encode_embedding(self, key: str, description: str) -> np.ndarray | None:
        stored = self.memory.get_embedding(key) if self.memory else None
        if stored is not None:
            try:
                vector = np.asarray(stored[0], dtype=float)
            except Exception:
                vector = None
        else:
            vector = self._encode_text(description)
            if vector is not None and self.memory is not None:
                try:
                    source_name, _, concept_name = key.partition(":")
                    self.memory.add_embedding(
                        key,
                        vector.tolist(),
                        {"source": source_name or "source", "concept": concept_name or key},
                    )
                except Exception:  # pragma: no cover - persistence optional
                    LOGGER.debug("Failed to persist embedding for %s", key, exc_info=True)
        return vector

    def _encode_text(self, text: str) -> np.ndarray | None:
        text = (text or "").strip()
        if not text:
            return None
        if self.embedder is not None:
            try:
                embedding = self.embedder.encode(text)
            except Exception:
                LOGGER.debug("Embedder encode failed; using fallback hash embedding.", exc_info=True)
            else:
                arr = np.asarray(embedding, dtype=float)
                if arr.ndim > 1:
                    arr = arr.reshape(-1)
                return arr
        return self._hash_embedding(text)

    def _hash_embedding(self, text: str) -> np.ndarray:
        tokens = [token for token in text.lower().split() if token]
        vector = np.zeros(self._fallback_dimension, dtype=float)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self._fallback_dimension
            vector[index] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def _register_vector(self, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        if vector is None or vector.size == 0:
            return
        if self.vector_store is None:
            self.vector_store = LocalVectorStore(vector.shape[0])
        elif self.vector_store.dimension != vector.shape[0]:
            LOGGER.debug(
                "Vector dimension mismatch: existing=%s new=%s",
                self.vector_store.dimension,
                vector.shape[0],
            )
            return

        self.vector_store.add(vector, dict(metadata))
        text = str(metadata.get("text") or "").strip()
        cache_entry = dict(metadata)
        cache_entry.setdefault("text", text)
        self._fact_cache.append(cache_entry)

        if self.memory is not None:
            key = str(metadata.get("key") or metadata.get("subject") or hash(text))
            try:
                self.memory.add_embedding(key, vector.tolist(), metadata)
            except Exception:  # pragma: no cover - persistence optional
                LOGGER.debug("Failed to mirror embedding into long term memory.", exc_info=True)

    def _record_fact(self, fact: KnowledgeFact, *, embed: bool) -> None:
        text = self._fact_to_text(fact)
        metadata = self._fact_metadata(fact)
        metadata.setdefault("text", text)

        if self.memory is not None:
            try:
                self.memory.add(
                    "knowledge_fact",
                    text,
                    metadata={key: value for key, value in metadata.items() if key != "text"},
                )
            except Exception:  # pragma: no cover - persistence optional
                LOGGER.debug("Failed to store fact in long term memory.", exc_info=True)

        self._fact_cache.append(dict(metadata))
        if embed:
            vector = self._encode_text(text)
            if vector is not None:
                self._register_vector(vector, metadata)

    # ------------------------------------------------------------------
    # Fact and graph lookup helpers
    # ------------------------------------------------------------------
    def _fact_to_text(self, fact: KnowledgeFact) -> str:
        subject = str(getattr(fact, "subject", "")).strip()
        predicate = str(getattr(fact, "predicate", "")).strip()
        obj = str(getattr(fact, "obj", getattr(fact, "object", ""))).strip()
        context = str(getattr(fact, "context", "") or "").strip()
        parts = [part for part in (subject, predicate, obj, context) if part]
        return " ".join(parts)

    def _fact_metadata(self, fact: KnowledgeFact) -> Dict[str, Any]:
        metadata = dict(getattr(fact, "metadata", {}) or {})
        subject = str(getattr(fact, "subject", "")).strip()
        predicate = str(getattr(fact, "predicate", "")).strip()
        obj = str(getattr(fact, "obj", getattr(fact, "object", ""))).strip()
        if subject:
            metadata.setdefault("subject", subject)
        if predicate:
            metadata.setdefault("predicate", predicate)
        if obj:
            metadata.setdefault("object", obj)
        source = getattr(fact, "source", None)
        if source:
            metadata.setdefault("source", source)
        confidence = getattr(fact, "confidence", None)
        if confidence is not None:
            try:
                metadata.setdefault("confidence", float(confidence))
            except (TypeError, ValueError):
                pass
        context = getattr(fact, "context", None)
        if context:
            metadata.setdefault("context", context)
        timestamp = getattr(fact, "timestamp", None)
        if timestamp is not None:
            try:
                metadata.setdefault("timestamp", float(timestamp))
            except (TypeError, ValueError):
                pass
        metadata.setdefault("key", f"fact:{subject}:{predicate}:{obj}")
        return metadata

    def _search_facts(self, query: str, *, top_k: int) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        vector = self._encode_text(query)
        if vector is not None and self.vector_store is not None:
            try:
                hits = [dict(hit) for hit in self.vector_store.search(vector, top_k=top_k)]
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.debug("Vector similarity search failed.", exc_info=True)
        if not hits and vector is not None and self.memory is not None:
            try:
                similar = self.memory.similarity_search(
                    vector.tolist(), top_k=top_k, category="knowledge_fact"
                )
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.debug(
                    "Long-term memory similarity search failed.", exc_info=True
                )
                similar = []
            hits = [
                {
                    "id": result.get("memory_id") or result["key"],
                    "text": (result.get("memory") or {}).get("content")
                    or (result.get("metadata") or {}).get("text", ""),
                    "metadata": {
                        **(result.get("metadata") or {}),
                        "similarity": result.get("score", 0.0),
                    },
                }
                for result in similar
            ]
        if hits:
            return hits
        return self._lexical_search(query, top_k=top_k)

    def _lexical_search(self, query: str, *, top_k: int) -> List[Dict[str, Any]]:
        tokens = {token for token in query.lower().split() if token}
        if not tokens or not self._fact_cache:
            return []

        scored: List[Tuple[int, Dict[str, Any]]] = []
        for entry in self._fact_cache:
            text = str(entry.get("text") or "").lower()
            if not text:
                continue
            entry_tokens = set(text.split())
            score = len(tokens & entry_tokens)
            if score:
                scored.append((score, entry))
        if not scored:
            return []
        scored.sort(key=lambda item: item[0], reverse=True)
        return [dict(item[1]) for item in scored[:top_k]]

    def _graph_context(self, query: str, *, top_k: int) -> List[str]:
        if self.graph_store is None:
            return []
        try:
            snapshot = self.graph_store.get_snapshot()
        except Exception:  # pragma: no cover - backend may be unavailable
            LOGGER.debug("Graph snapshot retrieval failed.", exc_info=True)
            return []

        nodes = snapshot.get("nodes", [])
        edges = snapshot.get("edges", [])
        node_map = {self._node_id(node): node for node in nodes}
        query_tokens = {token for token in query.lower().split() if token}
        scored: List[Tuple[int, Any]] = []
        for node in nodes:
            label = self._node_label(node)
            text_parts = [label.lower()]
            properties = getattr(node, "properties", {}) or {}
            for value in properties.values():
                if isinstance(value, str):
                    text_parts.append(value.lower())
            text = " ".join(text_parts)
            score = sum(1 for token in query_tokens if token in text) or 0
            if score:
                scored.append((score, node))
        if not scored:
            return []
        scored.sort(key=lambda item: item[0], reverse=True)

        snippets: List[str] = []
        for score, node in scored[:top_k]:
            label = self._node_label(node)
            summary = self._summarise_properties(getattr(node, "properties", {}) or {})
            node_type = getattr(getattr(node, "type", None), "name", None) or str(getattr(node, "type", ""))
            line = label
            if node_type:
                line = f"{label} [{node_type}]"
            if summary:
                line = f"{line}: {summary}"
            snippets.append(line)
            snippets.extend(
                self._format_edges_for_node(
                    self._node_id(node),
                    edges,
                    node_map,
                    relation_limit=3,
                )
            )
        return snippets

    def _node_id(self, node: Any) -> str:
        return str(getattr(node, "id", getattr(getattr(node, "properties", {}), "get", lambda *_: "")("id", "")))

    def _node_label(self, node: Any) -> str:
        properties = getattr(node, "properties", {}) or {}
        label = properties.get("label") or properties.get("name")
        if label:
            return str(label)
        node_id = getattr(node, "id", None)
        if node_id:
            return str(node_id)
        return "unknown"

    def _summarise_properties(self, properties: Dict[str, Any]) -> str:
        if not properties:
            return ""
        ignored = {"label", "name", "id"}
        summary_parts: List[str] = []
        for key, value in properties.items():
            if key in ignored or value in (None, ""):
                continue
            if isinstance(value, (int, float)):
                summary_parts.append(f"{key}={value:.2f}")
            elif isinstance(value, str):
                summary_parts.append(f"{key}={value}")
            elif isinstance(value, list):
                summary_parts.append(f"{key}={len(value)} items")
            if len(summary_parts) >= 3:
                break
        return "; ".join(summary_parts)

    def _format_edges_for_node(
        self,
        node_id: str,
        edges: Iterable[Any],
        node_map: Dict[str, Any],
        *,
        relation_limit: int,
    ) -> List[str]:
        formatted: List[str] = []
        for edge in edges:
            source = getattr(edge, "source", "")
            target = getattr(edge, "target", "")
            if source != node_id and target != node_id:
                continue
            other_id = target if source == node_id else source
            other_node = node_map.get(other_id)
            other_label = self._node_label(other_node) if other_node else other_id
            relation = getattr(edge, "type", None)
            relation_label = (
                getattr(relation, "value", None)
                or getattr(relation, "name", None)
                or str(relation or "related")
            )
            direction = "->" if source == node_id else "<-"
            line = f"  - {node_id} {direction} {relation_label} {other_label}"
            properties = getattr(edge, "properties", {}) or {}
            weight = properties.get("weight")
            if isinstance(weight, (int, float)):
                line += f" (weight={float(weight):.2f})"
            formatted.append(line)
            if len(formatted) >= relation_limit:
                break
        return formatted
