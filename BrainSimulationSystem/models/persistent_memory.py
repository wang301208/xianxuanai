"""Persistent memory utilities with pluggable embedding backends and indexing."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback when faiss unavailable
    faiss = None

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback when transformers unavailable
    SentenceTransformer = None  # type: ignore


@dataclass
class PersistentMemoryConfig:
    """Configuration for the persistent memory manager."""

    path: str = "BrainSimulationSystem/data/persistent_memory.json"
    embedding_backend: str = "hash"
    embedding_model: Optional[str] = None
    embedding_cache_path: Optional[str] = "BrainSimulationSystem/data/embedding_cache"
    embedding_dim: int = 128
    index_backend: str = "auto"
    index_path: Optional[str] = "BrainSimulationSystem/data/persistent_memory.index"
    ingestion_batch_size: int = 64
    max_entries: int = 10_000
    working_memory_size: int = 32
    decay: float = 0.98  # forgetting factor for similarity weighting
class DiskEmbeddingCache:
    """Stores embeddings on disk keyed by deterministic hashes."""

    def __init__(self, cache_dir: Optional[Path]) -> None:
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash(text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return digest

    def get(self, text: str) -> Optional[np.ndarray]:
        if self.cache_dir is None:
            return None
        key = self._hash(text)
        path = self.cache_dir / f"{key}.npy"
        if not path.exists():
            return None
        try:
            return np.load(path, allow_pickle=False)
        except Exception:
            return None

    def set(self, text: str, vector: np.ndarray) -> None:
        if self.cache_dir is None:
            return
        key = self._hash(text)
        path = self.cache_dir / f"{key}.npy"
        np.save(path, vector)


class EmbeddingBackend:
    """Interface for embedding backends."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def dimension(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class HashEmbeddingBackend(EmbeddingBackend):
    """Deterministic hashing based embedding backend."""

    def __init__(self, dim: int, cache: Optional[DiskEmbeddingCache] = None) -> None:
        self._dim = dim
        self._cache = cache

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_embedding(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        if not text:
            return vector
        tokens = text.split()
        for token in tokens:
            token_hash = hash(token)
            idx = abs(token_hash) % self._dim
            sign = 1.0 if token_hash >= 0 else -1.0
            vector[idx] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for text in texts:
            cached = self._cache.get(text) if self._cache else None
            if cached is not None:
                vectors.append(cached.astype(np.float32))
                continue
            vec = self._hash_embedding(text)
            if self._cache:
                self._cache.set(text, vec)
            vectors.append(vec)
        return np.vstack(vectors) if vectors else np.zeros((0, self._dim), dtype=np.float32)


class SentenceTransformerBackend(EmbeddingBackend):
    """Embedding backend backed by a sentence-transformer model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache: Optional[DiskEmbeddingCache] = None,
    ) -> None:
        if SentenceTransformer is None:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence_transformers is not installed")
        self.model_name = model_name or "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)
        self._dim = int(self.model.get_sentence_embedding_dimension())
        self._cache = cache

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing: List[str] = []
        cached_vectors: Dict[int, np.ndarray] = {}
        if self._cache:
            for idx, text in enumerate(texts):
                cached = self._cache.get(text)
                if cached is not None:
                    cached_vectors[idx] = cached.astype(np.float32)
                else:
                    missing.append(text)
        else:
            missing = list(texts)

        vectors = [None] * len(texts)
        for idx, vec in cached_vectors.items():
            vectors[idx] = vec

        if missing:
            new_vecs = self.model.encode(missing, convert_to_numpy=True)
            missing_iter = iter(new_vecs)
            for idx, text in enumerate(texts):
                if vectors[idx] is None:
                    vec = np.array(next(missing_iter), dtype=np.float32)
                    vectors[idx] = vec
                    if self._cache:
                        self._cache.set(text, vec)

        filled = [vec for vec in vectors if vec is not None]
        return np.vstack(filled) if filled else np.zeros((0, self._dim), dtype=np.float32)


class VectorIndex:
    """Simple vector index with optional FAISS acceleration."""

    def __init__(self, dim: int, backend: str = "auto", path: Optional[Path] = None) -> None:
        self.dim = dim
        self.backend = backend
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss_index = None
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._use_faiss = False
        self._init_index()

    def _init_index(self) -> None:
        backend = self.backend
        if backend == "auto":
            backend = "faiss" if faiss is not None else "brute"
        if backend == "faiss" and faiss is not None:
            self._use_faiss = True
            if self.path and self.path.exists():  # pragma: no cover - optional loading
                try:
                    self._faiss_index = faiss.read_index(str(self.path))
                except Exception:
                    self._faiss_index = faiss.IndexFlatIP(self.dim)
            else:
                self._faiss_index = faiss.IndexFlatIP(self.dim)
        else:
            self._use_faiss = False

    def rebuild(self, vectors: np.ndarray) -> None:
        if vectors.size == 0:
            self._vectors = np.zeros((0, self.dim), dtype=np.float32)
            if self._use_faiss and self._faiss_index is not None:
                self._faiss_index.reset()
            return
        self._vectors = np.asarray(vectors, dtype=np.float32)
        if self._use_faiss and self._faiss_index is not None:
            self._faiss_index.reset()
            self._faiss_index.add(self._vectors)
            self._save_index()

    def add(self, vectors: np.ndarray) -> None:
        if vectors.size == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if not self._vectors.size:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        if self._use_faiss and self._faiss_index is not None:
            self._faiss_index.add(vectors)
            self._save_index()

    def _save_index(self) -> None:
        if self._use_faiss and self._faiss_index is not None and self.path is not None:
            faiss.write_index(self._faiss_index, str(self.path))  # pragma: no cover - optional

    def search(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[float, int]]:
        if not self._vectors.size:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if self._use_faiss and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query, top_k)
            return [(float(scores[0][i]), int(indices[0][i])) for i in range(len(indices[0])) if indices[0][i] != -1]
        similarities = np.dot(self._vectors, query.T).reshape(-1)
        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query)
        with np.errstate(divide="ignore", invalid="ignore"):
            similarities = np.divide(similarities, norms, out=np.zeros_like(similarities), where=norms != 0)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(float(similarities[idx]), int(idx)) for idx in top_indices]


@dataclass
class MemoryEntry:
    """Represents a single long-term memory item."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    uid: str = field(default_factory=lambda: uuid.uuid4().hex)


class WorkingMemoryBuffer:
    """Short-term memory buffer with a fixed capacity."""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity
        self.items: List[Dict[str, Any]] = []

    def add(self, item: Dict[str, Any]) -> None:
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.pop(0)

    def to_dict(self) -> Dict[str, Any]:
        return {"capacity": self.capacity, "items": self.items}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingMemoryBuffer":
        buffer = cls(capacity=data.get("capacity", 32))
        buffer.items = list(data.get("items", []))
        return buffer


class PersistentMemoryManager:
    """Handles storage and retrieval of long-term and working memory."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = PersistentMemoryConfig(**(config or {}))
        self.path = Path(self.config.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(self.config.embedding_cache_path) if self.config.embedding_cache_path else None
        cache = DiskEmbeddingCache(cache_dir)
        self.embedding_backend = self._build_backend(cache)
        self.entries: List[MemoryEntry] = []
        self.working_memory = WorkingMemoryBuffer(self.config.working_memory_size)
        index_path = Path(self.config.index_path) if self.config.index_path else None
        self.index = VectorIndex(self.embedding_backend.dimension, self.config.index_backend, index_path)
        self._load()
        self._refresh_index()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        metadata = metadata.copy() if metadata else {}
        embedding = self.embedding_backend.embed([content])
        if embedding.size == 0:
            raise ValueError("Embedding backend returned no data")
        entry = MemoryEntry(content=content, metadata=metadata, embedding=embedding[0].tolist())
        self.entries.append(entry)
        if len(self.entries) > self.config.max_entries:
            self.entries.pop(0)
        self._save()
        self._refresh_index()
        self.working_memory.add({"content": content, "metadata": metadata, "timestamp": entry.timestamp})
        return entry

    def add_memories(self, items: Sequence[Tuple[str, Dict[str, Any]]]) -> List[MemoryEntry]:
        """Batch ingest memories respecting configured batch size."""

        created: List[MemoryEntry] = []
        batch_size = max(1, self.config.ingestion_batch_size)
        batch: List[str] = []
        meta_batch: List[Dict[str, Any]] = []

        def _flush() -> None:
            nonlocal batch, meta_batch
            if not batch:
                return
            embeddings = self.embedding_backend.embed(batch)
            for idx, content in enumerate(batch):
                metadata = meta_batch[idx]
                embedding = embeddings[idx].tolist()
                entry = MemoryEntry(content=content, metadata=metadata, embedding=embedding)
                self.entries.append(entry)
                created.append(entry)
                self.working_memory.add({"content": content, "metadata": metadata, "timestamp": entry.timestamp})
            batch = []
            meta_batch = []

        for content, metadata in items:
            batch.append(content)
            meta_batch.append(metadata.copy() if metadata else {})
            if len(batch) >= batch_size:
                _flush()
        _flush()

        if len(self.entries) > self.config.max_entries:
            self.entries = self.entries[-self.config.max_entries :]
        self._save()
        self._refresh_index()
        return created

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.entries:
            return []
        query_vec = self.embedding_backend.embed([query])
        if query_vec.size == 0:
            return []
        scored_indices = self.index.search(query_vec[0], top_k)
        results: List[Dict[str, Any]] = []
        for score, idx in scored_indices:
            if idx < 0 or idx >= len(self.entries):
                continue
            entry = self.entries[idx]
            results.append(
                {
                    "id": entry.uid,
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "score": float(score),
                    "timestamp": entry.timestamp,
                }
            )
        return results

    def working_items(self) -> List[Dict[str, Any]]:
        return list(self.working_memory.items)

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def _save(self) -> None:
        data = {
            "entries": [
                {
                    "id": entry.uid,
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "embedding": entry.embedding,
                    "timestamp": entry.timestamp,
                }
                for entry in self.entries
            ],
            "working_memory": self.working_memory.to_dict(),
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = [
                MemoryEntry(
                    uid=str(item.get("id")) if item.get("id") else uuid.uuid4().hex,
                    content=str(item.get("content", "")),
                    metadata=item.get("metadata", {}),
                    embedding=list(item.get("embedding", [])),
                    timestamp=float(item.get("timestamp", time.time())),
                )
                for item in data.get("entries", [])
            ]
            self.working_memory = WorkingMemoryBuffer.from_dict(data.get("working_memory", {}))
        except Exception:
            self.entries = []
            self.working_memory = WorkingMemoryBuffer(self.config.working_memory_size)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [
                {
                    "id": entry.uid,
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "embedding": entry.embedding,
                    "timestamp": entry.timestamp,
                }
                for entry in self.entries
            ],
            "working_memory": self.working_memory.to_dict(),
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self.entries = [
            MemoryEntry(
                uid=str(item.get("id")) if item.get("id") else uuid.uuid4().hex,
                content=item.get("content", ""),
                metadata=item.get("metadata", {}),
                embedding=list(item.get("embedding", [])),
                timestamp=float(item.get("timestamp", time.time())),
            )
            for item in data.get("entries", [])
        ]
        self.working_memory = WorkingMemoryBuffer.from_dict(data.get("working_memory", {}))
        self._save()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_backend(self, cache: DiskEmbeddingCache) -> EmbeddingBackend:
        backend = self.config.embedding_backend.lower()
        if backend == "sentence-transformer":
            try:
                st_backend = SentenceTransformerBackend(self.config.embedding_model, cache)
                self.config.embedding_dim = st_backend.dimension
                return st_backend
            except Exception:
                # fall back to hashing if sentence transformers unavailable
                pass
        return HashEmbeddingBackend(self.config.embedding_dim, cache)

    def _refresh_index(self) -> None:
        if not self.entries:
            self.index.rebuild(np.zeros((0, self.embedding_backend.dimension), dtype=np.float32))
            return
        vectors = np.asarray([entry.embedding for entry in self.entries], dtype=np.float32)
        self.index.rebuild(vectors)
