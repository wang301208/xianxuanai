from __future__ import annotations

import logging
import json
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback when faiss unavailable
    faiss = None

EmbeddingFn = Callable[[str], Sequence[float]]
if TYPE_CHECKING:  # pragma: no cover - typing only
    from .embedders import TextEmbedder

EmbeddingLike = Union[EmbeddingFn, "TextEmbedder"]
logger = logging.getLogger(__name__)

from .backends import (
    VectorANNBackend,
    VectorArchiveBackend,
    build_ann_backend_from_env,
    build_archive_backend_from_env,
)


@dataclass
class VectorRecord:
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float = 0.0


class _LRUCache:
    """Simple LRU cache used for query hot data."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, capacity)
        self._store: "OrderedDict[Tuple[Any, ...], List[VectorRecord]]" = OrderedDict()

    def get(self, key: Tuple[Any, ...]) -> Optional[List[VectorRecord]]:
        if self.capacity <= 0 or key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value
        return [VectorRecord(r.id, r.text, dict(r.metadata), r.score) for r in value]

    def put(self, key: Tuple[Any, ...], value: List[VectorRecord]) -> None:
        if self.capacity <= 0:
            return
        snapshot = [VectorRecord(r.id, r.text, dict(r.metadata), r.score) for r in value]
        if key in self._store:
            self._store.pop(key)
        self._store[key] = snapshot
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()


class VectorMemoryStore:
    """Layered vector memory with hot cache, local store, and optional ANN/archive."""

    def __init__(
        self,
        storage_path: Path,
        embedder: EmbeddingLike | str | None = None,
        backend: str | None = None,
        *,
        embedder_options: Optional[Dict[str, Any]] = None,
        ann_backend: Optional[VectorANNBackend] = None,
        archive_backend: Optional[VectorArchiveBackend] = None,
        hot_cache_size: int = 256,
    ) -> None:
        self.storage_path = storage_path
        self.embedder, self._embedder_kind = self._initialise_embedder(embedder, embedder_options)
        self.backend = backend or ("faiss" if faiss is not None else "brute")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_path / "index.faiss"
        self.vectors_path = self.storage_path / "vectors.npy"
        self.meta_path = self.storage_path / "metadata.json"

        self._dimension: Optional[int] = None
        self._index = None
        self._vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._metadata: List[Dict[str, Any]] = []

        self._ann_backend = ann_backend if ann_backend is not None else build_ann_backend_from_env()
        self._archive_backend = (
            archive_backend
            if archive_backend is not None
            else build_archive_backend_from_env(self.storage_path)
        )
        self._query_cache = _LRUCache(hot_cache_size)
        self._ann_backlog: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
        self._record_shards: Dict[str, str] = {}
        self._shards_last_seen: Dict[str, float] = {}
        self._shard_strategy = os.getenv("VECTOR_SHARD_STRATEGY", "time").lower()
        self._default_shard = os.getenv("VECTOR_DEFAULT_SHARD", "default")
        self._max_shard_history = max(1, int(os.getenv("VECTOR_SHARD_HISTORY", "32") or 32))

        self._load()

    # ------------------------------------------------------------------ public API
    def add_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        record_id: Optional[str] = None,
    ) -> str:
        self._drain_ann_backlog()
        vector = self._encode(text)
        if record_id is None:
            record_id = str(uuid.uuid4())
        entry_metadata = dict(metadata or {})
        now = entry_metadata.get("created_at", time.time())
        entry_metadata.setdefault("created_at", now)
        entry_metadata.setdefault("last_access", now)
        entry_metadata.setdefault("usage", 0)
        entry_metadata.setdefault("weight", 1.0)
        entry_metadata.setdefault("promoted", False)
        entry_metadata.setdefault("archived", False)
        shard = entry_metadata.get("shard") or self._assign_shard(entry_metadata)
        entry_metadata["shard"] = shard
        entry = {"id": record_id, "text": text, "metadata": entry_metadata}
        self._metadata.append(entry)
        self._add_vector(vector)
        self._record_shards[record_id] = shard
        self._track_shard(shard, now)
        self._persist()
        self._query_cache.clear()
        if not self._upsert_ann(record_id, vector, entry):
            self._ann_backlog.append((record_id, vector.copy(), entry))
        return record_id

    def add_batch(
        self,
        texts: Iterable[str],
        metadatas: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> List[str]:
        ids: List[str] = []
        metadata_iter = iter(metadatas or [])
        for text in texts:
            metadata = next(metadata_iter, {})
            ids.append(self.add_text(text, metadata))
        return ids

    def query(
        self,
        text: str,
        top_k: int = 5,
        *,
        shard: Optional[str] = None,
        shards: Optional[Sequence[str]] = None,
        include_archived: bool = False,
    ) -> List[VectorRecord]:
        try:
            top_k = int(top_k)
        except Exception:
            top_k = 5
        if top_k <= 0:
            return []
        cache_key = (text, top_k, shard or "", tuple(shards or []), bool(include_archived))
        cached = self._query_cache.get(cache_key)
        if cached:
            for record in cached:
                self.record_access(record.id)
            return cached

        self._drain_ann_backlog()
        vector = self._encode(text)
        results: Dict[str, VectorRecord] = {}
        for record in self._query_local(vector, top_k, include_archived=include_archived):
            results[record.id] = record

        if self._ann_backend is not None:
            try:
                candidate_shards = self._resolve_query_shards(shard, shards)
                ann_candidates = self._ann_backend.query(
                    vector, top_k * 2, collections=candidate_shards
                )
            except Exception:
                logger.debug("ANN backend query failed", exc_info=True)
                ann_candidates = []
            for record_id, score, ann_meta in ann_candidates:
                if record_id in results:
                    results[record_id].score = max(results[record_id].score, score)
                    continue
                record = self._load_record_from_backends(record_id, ann_meta)
                if record is None:
                    continue
                if not include_archived and record.metadata.get("archived"):
                    continue
                record.score = score
                results[record.id] = record

        ordered = sorted(results.values(), key=lambda r: r.score, reverse=True)[: top_k]
        if ordered:
            for record in ordered:
                self.record_access(record.id)
            self._query_cache.put(cache_key, ordered)
        return ordered

    def __len__(self) -> int:
        return self._metadata_count

    def archive_record(self, record_id: str) -> bool:
        record = self._get_record(record_id)
        if record is None:
            return False
        record.setdefault("metadata", {})["archived"] = True
        if self._archive_backend is not None:
            try:
                self._archive_backend.store(dict(record))
            except Exception:
                logger.debug("Archive backend store failed", exc_info=True)
        if self._remove_local_record(record_id):
            if self._ann_backend is not None:
                try:
                    self._ann_backend.delete(record_id)
                except Exception:
                    logger.debug("ANN backend delete failed", exc_info=True)
            self._persist()
            self._query_cache.clear()
        return True

    def record_access(
        self,
        record_id: str,
        *,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        ts = timestamp or time.time()
        record = self._get_record(record_id)
        if record is None:
            return None
        metadata = record.setdefault("metadata", {})
        metadata["last_access"] = ts
        metadata["usage"] = int(metadata.get("usage", 0)) + 1
        return metadata

    def update_metadata(self, record_id: str, updates: Dict[str, Any]) -> bool:
        record = self._get_record(record_id)
        if record is None:
            return False
        metadata = record.setdefault("metadata", {})
        metadata.update(updates)
        return True

    def iter_records(self) -> Iterable[Dict[str, Any]]:
        return iter(self._metadata)

    # ------------------------------------------------------------------ internals
    def _query_local(self, vector: np.ndarray, top_k: int, *, include_archived: bool) -> List[VectorRecord]:
        if self._metadata_count == 0:
            return []
        limit = min(max(1, top_k) * 4, self._metadata_count)
        if self._use_faiss:
            distances, indices = self._index.search(vector[None], limit)
            idxs = indices[0]
            scores = distances[0]
        else:
            similarities = np.dot(self._vectors, vector)
            idxs = np.argsort(-similarities)[:limit]
            scores = similarities[idxs]
        results: List[VectorRecord] = []
        for idx, score in zip(idxs, scores):
            if idx < 0 or idx >= self._metadata_count:
                continue
            record = self._metadata[idx]
            metadata = dict(record.get("metadata", {}))
            if not include_archived and metadata.get("archived"):
                continue
            results.append(
                VectorRecord(
                    id=record["id"],
                    text=record["text"],
                    metadata=metadata,
                    score=float(score),
                )
            )
            if len(results) >= top_k:
                break
        return results

    @property
    def _metadata_count(self) -> int:
        return len(self._metadata)

    @property
    def _use_faiss(self) -> bool:
        return self.backend == "faiss" and faiss is not None and self._index is not None

    def _encode(self, text: str) -> np.ndarray:
        if hasattr(self.embedder, "encode"):
            encode_fn = getattr(self.embedder, "encode")
            if callable(encode_fn):
                vector = np.asarray(encode_fn(text), dtype=np.float32)
            else:
                vector = np.asarray(self.embedder(text), dtype=np.float32)
        else:
            vector = np.asarray(self.embedder(text), dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("embedding function must return a 1D vector")
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        if self._dimension is None:
            self._dimension = vector.shape[0]
            if self.backend == "faiss" and faiss is not None:
                self._index = faiss.IndexFlatIP(self._dimension)
                if self._vectors.size:
                    self._index.add(self._vectors)
        elif vector.shape[0] != self._dimension:
            raise ValueError(
                f"embedding dimension mismatch: expected {self._dimension}, got {vector.shape[0]}"
            )
        return vector

    def _initialise_embedder(
        self,
        embedder: EmbeddingLike | str | None,
        embedder_options: Optional[Dict[str, Any]],
    ) -> Tuple[EmbeddingLike, str]:
        options = dict(embedder_options or {})
        if embedder is not None and not isinstance(embedder, str):
            return embedder, type(embedder).__name__

        preference = options.pop("name", None)
        if isinstance(embedder, str):
            preference = embedder
        if preference is None:
            preference = os.getenv("VECTOR_EMBEDDER", "auto")
        preference = (preference or "auto").strip().lower()

        transformer_aliases = {
            "transformer",
            "sentence",
            "sentence-transformer",
            "sentence_transformer",
            "auto",
        }

        if preference in transformer_aliases:
            from .embedders import HashingEmbedder, TransformerEmbedder

            transformer_kwargs = self._transformer_options(options)
            try:
                embedder_instance = TransformerEmbedder(**transformer_kwargs)
                return embedder_instance, "TransformerEmbedder"
            except Exception:
                if preference not in {"auto"}:
                    logger.warning(
                        "Failed to initialise transformer embedder; falling back to HashingEmbedder.",
                        exc_info=True,
                    )
                else:
                    logger.debug(
                        "Transformer embeddings unavailable; using HashingEmbedder fallback.",
                        exc_info=True,
                    )
            hashing_kwargs = self._hashing_options(options)
            return HashingEmbedder(**hashing_kwargs), "HashingEmbedder"

        from .embedders import HashingEmbedder

        hashing_kwargs = self._hashing_options(options)
        return HashingEmbedder(**hashing_kwargs), "HashingEmbedder"

    def _transformer_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        opts: Dict[str, Any] = {}
        env_model = os.getenv("VECTOR_TRANSFORMER_MODEL")
        if env_model:
            opts.setdefault("model_name", env_model)
        env_device = os.getenv("VECTOR_TRANSFORMER_DEVICE")
        if env_device:
            opts.setdefault("device", env_device)
        env_normalize = os.getenv("VECTOR_TRANSFORMER_NORMALIZE")
        if env_normalize:
            opts.setdefault("normalize", env_normalize.strip().lower() not in {"0", "false", "no"})
        env_batch = os.getenv("VECTOR_TRANSFORMER_BATCH_SIZE")
        if env_batch:
            try:
                opts.setdefault("batch_size", int(env_batch))
            except ValueError:
                logger.warning(
                    "Invalid VECTOR_TRANSFORMER_BATCH_SIZE '%s'; ignoring.", env_batch
                )
        opts.update(options)
        return opts

    def _hashing_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        opts: Dict[str, Any] = {}
        env_dim = os.getenv("VECTOR_HASH_DIMENSION")
        if env_dim:
            try:
                opts.setdefault("dimension", int(env_dim))
            except ValueError:
                logger.warning("Invalid VECTOR_HASH_DIMENSION '%s'; ignoring.", env_dim)
        env_window = os.getenv("VECTOR_HASH_WINDOW")
        if env_window:
            try:
                opts.setdefault("window_size", int(env_window))
            except ValueError:
                logger.warning("Invalid VECTOR_HASH_WINDOW '%s'; ignoring.", env_window)
        for key in ("dimension", "window_size"):
            if key in options:
                opts[key] = options[key]
        return opts

    def _add_vector(self, vector: np.ndarray) -> None:
        if self._vectors.size == 0:
            self._vectors = vector.reshape(1, -1)
        else:
            self._vectors = np.vstack([self._vectors, vector])
        if self._use_faiss:
            self._index.add(vector[None])

    def _load_record_from_backends(
        self,
        record_id: str,
        ann_metadata: Optional[Dict[str, Any]],
    ) -> Optional[VectorRecord]:
        record = self._get_record(record_id)
        text: Optional[str] = None
        metadata: Dict[str, Any] = {}
        if record is not None:
            text = record.get("text")
            metadata = dict(record.get("metadata", {}))
        if isinstance(ann_metadata, dict):
            text = text or ann_metadata.get("text")
            meta_payload = ann_metadata.get("metadata")
            if isinstance(meta_payload, dict):
                metadata.update(meta_payload)
            else:
                metadata.update({k: v for k, v in ann_metadata.items() if k != "text"})
        if (text is None or text == "") and self._archive_backend is not None:
            try:
                archived = self._archive_backend.load(record_id)
            except Exception:
                logger.debug("Archive backend load failed", exc_info=True)
                archived = None
            if archived:
                text = archived.get("text", text)
                meta_payload = archived.get("metadata")
                if isinstance(meta_payload, dict):
                    metadata.update(meta_payload)
        if not text:
            return None
        shard = metadata.get("shard")
        if shard:
            self._record_shards[record_id] = shard
            self._track_shard(
                shard, float(metadata.get("last_access", metadata.get("created_at", time.time())))
            )
        return VectorRecord(record_id, text, metadata, 0.0)

    def _upsert_ann(self, record_id: str, vector: np.ndarray, entry: Dict[str, Any]) -> bool:
        if self._ann_backend is None:
            return True
        metadata = {
            "text": entry.get("text", ""),
            "metadata": dict(entry.get("metadata", {})),
        }
        shard = metadata["metadata"].get("shard")
        try:
            self._ann_backend.upsert(record_id, vector, metadata)
        except Exception:
            logger.debug("ANN backend upsert failed", exc_info=True)
            return False
        if shard:
            self._record_shards[record_id] = shard
            self._track_shard(shard, float(metadata["metadata"].get("last_access", time.time())))
        return True

    def _find_record_index(self, record_id: str) -> int:
        for idx, entry in enumerate(self._metadata):
            if entry.get("id") == record_id:
                return idx
        return -1

    def _remove_local_record(self, record_id: str) -> bool:
        idx = self._find_record_index(record_id)
        if idx == -1:
            return False
        self._metadata.pop(idx)
        if self._vectors.size:
            self._vectors = np.delete(self._vectors, idx, axis=0)
        if self._dimension is not None and faiss is not None:
            if self._vectors.size:
                self._index = faiss.IndexFlatIP(self._dimension)
                self._index.add(self._vectors)
            else:
                self._index = faiss.IndexFlatIP(self._dimension)
        else:
            self._index = None
        self._record_shards.pop(record_id, None)
        return True

    def _drain_ann_backlog(self) -> None:
        if self._ann_backend is None or not self._ann_backlog:
            return
        pending = list(self._ann_backlog)
        self._ann_backlog.clear()
        for record_id, vector, entry in pending:
            if not self._upsert_ann(record_id, vector, entry):
                self._ann_backlog.append((record_id, vector, entry))
                break

    def _track_shard(self, shard: Optional[str], timestamp: float) -> None:
        if not shard:
            return
        current = self._shards_last_seen.get(shard, 0.0)
        if timestamp <= current:
            return
        self._shards_last_seen[shard] = timestamp
        if len(self._shards_last_seen) > self._max_shard_history:
            surplus = len(self._shards_last_seen) - self._max_shard_history
            for name, _ in sorted(self._shards_last_seen.items(), key=lambda item: item[1])[:surplus]:
                self._shards_last_seen.pop(name, None)

    def _assign_shard(self, metadata: Dict[str, Any]) -> str:
        shard_hint = metadata.get("shard") or metadata.get("collection")
        if shard_hint:
            return str(shard_hint)
        strategy = self._shard_strategy
        if strategy in {"domain", "topic"}:
            domain = metadata.get("domain") or metadata.get("topic")
            if isinstance(domain, (list, tuple, set)):
                domain = next((item for item in domain if item), None)
            if domain:
                return f"domain:{str(domain).strip().lower()}"
        if strategy in {"time", "temporal"}:
            ts = float(metadata.get("created_at", time.time()))
            bucket = time.strftime("%Y%m%d", time.gmtime(ts))
            return f"date:{bucket}"
        return self._default_shard

    def _resolve_query_shards(
        self,
        shard: Optional[str],
        shards: Optional[Sequence[str]],
    ) -> List[str]:
        if shards:
            selection = [str(name) for name in shards if name]
        elif shard:
            selection = [shard]
        else:
            ordered = sorted(
                self._shards_last_seen.items(), key=lambda item: item[1], reverse=True
            )
            selection = [name for name, _ in ordered]
            if self._default_shard not in selection:
                selection.append(self._default_shard)
        if not selection:
            selection = [self._default_shard]
        unique: List[str] = []
        for name in selection:
            if name not in unique:
                unique.append(name)
            if len(unique) >= self._max_shard_history:
                break
        return unique

    def _replace_vector(self, record_id: str, vector: np.ndarray) -> None:
        idx = self._find_record_index(record_id)
        if idx == -1:
            return
        if self._dimension is None:
            self._dimension = vector.shape[0]
        if vector.shape[0] != self._dimension:
            logger.warning(
                "Vector dimension mismatch while restoring record %s: expected %s, got %s",
                record_id,
                self._dimension,
                vector.shape[0],
            )
            return
        if self._vectors.size == 0:
            self._vectors = vector.reshape(1, -1)
        else:
            self._vectors[idx] = vector
        if self._use_faiss:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(self._vectors)

    def run_heat_maintenance(
        self,
        *,
        max_hot: Optional[int] = None,
        idle_seconds: Optional[float] = None,
        batch_size: int = 64,
    ) -> Dict[str, int]:
        now = time.time()
        active: List[Tuple[str, int, float]] = []
        for entry in self._metadata:
            meta = entry.get("metadata", {})
            if meta.get("archived"):
                continue
            last_access = float(meta.get("last_access", meta.get("created_at", now)))
            usage = int(meta.get("usage", 0))
            active.append((entry["id"], usage, last_access))
        cold_set: set[str] = set()
        if idle_seconds is not None:
            for record_id, _, last_access in active:
                if now - last_access >= idle_seconds:
                    cold_set.add(record_id)
        if max_hot is not None:
            remaining = [item for item in active if item[0] not in cold_set]
            excess = len(remaining) - max_hot
            if excess > 0:
                ranked = sorted(remaining, key=lambda item: (item[1], item[2]))
                for record_id, _, _ in ranked:
                    if excess <= 0 or len(cold_set) >= batch_size:
                        break
                    cold_set.add(record_id)
                    excess -= 1
        archived = 0
        for record_id in list(cold_set)[:batch_size]:
            if self.archive_record(record_id):
                archived += 1
        return {"archived": archived, "active": self._metadata_count}

    def restore_record(self, record_id: str) -> bool:
        if self._get_record(record_id) is not None:
            return True
        if self._archive_backend is None:
            return False
        try:
            archived = self._archive_backend.load(record_id)
        except Exception:
            logger.debug("Archive backend load failed during restore", exc_info=True)
            archived = None
        if not archived:
            return False
        text = archived.get("text")
        if not text:
            return False
        metadata = dict(archived.get("metadata", {}))
        vector_data = archived.get("vector")
        restored_id = self.add_text(text, metadata, record_id=record_id)
        if restored_id != record_id:
            return False
        if isinstance(vector_data, list):
            try:
                vector = np.asarray(vector_data, dtype=np.float32)
            except Exception:
                vector = None
            if vector is not None and vector.ndim == 1:
                self._replace_vector(record_id, vector)
        return True

    def _get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        for record in self._metadata:
            if record.get("id") == record_id:
                return record
        return None

    def close(self) -> None:
        if self._ann_backend is not None:
            try:
                self._ann_backend.close()
            except Exception:
                logger.debug("ANN backend close failed", exc_info=True)
        self._persist()

    def _persist(self) -> None:
        np.save(self.vectors_path, self._vectors)
        payload = {
            "backend": self.backend,
            "dimension": self._dimension,
            "records": self._metadata,
        }
        self.meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if self._use_faiss:
            faiss.write_index(self._index, str(self.index_path))

    def _load(self) -> None:
        if self.meta_path.exists():
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.backend = data.get("backend", self.backend)
            self._dimension = data.get("dimension")
            self._metadata = data.get("records", [])
            for entry in self._metadata:
                meta = entry.get("metadata", {})
                shard = meta.get("shard")
                record_id = entry.get("id")
                if shard and record_id:
                    self._record_shards[record_id] = shard
                    self._track_shard(
                        shard,
                        float(meta.get("last_access", meta.get("created_at", time.time()))),
                    )
        if self.backend == "faiss" and faiss is None:
            self.backend = "brute"
        if self.vectors_path.exists():
            self._vectors = np.load(self.vectors_path)
            if self._vectors.ndim == 1:
                self._vectors = self._vectors.reshape(1, -1)
        if self._dimension and self.backend == "faiss" and faiss is not None:
            if self.index_path.exists():
                self._index = faiss.read_index(str(self.index_path))
            else:
                self._index = faiss.IndexFlatIP(self._dimension)
                if self._vectors.size:
                    self._index.add(self._vectors)
        if not self._use_faiss and self._vectors.size and self._dimension is None:
            self._dimension = self._vectors.shape[1]
