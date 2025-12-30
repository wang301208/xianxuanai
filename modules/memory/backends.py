from __future__ import annotations

"""Pluggable ANN and archive backends for vector memory."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None
try:  # optional dependency
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http import models as qdrant_models  # type: ignore
except Exception:  # pragma: no cover
    QdrantClient = None
    qdrant_models = None  # type: ignore
try:  # optional dependency
    import boto3  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None


logger = logging.getLogger(__name__)


class VectorANNBackend:
    """Abstract interface for approximate nearest-neighbour services."""

    def upsert(self, record_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        raise NotImplementedError

    def delete(self, record_id: str) -> None:
        raise NotImplementedError

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional
        """Close backend resources."""


class NoOpANNBackend(VectorANNBackend):
    """Fallback ANN backend that does nothing."""

    def upsert(self, record_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        logger.debug("NoOpANNBackend.upsert(%s) noop", record_id)

    def delete(self, record_id: str) -> None:
        logger.debug("NoOpANNBackend.delete(%s) noop", record_id)

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        return []


class InMemoryANNBackend(VectorANNBackend):
    """Simple ANN backend using FAISS or brute force for development."""

    def __init__(self) -> None:
        self._dimension: Optional[int] = None
        self._index = None
        self._vectors: List[np.ndarray] = []
        self._records: Dict[str, Dict[str, Any]] = {}
        self._id_to_pos: Dict[str, int] = {}

    def upsert(self, record_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError("ANN backend expects 1D vectors")
        if self._dimension is None:
            self._dimension = vec.shape[0]
            if faiss is not None:
                self._index = faiss.IndexFlatIP(self._dimension)
        elif vec.shape[0] != self._dimension:
            raise ValueError(
                f"Inconsistent vector dimension: expected {self._dimension}, received {vec.shape[0]}"
            )
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        if record_id in self._records:
            pos = self._id_to_pos[record_id]
            self._vectors[pos] = vec
        else:
            self._id_to_pos[record_id] = len(self._vectors)
            self._vectors.append(vec)
        self._records[record_id] = dict(metadata)
        if self._index is not None:
            self._index = self._rebuild_index()

    def delete(self, record_id: str) -> None:
        if record_id not in self._records:
            return
        pos = self._id_to_pos.pop(record_id)
        self._records.pop(record_id, None)
        self._vectors.pop(pos)
        self._id_to_pos = {rid: idx for idx, rid in enumerate(self._records)}
        if self._index is not None:
            self._index = self._rebuild_index()

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not self._records:
            return []
        vec = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        top_k = min(max(top_k, 1), len(self._records))
        if self._index is not None:
            distances, indices = self._index.search(vec[None], top_k)
            idxs = indices[0]
            scores = distances[0]
        else:
            matrix = np.vstack(self._vectors)
            similarities = matrix @ vec
            idxs = np.argsort(-similarities)[:top_k]
            scores = similarities[idxs]
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for idx, score in zip(idxs, scores):
            if idx < 0 or idx >= len(self._vectors):
                continue
            record_id = list(self._records.keys())[idx]
            results.append((record_id, float(score), dict(self._records[record_id])))
        return results

    def _rebuild_index(self):
        if faiss is None or self._dimension is None:
            return None
        index = faiss.IndexFlatIP(self._dimension)
        if self._vectors:
            index.add(np.vstack(self._vectors))
        return index


class QdrantANNBackend(VectorANNBackend):
    """Qdrant-backed ANN service for distributed vector search."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: Optional[int] = None,
        https: bool = False,
        api_key: Optional[str] = None,
        collection: str = "skills_vector_store",
        prefer_grpc: Optional[bool] = None,
        timeout: Optional[float] = None,
        distance: str = "cosine",
        vector_size: Optional[int] = None,
        shard_number: Optional[int] = None,
        on_disk: bool = False,
        wait_result: bool = True,
        **client_kwargs: Any,
    ) -> None:
        if QdrantClient is None or qdrant_models is None:
            raise RuntimeError("qdrant-client package is required for QdrantANNBackend.")

        self._wait = wait_result
        self._shard_number = shard_number
        self._on_disk = on_disk
        self._timeout = timeout

        prefer_grpc = prefer_grpc if prefer_grpc is not None else bool(grpc_port)
        self._client = QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            timeout=timeout,
            **client_kwargs,
        )
        self._default_collection = collection
        self._collections_ready: Dict[str, bool] = {}
        self._vector_size_hint = vector_size
        self._collection_distance = self._resolve_distance(distance)
        self._shard_number = shard_number
        self._on_disk = on_disk
        self._wait = wait_result
        self._timeout = timeout
        self._vector_sizes: Dict[str, int] = {}
        self._record_to_collection: Dict[str, str] = {}

    # ------------------------------------------------------------------ VectorANNBackend API
    def upsert(self, record_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        vector = self._as_float_vector(vector)
        collection = self._resolve_collection(metadata)
        self._ensure_collection(collection, vector.shape[0])
        payload = dict(metadata or {})
        point = qdrant_models.PointStruct(
            id=record_id,
            vector=vector.tolist(),
            payload=payload,
        )
        self._client.upsert(
            collection_name=collection,
            wait=self._wait,
            points=[point],
        )
        self._record_to_collection[record_id] = collection

    def delete(self, record_id: str) -> None:
        collection = self._record_to_collection.pop(record_id, None)
        candidates = [collection] if collection is not None else list(self._collections_ready.keys())
        selector = qdrant_models.PointIdsList(points=[record_id])
        for name in candidates:
            if not name:
                continue
            if not self._collections_ready.get(name):
                continue
            try:
                self._client.delete(
                    collection_name=name,
                    wait=self._wait,
                    points_selector=selector,
                )
                break
            except Exception:
                logger.debug("Qdrant delete failed for collection %s", name, exc_info=True)

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        **kwargs: Any,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        vector = self._as_float_vector(vector)
        collections = kwargs.get("collections")
        if collections:
            names = [str(name) for name in collections if name]
        else:
            collection_hint = kwargs.get("collection")
            names = [str(collection_hint)] if collection_hint else [self._default_collection]
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        remaining = max(1, top_k)
        for name in names:
            try:
                self._ensure_collection(name, vector.shape[0])
            except Exception:
                continue
            try:
                search_result = self._client.search(
                    collection_name=name,
                    query_vector=vector.tolist(),
                    limit=max(1, remaining),
                    with_payload=True,
                )
            except Exception:
                logger.debug("Qdrant search failed for collection %s", name, exc_info=True)
                continue
            for point in search_result:
                payload = dict(getattr(point, "payload", {}) or {})
                record_id = str(point.id)
                results.append((record_id, float(point.score), payload))
                self._record_to_collection.setdefault(record_id, name)
            if len(results) >= top_k:
                break
        return results[:top_k]

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("Failed to close Qdrant client", exc_info=True)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _as_float_vector(vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    @staticmethod
    def _resolve_distance(distance: str):
        if qdrant_models is None:
            raise RuntimeError("qdrant-client models unavailable.")
        manhattan = getattr(qdrant_models.Distance, "MANHATTAN", qdrant_models.Distance.EUCLID)
        mapping = {
            "cosine": qdrant_models.Distance.COSINE,
            "dot": qdrant_models.Distance.DOT,
            "ip": qdrant_models.Distance.DOT,
            "euclid": qdrant_models.Distance.EUCLID,
            "l2": qdrant_models.Distance.EUCLID,
            "manhattan": manhattan,
        }
        key = str(distance or "cosine").lower()
        return mapping.get(key, qdrant_models.Distance.COSINE)

    def _ensure_collection(self, collection: str, vector_size: int) -> None:
        ready = self._collections_ready.get(collection)
        known_size = self._vector_sizes.get(collection)
        if ready and known_size:
            if vector_size != known_size:
                raise ValueError(
                    f"Qdrant collection '{collection}' expects dimension {known_size}, got {vector_size}"
                )
            return

        try:
            info = self._client.get_collection(collection_name=collection)
        except Exception:
            info = None

        if info is not None and getattr(info, "status", None):
            params = info.config.params  # type: ignore[attr-defined]
            if hasattr(params, "vectors") and getattr(params.vectors, "size", None):
                size = int(params.vectors.size)
                self._vector_sizes[collection] = size
            else:
                self._vector_sizes[collection] = vector_size
            self._collections_ready[collection] = True
            return

        target_size = (
            known_size
            or self._vector_sizes.get(self._default_collection)
            or self._vector_size_hint
            or vector_size
        )
        if target_size is None or target_size <= 0:
            raise RuntimeError("Qdrant vector size must be specified before first upsert.")

        vector_params = qdrant_models.VectorParams(
            size=int(target_size),
            distance=self._collection_distance,
            on_disk=self._on_disk,
        )
        try:
            self._client.recreate_collection(
                collection_name=collection,
                vectors_config=vector_params,
                shard_number=self._shard_number,
                timeout=self._timeout,
            )
            self._vector_sizes[collection] = int(target_size)
            self._collections_ready[collection] = True
        except Exception as exc:
            raise RuntimeError(f"Failed to prepare Qdrant collection '{collection}': {exc}") from exc

    def _resolve_collection(self, metadata: Dict[str, Any]) -> str:
        collection = metadata.get("collection")
        meta = metadata.get("metadata")
        if isinstance(meta, dict):
            collection = collection or meta.get("shard") or meta.get("collection")
        if isinstance(collection, str) and collection.strip():
            return collection.strip()
        return self._default_collection


class VectorArchiveBackend:
    """Abstract interface for long-term storage of vector payloads."""

    def store(self, record: Dict[str, Any]) -> None:
        raise NotImplementedError

    def load(self, record_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def delete(self, record_id: str) -> None:
        raise NotImplementedError


@dataclass
class FileArchiveBackend(VectorArchiveBackend):
    """Simple JSON-lines archive stored on disk."""

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "archive.jsonl"
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def store(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self, record_id: str) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in reversed(handle.readlines()):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("id") == record_id:
                    return data
        return None

    def delete(self, record_id: str) -> None:
        if not self.path.exists():
            return
        lines = self.path.read_text(encoding="utf-8").splitlines()
        updated = [line for line in lines if json.loads(line).get("id") != record_id]
        self.path.write_text("\n".join(updated) + ("\n" if updated else ""), encoding="utf-8")


@dataclass
class S3ArchiveBackend(VectorArchiveBackend):
    """Archive backend backed by S3/MinIO compatible storage."""

    bucket: str
    prefix: str = "vectors"
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    session_token: Optional[str] = None
    region: Optional[str] = None
    secure: bool = True

    def __post_init__(self) -> None:
        if boto3 is None:
            raise RuntimeError("boto3 is required for the S3 archive backend.")
        session_kwargs: Dict[str, Any] = {}
        if self.access_key and self.secret_key:
            session_kwargs["aws_access_key_id"] = self.access_key
            session_kwargs["aws_secret_access_key"] = self.secret_key
        if self.session_token:
            session_kwargs["aws_session_token"] = self.session_token
        if self.region:
            session_kwargs["region_name"] = self.region
        self._client = boto3.client(  # type: ignore[attr-defined]
            "s3",
            endpoint_url=self.endpoint_url,
            use_ssl=self.secure,
            **session_kwargs,
        )
        self.prefix = (self.prefix or "").strip("/")

    def _key(self, record_id: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{record_id}.json"
        return f"{record_id}.json"

    def store(self, record: Dict[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False).encode("utf-8")
        try:
            self._client.put_object(Bucket=self.bucket, Key=self._key(record["id"]), Body=payload)
        except Exception:
            logger.warning("S3 archive store failed for %s", record["id"], exc_info=True)

    def load(self, record_id: str) -> Optional[Dict[str, Any]]:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=self._key(record_id))
        except Exception:
            return None
        body = response.get("Body")
        if body is None:
            return None
        raw = body.read().decode("utf-8")
        return json.loads(raw)

    def delete(self, record_id: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=self._key(record_id))
        except Exception:
            logger.debug("S3 archive delete failed for %s", record_id, exc_info=True)


def _parse_bool(value: Optional[str], *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_ann_backend_from_env() -> Optional[VectorANNBackend]:
    backend_name = os.getenv("VECTOR_ANN_BACKEND", "").lower()
    if backend_name == "memory":
        logger.info("Using in-memory ANN backend for vector store.")
        return InMemoryANNBackend()
    if backend_name == "qdrant":
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333") or 6333)
        grpc_port_env = os.getenv("QDRANT_GRPC_PORT", "")
        grpc_port = int(grpc_port_env) if grpc_port_env else None
        prefer_grpc = _parse_bool(os.getenv("QDRANT_PREFER_GRPC"))
        api_key = os.getenv("QDRANT_API_KEY")
        collection = os.getenv("QDRANT_COLLECTION", "skills_vector_store")
        distance = os.getenv("QDRANT_DISTANCE", "cosine")
        vector_size_env = os.getenv("QDRANT_VECTOR_SIZE", "")
        vector_size = int(vector_size_env) if vector_size_env else None
        shard_env = os.getenv("QDRANT_SHARD_NUMBER", "")
        shard_number = int(shard_env) if shard_env else None
        on_disk = _parse_bool(os.getenv("QDRANT_ON_DISK"))
        wait_result = _parse_bool(os.getenv("QDRANT_WAIT_RESULT"), default=True)
        timeout_env = os.getenv("QDRANT_TIMEOUT", "")
        timeout = float(timeout_env) if timeout_env else None
        https = _parse_bool(os.getenv("QDRANT_HTTPS"))
        if _parse_bool(os.getenv("QDRANT_PREFER_TLS")):
            prefer_grpc = False
            https = True
        try:
            backend = QdrantANNBackend(
                host=host,
                port=port,
                grpc_port=grpc_port,
                https=https,
                api_key=api_key,
                collection=collection,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
                distance=distance,
                vector_size=vector_size,
                shard_number=shard_number,
                on_disk=on_disk,
                wait_result=wait_result,
            )
            logger.info("Using Qdrant ANN backend (collection=%s).", collection)
            return backend
        except Exception as exc:
            logger.warning("Failed to initialise Qdrant ANN backend: %s", exc, exc_info=True)
            return None
    if backend_name:
        logger.warning("Unsupported VECTOR_ANN_BACKEND '%s'; defaulting to NoOp.", backend_name)
        return NoOpANNBackend()
    return None


def build_archive_backend_from_env(default_root: Path) -> Optional[VectorArchiveBackend]:
    backend_name = os.getenv("VECTOR_ARCHIVE_BACKEND", "").lower()
    if backend_name in {"file", "filesystem", "local"}:
        logger.info("Using file-based archive backend for vector store.")
        return FileArchiveBackend(default_root / "archive")
    if backend_name in {"s3", "minio", "s3-compatible", "s3compatible"}:
        if boto3 is None:
            logger.warning("boto3 not available; cannot configure S3 archive backend.")
            return None
        bucket = os.getenv("VECTOR_ARCHIVE_S3_BUCKET")
        if not bucket:
            logger.warning("VECTOR_ARCHIVE_S3_BUCKET must be set for S3 archive backend.")
            return None
        prefix = os.getenv("VECTOR_ARCHIVE_S3_PREFIX", "vectors")
        endpoint = os.getenv("VECTOR_ARCHIVE_S3_ENDPOINT")
        access_key = os.getenv("VECTOR_ARCHIVE_S3_ACCESS_KEY")
        secret_key = os.getenv("VECTOR_ARCHIVE_S3_SECRET_KEY")
        session_token = os.getenv("VECTOR_ARCHIVE_S3_SESSION_TOKEN")
        region = os.getenv("VECTOR_ARCHIVE_S3_REGION")
        secure = _parse_bool(os.getenv("VECTOR_ARCHIVE_S3_SECURE"), default=True)
        logger.info("Using S3 archive backend (bucket=%s).", bucket)
        return S3ArchiveBackend(
            bucket=bucket,
            prefix=prefix,
            endpoint_url=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            session_token=session_token,
            region=region,
            secure=secure,
        )
    if backend_name:
        logger.warning("Unsupported VECTOR_ARCHIVE_BACKEND '%s'; archive disabled.", backend_name)
    return None


__all__ = [
    "VectorANNBackend",
    "VectorArchiveBackend",
    "NoOpANNBackend",
    "InMemoryANNBackend",
    "QdrantANNBackend",
    "FileArchiveBackend",
    "S3ArchiveBackend",
    "build_ann_backend_from_env",
    "build_archive_backend_from_env",
]
