"""SQLite backed long-term memory store.

The implementation is intentionally minimal – it creates a single table to
persist pieces of information along with a category.  This allows the system to
record dialogue snippets, tasks, or references to external knowledge sources
and retrieve them later.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


class LongTermMemory:
    """Light‑weight long-term memory based on SQLite."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_entries: Optional[int] = None,
        vacuum_interval: int = 1000,
        cache_pages: int = 1000,
        forget_interval: int = 0,
        recycle_interval: int = 0,
    ) -> None:
        """Parameters

        path:
            Location of the sqlite database.
        max_entries:
            Optional upper bound for stored memories. When exceeded the oldest
            entries are purged.  ``None`` disables the limit.
        vacuum_interval:
            Perform a ``VACUUM`` after this many insert operations. ``0``
            disables automatic vacuuming.
        forget_interval:
            Apply the forgetting strategy after this many insert operations.
            ``0`` disables automatic forgetting.
        recycle_interval:
            Apply the recycling strategy after this many insert operations.
            ``0`` disables automatic recycling.
        cache_pages:
            Page count for the SQLite cache.  Smaller values keep the memory
            footprint low.
        """

        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)

        # Tune the database for a small memory footprint and concurrent access
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(f"PRAGMA cache_size={int(cache_pages)}")

        self.max_entries = max_entries
        self.vacuum_interval = vacuum_interval
        self.forget_interval = forget_interval
        self.recycle_interval = recycle_interval
        self.forget_strategy: Optional[Callable[[sqlite3.Connection], None]] = None
        self.recycle_strategy: Optional[Callable[[sqlite3.Connection], None]] = None
        self.compression_strategy: Optional[Callable[[sqlite3.Connection], None]] = None
        self._pending_adds = 0
        self._total_adds = 0
        self._in_batch = False

        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL DEFAULT (strftime('%s','now')),
                tags TEXT
            )
            """
        )
        # Migration for existing databases missing new columns
        cur.execute("PRAGMA table_info(memory)")
        columns = {row[1] for row in cur.fetchall()}
        if "timestamp" not in columns:
            cur.execute(
                "ALTER TABLE memory ADD COLUMN timestamp REAL NOT NULL DEFAULT (strftime('%s','now'))"
            )
        if "tags" not in columns:
            cur.execute("ALTER TABLE memory ADD COLUMN tags TEXT")
        if "confidence" not in columns:
            cur.execute(
                "ALTER TABLE memory ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0"
            )
        if "status" not in columns:
            cur.execute(
                "ALTER TABLE memory ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
            )
        if "metadata" not in columns:
            cur.execute("ALTER TABLE memory ADD COLUMN metadata TEXT")
        # Create indexes for faster lookups
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_category ON memory(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_category_ts ON memory(category, timestamp DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags ON memory(tags)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_status ON memory(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_confidence ON memory(confidence)")
        # Table for embedding vectors
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                vector TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Strategy configuration
    # ------------------------------------------------------------------
    def set_compression_strategy(
        self, strategy: Callable[[sqlite3.Connection], None]
    ) -> None:
        """Define custom compression strategy."""

        self.compression_strategy = strategy

    def set_forget_strategy(self, strategy: Callable[[sqlite3.Connection], None]) -> None:
        """Define custom forgetting strategy."""

        self.forget_strategy = strategy

    def set_recycle_strategy(
        self, strategy: Callable[[sqlite3.Connection], None]
    ) -> None:
        """Define custom recycling strategy."""

        self.recycle_strategy = strategy

    def store(
        self, item: str, *, metadata: Optional[Mapping[str, Any]] = None
    ) -> int:
        """Persist ``item`` with normalised ``metadata`` for protocol compatibility."""

        params = dict(metadata or {})
        category = str(params.pop("category", "general"))
        tags = params.pop("tags", None)
        timestamp = params.pop("timestamp", None)
        confidence = params.pop("confidence", None)
        status = params.pop("status", "active")
        record_metadata = params.pop("metadata", None)
        if params:
            extra_meta = dict(record_metadata or {})
            extra_meta.update(params)
            record_metadata = extra_meta
        return self._insert_entry(
            category,
            item,
            tags=tags,
            timestamp=timestamp,
            confidence=confidence,
            status=status,
            metadata=record_metadata,
        )

    def add(
        self,
        category: str,
        content: str,
        *,
        tags: Optional[Sequence[str]] = None,
        timestamp: Optional[float] = None,
        confidence: Optional[float] = None,
        status: str = "active",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Store a piece of ``content`` under ``category`` (backwards compatible)."""

        return self._insert_entry(
            category,
            content,
            tags=tags,
            timestamp=timestamp,
            confidence=confidence,
            status=status,
            metadata=metadata,
        )

    def _insert_entry(
        self,
        category: str,
        content: str,
        *,
        tags: Optional[Sequence[str]] = None,
        timestamp: Optional[float] = None,
        confidence: Optional[float] = None,
        status: str = "active",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        if timestamp is None:
            timestamp = datetime.utcnow().timestamp()
        confidence_value = 1.0 if confidence is None else float(confidence)
        if confidence_value < 0.0:
            confidence_value = 0.0
        elif confidence_value > 1.0:
            confidence_value = 1.0
        status_value = status or "active"
        metadata_json = json.dumps(dict(metadata), ensure_ascii=False) if metadata else None
        tags_str = ",".join(sorted(tags)) if tags else None
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO memory (category, content, timestamp, tags, confidence, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (category, content, timestamp, tags_str, confidence_value, status_value, metadata_json),
        )
        self._pending_adds += 1
        row_id = cur.lastrowid
        if not self._in_batch:
            self.conn.commit()
            self._post_commit()
        return int(row_id)

    def retrieve(
        self, filters: Optional[Mapping[str, Any]] = None
    ) -> Iterable[Any]:
        """Retrieve stored memories using the unified protocol signature."""

        params = dict(filters or {})
        include_metadata = bool(params.pop("include_metadata", False))
        metadata_filter = params.pop("metadata_filter", None)
        if params:
            extra_filter = dict(metadata_filter or {})
            extra_filter.update(params)
            metadata_filter = extra_filter
        return self._select_entries(
            include_metadata=include_metadata,
            category=params.pop("category", None),
            tags=params.pop("tags", None),
            start_ts=params.pop("start_ts", None),
            end_ts=params.pop("end_ts", None),
            limit=params.pop("limit", None),
            newest_first=bool(params.pop("newest_first", False)),
            status=params.pop("status", None),
            exclude_status=params.pop("exclude_status", None),
            min_confidence=params.pop("min_confidence", None),
            metadata_filter=metadata_filter,
        )

    def get(
        self,
        category: Optional[str] = None,
        *,
        tags: Optional[Sequence[str]] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
        status: Optional[Sequence[str]] = None,
        exclude_status: Optional[Sequence[str]] = None,
        min_confidence: Optional[float] = None,
        include_metadata: bool = False,
    ) -> Iterable[Any]:
        """Retrieve stored memories (legacy signature)."""

        return self._select_entries(
            include_metadata=include_metadata,
            category=category,
            tags=tags,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit,
            newest_first=newest_first,
            status=status,
            exclude_status=exclude_status,
            min_confidence=min_confidence,
        )

    def _select_entries(
        self,
        *,
        include_metadata: bool = False,
        category: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
        status: Optional[Sequence[str]] = None,
        exclude_status: Optional[Sequence[str]] = None,
        min_confidence: Optional[float] = None,
        metadata_filter: Optional[Mapping[str, Any]] = None,
    ) -> Iterable[Any]:
        cur = self.conn.cursor()
        select_clause = (
            "id, category, content, timestamp, tags, confidence, status, metadata"
            if include_metadata
            else "content"
        )
        query = f"SELECT {select_clause} FROM memory"
        conditions = []
        params: list[object] = []
        if category is not None:
            conditions.append("category = ?")
            params.append(category)
        if tags:
            for tag in tags:
                conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
        if start_ts is not None:
            conditions.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts is not None:
            conditions.append("timestamp < ?")
            params.append(end_ts)
        if status:
            placeholders = ",".join("?" for _ in status)
            conditions.append(f"status IN ({placeholders})")
            params.extend(status)
        if exclude_status:
            placeholders = ",".join("?" for _ in exclude_status)
            conditions.append(f"status NOT IN ({placeholders})")
            params.extend(exclude_status)
        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(float(min_confidence))
        if metadata_filter:
            for key, value in metadata_filter.items():
                conditions.append("json_extract(metadata, ?) = ?")
                params.append(f'$."{key}"')
                params.append(value)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        if newest_first:
            query += " ORDER BY timestamp DESC"
        if limit is not None:
            limit_value = int(limit)
            if limit_value > 0:
                query += " LIMIT ?"
                params.append(limit_value)
        cur.execute(query, params)
        for row in cur:
            if include_metadata:
                tags_value = row[4].split(",") if row[4] else []
                metadata_value = json.loads(row[7]) if row[7] else None
                yield {
                    "id": row[0],
                    "category": row[1],
                    "content": row[2],
                    "timestamp": row[3],
                    "tags": tags_value,
                    "confidence": row[5],
                    "status": row[6],
                    "metadata": metadata_value or {},
                }
            else:
                yield row[0]

    def clear(self) -> None:
        """Remove all stored memory and embedding entries."""

        cur = self.conn.cursor()
        cur.execute("DELETE FROM memory")
        cur.execute("DELETE FROM embeddings")
        self.conn.commit()

    def search(
        self,
        query: str,
        *,
        limit: Optional[int] = None,
        category: Optional[str] = None,
    ) -> Iterable[dict[str, Any]]:
        """Simple LIKE-based search returning entries with metadata."""

        cur = self.conn.cursor()
        sql = (
            "SELECT id, category, content, timestamp, tags, confidence, status, metadata"
            " FROM memory WHERE content LIKE ?"
        )
        params: list[object] = [f"%{query}%"]
        if category is not None:
            sql += " AND category = ?"
            params.append(category)
        sql += " ORDER BY timestamp DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        cur.execute(sql, params)
        for row in cur.fetchall():
            tags_value = row[4].split(",") if row[4] else []
            metadata_value = json.loads(row[7]) if row[7] else None
            yield {
                "id": row[0],
                "category": row[1],
                "content": row[2],
                "timestamp": row[3],
                "tags": tags_value,
                "confidence": row[5],
                "status": row[6],
                "metadata": metadata_value or {},
            }

    def update_entry(
        self,
        entry_id: int,
        *,
        confidence: Optional[float] = None,
        status: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Update metadata for a single memory entry."""

        assignments: list[str] = []
        params: list[object] = []
        if confidence is not None:
            clamped = max(0.0, min(1.0, float(confidence)))
            assignments.append("confidence = ?")
            params.append(clamped)
        if status is not None:
            assignments.append("status = ?")
            params.append(status or "active")
        if tags is not None:
            tags_str = ",".join(sorted(tags)) if tags else None
            assignments.append("tags = ?")
            params.append(tags_str)
        if metadata is not None:
            metadata_json = json.dumps(dict(metadata), ensure_ascii=False)
            assignments.append("metadata = ?")
            params.append(metadata_json)
        if not assignments:
            return
        params.append(entry_id)
        cur = self.conn.cursor()
        cur.execute(
            f"UPDATE memory SET {', '.join(assignments)} WHERE id = ?", params
        )
        self.conn.commit()

    def mark_entries(
        self,
        entry_ids: Sequence[int],
        *,
        status: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """Bulk update ``status``/``confidence`` for the given ``entry_ids``."""

        ids = [int(entry_id) for entry_id in entry_ids]
        if not ids:
            return 0
        assignments: list[str] = []
        params: list[object] = []
        if status is not None:
            assignments.append("status = ?")
            params.append(status or "active")
        if confidence is not None:
            clamped = max(0.0, min(1.0, float(confidence)))
            assignments.append("confidence = ?")
            params.append(clamped)
        if not assignments:
            return 0
        placeholders = ",".join("?" for _ in ids)
        query = f"UPDATE memory SET {', '.join(assignments)} WHERE id IN ({placeholders})"
        cur = self.conn.cursor()
        cur.execute(query, params + ids)
        self.conn.commit()
        return cur.rowcount

    def configure_priority_forgetting(
        self,
        *,
        stale_statuses: Sequence[str] = ("stale", "deprecated"),
        confidence_threshold: Optional[float] = 0.2,
        max_removed_per_cycle: Optional[int] = None,
    ) -> None:
        """Install a forgetting strategy that prioritises stale and low-confidence entries."""

        stale_tuple = tuple(stale_statuses)
        threshold = None if confidence_threshold is None else float(confidence_threshold)
        limit_per_cycle = None if max_removed_per_cycle is None else int(max(0, max_removed_per_cycle))

        def _strategy(conn: sqlite3.Connection) -> None:
            cur = conn.cursor()
            remaining = limit_per_cycle

            def _delete_by_query(query: str, query_params: list[object]) -> int:
                cur.execute(query, query_params)
                ids = [row[0] for row in cur.fetchall()]
                if not ids:
                    return 0
                delete_placeholders = ",".join("?" for _ in ids)
                cur.execute(
                    f"DELETE FROM memory WHERE id IN ({delete_placeholders})",
                    ids,
                )
                return len(ids)

            if stale_tuple:
                params: list[object] = list(stale_tuple)
                select_query = "SELECT id FROM memory WHERE status IN ({}) ORDER BY timestamp ASC".format(
                    ",".join("?" for _ in stale_tuple)
                )
                if remaining is not None:
                    select_query += " LIMIT ?"
                    params.append(remaining)
                removed = _delete_by_query(select_query, params)
                if remaining is not None:
                    remaining = max(remaining - removed, 0)
                if remaining == 0:
                    return

            if threshold is not None:
                params = [threshold]
                select_query = "SELECT id FROM memory WHERE confidence < ?"
                if stale_tuple:
                    select_query += " AND status NOT IN ({})".format(
                        ",".join("?" for _ in stale_tuple)
                    )
                    params.extend(stale_tuple)
                select_query += " ORDER BY confidence ASC, timestamp ASC"
                if remaining is not None:
                    select_query += " LIMIT ?"
                    params.append(remaining)
                removed = _delete_by_query(select_query, params)
                if remaining is not None:
                    remaining = max(remaining - removed, 0)
                if remaining == 0:
                    return

            if self.max_entries is not None:
                cur.execute("SELECT COUNT(*) FROM memory")
                (count,) = cur.fetchone()
                if count <= self.max_entries:
                    return
                excess = count - self.max_entries
                if remaining is not None:
                    excess = min(excess, remaining)
                    if excess <= 0:
                        return
                cur.execute(
                    "SELECT id FROM memory ORDER BY confidence ASC, timestamp ASC LIMIT ?",
                    (excess,),
                )
                fallback_ids = [row[0] for row in cur.fetchall()]
                if fallback_ids:
                    placeholders = ",".join("?" for _ in fallback_ids)
                    cur.execute(
                        f"DELETE FROM memory WHERE id IN ({placeholders})",
                        fallback_ids,
                    )

        self.set_forget_strategy(_strategy)

    # ------------------------------------------------------------------
    # Embedding specific helpers
    # ------------------------------------------------------------------
    def add_embedding(
        self, key: str, vector: Sequence[float], metadata: Optional[dict] = None
    ) -> None:
        """Store an embedding vector and optional metadata."""

        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO embeddings (key, vector, metadata) VALUES (?, ?, ?)",
            (
                key,
                json.dumps(list(map(float, vector))),
                json.dumps(metadata) if metadata is not None else None,
            ),
        )
        self.conn.commit()

    def get_embedding(self, key: str) -> Optional[tuple[list[float], Optional[dict]]]:
        """Retrieve an embedding and metadata for ``key``."""

        cur = self.conn.cursor()
        cur.execute("SELECT vector, metadata FROM embeddings WHERE key = ?", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        vector = json.loads(row[0])
        metadata = json.loads(row[1]) if row[1] is not None else None
        return vector, metadata

    def similarity_search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        category: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> list[dict[str, Any]]:
        """Return embeddings most similar to ``query_vector``.

        The method performs cosine similarity over stored embedding vectors and
        can filter results by the associated memory category or tags when that
        metadata is available. Each hit includes the embedding ``key`` plus any
        linked memory metadata to allow callers to join back to the ``memory``
        table.
        """

        normalized_query = self._normalize_vector(query_vector)
        if normalized_query is None:
            return []

        tag_filter = set(tags or [])
        cur = self.conn.cursor()
        cur.execute("SELECT key, vector, metadata FROM embeddings")

        results: list[dict[str, Any]] = []
        for key, vec_json, meta_json in cur.fetchall():
            try:
                stored_vec = json.loads(vec_json)
            except json.JSONDecodeError:
                continue
            normalized_stored = self._normalize_vector(stored_vec)
            if normalized_stored is None or len(normalized_stored) != len(normalized_query):
                continue

            meta = json.loads(meta_json) if meta_json is not None else {}
            memory_id = meta.get("memory_id") if isinstance(meta, dict) else None
            if memory_id is None:
                try:
                    memory_id = int(key)
                except (TypeError, ValueError):
                    memory_id = None

            memory_row = self._load_memory_row(memory_id) if memory_id is not None else None
            if category is not None:
                candidate_category = None
                if memory_row is not None:
                    candidate_category = memory_row.get("category")
                elif isinstance(meta, dict):
                    candidate_category = meta.get("category")
                if candidate_category != category:
                    continue
            if tag_filter:
                candidate_tags: set[str] = set()
                if memory_row is not None:
                    candidate_tags = set(memory_row.get("tags") or [])
                elif isinstance(meta, dict):
                    candidate_tags = set(meta.get("tags") or [])
                if not tag_filter.issubset(candidate_tags):
                    continue

            similarity = float(
                sum(q * v for q, v in zip(normalized_query, normalized_stored))
            )
            results.append(
                {
                    "key": key,
                    "score": similarity,
                    "vector": stored_vec,
                    "metadata": meta if isinstance(meta, dict) else {},
                    "memory_id": int(memory_id) if memory_id is not None else None,
                    "memory": memory_row,
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        limit = top_k if top_k is not None else len(results)
        return results[: limit if limit is not None else len(results)]

    def iter_embeddings(self) -> Iterable[tuple[str, list[float], Optional[dict]]]:
        """Iterate over all stored embeddings."""

        cur = self.conn.cursor()
        cur.execute("SELECT key, vector, metadata FROM embeddings")
        for key, vec, meta in cur.fetchall():
            yield key, json.loads(vec), json.loads(meta) if meta is not None else None

    def _normalize_vector(self, vector: Sequence[float]) -> Optional[list[float]]:
        floats: list[float] = []
        try:
            floats = [float(v) for v in vector]
        except Exception:
            return None
        norm = math.sqrt(sum(v * v for v in floats))
        if norm == 0.0:
            return None
        return [v / norm for v in floats]

    def _load_memory_row(self, entry_id: int) -> Optional[dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, category, content, timestamp, tags, confidence, status, metadata "
            "FROM memory WHERE id = ?",
            (int(entry_id),),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "category": row[1],
            "content": row[2],
            "timestamp": row[3],
            "tags": row[4].split(",") if row[4] else [],
            "confidence": row[5],
            "status": row[6],
            "metadata": json.loads(row[7]) if row[7] else {},
        }

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Transaction helpers and maintenance
    # ------------------------------------------------------------------
    @contextmanager
    def batch(self):
        """Group multiple ``add`` calls into a single transaction."""

        self._in_batch = True
        try:
            yield self
            self.conn.commit()
            self._post_commit()
        except Exception:  # pragma: no cover - rollback path
            self.conn.rollback()
            self._pending_adds = 0
            raise
        finally:
            self._in_batch = False

    def _post_commit(self) -> None:
        """Run housekeeping tasks after committing new entries."""

        self._total_adds += self._pending_adds
        self._pending_adds = 0

        if self.max_entries is not None:
            self._trim_to_limit()

        if (
            self.forget_interval
            and self.forget_strategy
            and self._total_adds % self.forget_interval == 0
        ):
            self.forget()

        if (
            self.recycle_interval
            and self.recycle_strategy
            and self._total_adds % self.recycle_interval == 0
        ):
            self.recycle()

        if self.vacuum_interval and self._total_adds % self.vacuum_interval == 0:
            self.compress()

    def compress(
        self, strategy: Optional[Callable[[sqlite3.Connection], None]] = None
    ) -> None:
        """Apply compression strategy or fall back to ``VACUUM``."""

        strategy = strategy or self.compression_strategy
        if strategy is not None:
            strategy(self.conn)
            self.conn.commit()
        else:
            self.vacuum()

    def forget(
        self,
        *,
        before_ts: Optional[float] = None,
        strategy: Optional[Callable[[sqlite3.Connection], None]] = None,
    ) -> None:
        """Forget memories using ``strategy`` or timestamp filtering."""

        strategy = strategy or self.forget_strategy
        if strategy is not None:
            strategy(self.conn)
            self.conn.commit()
            return
        if before_ts is None:
            return
        cur = self.conn.cursor()
        cur.execute("DELETE FROM memory WHERE timestamp < ?", (before_ts,))
        self.conn.commit()

    def recycle(
        self, strategy: Optional[Callable[[sqlite3.Connection], None]] = None
    ) -> None:
        """Recycle memory entries using ``strategy`` or ``VACUUM``."""

        strategy = strategy or self.recycle_strategy
        if strategy is not None:
            strategy(self.conn)
            self.conn.commit()
        else:
            self.vacuum()

    def _trim_to_limit(self) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM memory")
        (count,) = cur.fetchone()
        if count <= self.max_entries:
            return
        remove = count - self.max_entries
        cur.execute(
            "DELETE FROM memory WHERE id IN ("
            "SELECT id FROM memory ORDER BY timestamp ASC LIMIT ?"
            ")",
            (remove,),
        )
        self.conn.commit()

    def vacuum(self) -> None:
        """Reclaim unused space in the database."""

        cur = self.conn.cursor()
        cur.execute("VACUUM")
        self.conn.commit()

