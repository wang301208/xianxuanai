"""Time-series storage for aggregating event bus data."""

from __future__ import annotations

import json
import sqlite3
import time
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from events import EventBus


class TimeSeriesStorage:
    """Persist events from the global event bus into SQLite."""

    def __init__(self, db_path: Path | str = "monitoring.db") -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()
        self._subscriptions: List[Callable[[], None]] = []

    def _init_db(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    ts REAL,
                    topic TEXT,
                    data TEXT
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_topic_ts ON events(topic, ts)"
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # event ingestion
    # ------------------------------------------------------------------
    def store(self, topic: str, event: Dict[str, Any]) -> None:
        """Store *event* published on *topic*.

        In high-concurrency scenarios consider batching or queueing events
        before calling this method to reduce lock contention.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO events (ts, topic, data) VALUES (?, ?, ?)",
                (time.time(), topic, json.dumps(event)),
            )
            self._conn.commit()

    def subscribe_to(self, bus: EventBus, topics: Iterable[str]) -> None:
        """Subscribe to *topics* on *bus* and persist events."""
        for topic in topics:
            async def _store(event: Dict[str, Any], t: str = topic) -> None:
                self.store(t, event)

            cancel = bus.subscribe(topic, _store)
            self._subscriptions.append(cancel)

    # ------------------------------------------------------------------
    # event retrieval
    # ------------------------------------------------------------------
    def events(
        self,
        topic: str | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
        limit: int | None = None,
    ) -> list[Dict[str, Any]]:
        """Return stored events.

        Parameters
        ----------
        topic:
            Optional topic to filter events by.
        start_ts, end_ts:
            Optional timestamp range. ``start_ts`` is inclusive, ``end_ts`` is
            exclusive if provided.
        limit:
            Optional maximum number of events to return.
        """
        with self._lock:
            cur = self._conn.cursor()
            query = "SELECT data FROM events"
            clauses: list[str] = []
            params: list[Any] = []

            if topic is not None:
                clauses.append("topic = ?")
                params.append(topic)

            if start_ts is not None:
                clauses.append("ts >= ?")
                params.append(start_ts)

            if end_ts is not None:
                clauses.append("ts < ?")
                params.append(end_ts)

            if clauses:
                query += " WHERE " + " AND ".join(clauses)

            query += " ORDER BY ts"

            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows]

    # ------------------------------------------------------------------
    # aggregations
    # ------------------------------------------------------------------
    def success_rate(self) -> float:
        """Return overall success rate from stored events."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT AVG(CASE
                               WHEN json_extract(data, '$.status') = 'success' THEN 1.0
                               WHEN json_extract(data, '$.status') IS NULL THEN NULL
                               ELSE 0.0
                           END)
                FROM events
                WHERE topic = 'task'
                """
            )
            row = cur.fetchone()
            task_rate = row[0]

            if task_rate is None:
                cur.execute(
                    """
                    SELECT AVG(CASE
                                   WHEN json_extract(data, '$.status') = 'success' THEN 1.0
                                   WHEN json_extract(data, '$.status') IS NULL THEN NULL
                                   ELSE 0.0
                               END)
                    FROM events
                    WHERE topic = 'prediction'
                    """
                )
                row = cur.fetchone()
                return float(row[0] or 0.0)

        return float(task_rate or 0.0)

    def bottlenecks(self) -> Dict[str, int]:
        """Return counts of failed events grouped by stage."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT COALESCE(json_extract(data, '$.stage'), 'unknown') AS stage,
                       COUNT(*)
                FROM events
                WHERE json_extract(data, '$.status') != 'success'
                GROUP BY stage
                """
            )
            rows = cur.fetchall()
        return {str(stage): count for stage, count in rows}

    def blueprint_versions(self) -> Dict[str, int]:
        """Return counts of events grouped by blueprint version."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT COALESCE(json_extract(data, '$.blueprint_version'), 'unknown') AS ver,
                       COUNT(*)
                FROM events
                GROUP BY ver
                """
            )
            rows = cur.fetchall()
        return {str(ver): count for ver, count in rows}

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        for cancel in self._subscriptions:
            cancel()
        with self._lock:
            self._conn.close()
