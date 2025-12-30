"""Utilities for recording and replaying agent experiences."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Optional


@dataclass(slots=True)
class ExperienceRecord:
    """Minimal snapshot of an executed command and its outcome."""

    timestamp: str
    task_id: str
    cycle: int
    command_name: str
    command_args: dict
    result_status: str
    result_summary: str
    metadata: dict | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ExperienceRecord":
        return cls(
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            task_id=data.get("task_id", ""),
            cycle=int(data.get("cycle", 0)),
            command_name=data.get("command_name", ""),
            command_args=data.get("command_args", {}),
            result_status=data.get("result_status", "unknown"),
            result_summary=data.get("result_summary", ""),
            metadata=data.get("metadata"),
        )


class ExperienceLogStore(Iterable[ExperienceRecord]):
    """Append-only JSONL log that can be replayed as experience memory."""

    def __init__(
        self,
        log_path: Path,
        max_bytes: Optional[int] = None,
    ) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._max_bytes = max_bytes

    def append(self, record: ExperienceRecord) -> None:
        payload = json.dumps(asdict(record), ensure_ascii=False)
        line = payload + "\n"
        encoded = line.encode("utf-8")
        with self._lock:
            if self._max_bytes is not None and self._path.exists():
                current_size = self._path.stat().st_size
                if current_size + len(encoded) > self._max_bytes:
                    self._rotate()
            with self._path.open("ab") as f:
                f.write(encoded)

    def _rotate(self) -> None:
        backup_path = self._path.with_suffix(".old.jsonl")
        if backup_path.exists():
            backup_path.unlink()
        self._path.replace(backup_path)

    def __iter__(self) -> Iterator[ExperienceRecord]:
        if not self._path.exists():
            return iter(())
        with self._path.open("r", encoding="utf-8") as f:
            records = [
                ExperienceRecord.from_dict(json.loads(line))
                for line in f
                if line.strip()
            ]
        return iter(records)


class ExperienceRecorder:
    """High-level helper to emit ExperienceRecord entries."""

    def __init__(
        self,
        store: ExperienceLogStore,
        max_summary_chars: int = 4096,
    ) -> None:
        self._store = store
        self._max_summary_chars = max_summary_chars

    def record(
        self,
        *,
        task_id: str,
        cycle: int,
        command_name: str,
        command_args: dict,
        result_status: str,
        result_summary: str,
        metadata: dict | None = None,
    ) -> None:
        summary = result_summary
        if len(summary) > self._max_summary_chars:
            summary = summary[: self._max_summary_chars] + "…"
        record = ExperienceRecord(
            timestamp=datetime.utcnow().isoformat(),
            task_id=task_id,
            cycle=cycle,
            command_name=command_name,
            command_args=command_args,
            result_status=result_status,
            result_summary=summary,
            metadata=metadata or {},
        )
        self._store.append(record)
