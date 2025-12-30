from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Iterable, List, Optional

from .vector_store import VectorMemoryStore, VectorRecord


def format_experience_payload(payload: "ExperiencePayload") -> str:
    chunks: list[str] = []
    if payload.summary:
        chunks.append(payload.summary.strip())
    for message in payload.messages:
        role = message.get("role", "agent")
        content = message.get("content", "")
        chunks.append(f"[{role}] {content}".strip())
    text = "\n".join(chunks).strip()
    return text or payload.task_id


@dataclass
class ExperiencePayload:
    task_id: str
    summary: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskMemoryManager:
    """Persist AutoGPT conversations and retrieve relevant experiences."""

    def __init__(self, store: VectorMemoryStore):
        self.store = store

    def store_experience(self, payload: ExperiencePayload) -> str:
        text = format_experience_payload(payload)
        metadata = {
            "task_id": payload.task_id,
            "summary": payload.summary,
            **payload.metadata,
            "source": payload.metadata.get("source", "task_memory"),
        }
        return self.store.add_text(text, metadata=metadata)

    def recall(self, task_description: str, top_k: int = 5) -> List[VectorRecord]:
        records = self.store.query(task_description, top_k=top_k)
        for record in records:
            updated = self.store.record_access(record.id)
            if updated is not None:
                record.metadata.update(updated)
        return records

    def bulk_ingest(self, payloads: Iterable[ExperiencePayload]) -> List[str]:
        ids = []
        for payload in payloads:
            ids.append(self.store_experience(payload))
        return ids

    def _format_payload(self, payload: ExperiencePayload) -> str:
        return format_experience_payload(payload)
