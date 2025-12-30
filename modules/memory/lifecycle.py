
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .task_memory import ExperiencePayload, format_experience_payload
from .vector_store import VectorMemoryStore


logger = logging.getLogger(__name__)


class MemoryLifecycleManager:
    """Manage memory stages (short-term, working, long-term) with persistence."""

    def __init__(
        self,
        vector_store: VectorMemoryStore,
        *,
        short_term_limit: int = 25,
        working_memory_limit: int = 50,
        short_term_ttl: float = 900.0,
        working_memory_ttl: float = 7 * 24 * 3600.0,
        promotion_usage_threshold: int = 3,
        promotion_importance_threshold: float = 0.7,
        long_term_importance_threshold: float = 0.85,
        consolidation_interval: float = 1800.0,
        text_log_path: Path | str = Path("memory") / "long_term" / "log.jsonl",
        symbol_log_path: Path | str = Path("memory") / "long_term" / "symbols.jsonl",
        summarizer: Optional[Callable[[List[ExperiencePayload]], Dict[str, Any]]] = None,
        summary_batch_size: int = 5,
        summary_rate_limit: float = 900.0,
    ) -> None:
        self._vector_store = vector_store
        self._short_term_limit = max(1, short_term_limit)
        self._working_limit = max(1, working_memory_limit)
        self._short_term_ttl = max(60.0, short_term_ttl)
        self._working_ttl = max(300.0, working_memory_ttl)
        self._usage_threshold = max(1, promotion_usage_threshold)
        self._importance_threshold = promotion_importance_threshold
        self._long_term_importance = long_term_importance_threshold
        self._consolidation_interval = consolidation_interval
        self._last_consolidation = 0.0

        self._short_term: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._working: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._long_term_index: Dict[str, Dict[str, Any]] = {}

        self._text_log = Path(text_log_path)
        self._symbol_log = Path(symbol_log_path)
        self._text_log.parent.mkdir(parents=True, exist_ok=True)
        self._symbol_log.parent.mkdir(parents=True, exist_ok=True)
        if not self._text_log.exists():
            self._text_log.write_text("", encoding="utf-8")
        if not self._symbol_log.exists():
            self._symbol_log.write_text("", encoding="utf-8")

        self._summary_batch_size = max(1, summary_batch_size)
        self._summary_rate_limit = max(60.0, summary_rate_limit)
        self._summary_queue: List[str] = []
        self._last_summary_time = 0.0
        self._summarizer = summarizer or self._default_summarizer

    # ------------------------------------------------------------------ ingestion
    def ingest_interaction(
        self,
        payload: ExperiencePayload,
        *,
        importance: float = 0.5,
        symbols: Optional[Dict[str, Any]] = None,
        auto_promote: bool = False,
    ) -> str:
        now = time.time()
        entry_id = str(uuid.uuid4())
        metadata = dict(payload.metadata)
        payload_copy = ExperiencePayload(
            task_id=payload.task_id,
            summary=payload.summary,
            messages=list(payload.messages),
            metadata=metadata,
        )
        memory_metadata = {**metadata, "memory_stage": "short_term"}
        payload_copy.metadata = memory_metadata
        memory = {
            "id": entry_id,
            "payload": payload_copy,
            "text": format_experience_payload(payload),
            "metadata": memory_metadata,
            "symbols": dict(symbols or {}),
            "importance": float(importance),
            "usage": 0,
            "stage": "short_term",
            "vector_id": None,
            "created_at": now,
            "last_access": now,
        }
        self._add_short_term(memory)
        if auto_promote or importance >= self._importance_threshold:
            self._promote_to_working(memory["id"])
            if importance >= self._long_term_importance:
                self._promote_to_long_term(memory["id"], memory)
        self._summary_queue.append(entry_id)
        return entry_id

    # ------------------------------------------------------------------ recall & access
    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_short_term: bool = True,
        include_working: bool = True,
        include_long_term: bool = True,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        lower_query = query.lower().strip()
        if include_short_term:
            for memory in self._short_term.values():
                score = self._match_score(memory, lower_query)
                if score > 0.0:
                    results.append(self._build_result(memory, score))
        if include_working:
            for memory in self._working.values():
                score = self._match_score(memory, lower_query)
                if score > 0.0:
                    results.append(self._build_result(memory, score))
        if include_long_term:
            vector_hits = self._vector_store.query(query, top_k=top_k)
            for record in vector_hits:
                self._vector_store.record_access(record.id)
                result = {
                    "stage": "long_term",
                    "score": float(record.score),
                    "record": record,
                }
                results.append(result)
        results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return results[:top_k]

    def mark_usage(self, memory_id: str) -> None:
        memory = self._short_term.get(memory_id) or self._working.get(memory_id)
        now = time.time()
        if memory is not None:
            memory["usage"] += 1
            memory["last_access"] = now

    # ------------------------------------------------------------------ lifecycle & maintenance
    def consolidate(self) -> Dict[str, List[str]]:
        now = time.time()
        promoted: List[str] = []
        archived: List[str] = []
        for memory_id, memory in list(self._short_term.items()):
            age = now - memory["created_at"]
            if memory["usage"] >= self._usage_threshold or memory["importance"] >= self._importance_threshold:
                self._promote_to_working(memory_id)
                promoted.append(memory_id)
                continue
            if age >= self._short_term_ttl:
                self._short_term.pop(memory_id, None)
                archived.append(memory_id)
        for memory_id, memory in list(self._working.items()):
            age = now - memory["last_access"]
            if memory["importance"] >= self._long_term_importance or memory["usage"] >= (self._usage_threshold * 2):
                self._promote_to_long_term(memory_id, memory)
                promoted.append(memory_id)
            elif age >= self._working_ttl:
                if memory["usage"] > 0 or memory["importance"] >= 0.3:
                    self._promote_to_long_term(memory_id, memory)
                    promoted.append(memory_id)
                else:
                    self._working.pop(memory_id, None)
                    archived.append(memory_id)
        if now - self._last_consolidation >= self._consolidation_interval:
            self._vector_store.run_heat_maintenance()
            self._last_consolidation = now
        summaries = self._run_summary_generation(now)
        if promoted or archived:
            self._vector_store._persist()  # type: ignore[attr-defined]
        if summaries:
            self._vector_store._persist()  # type: ignore[attr-defined]
        return {"promoted": promoted, "archived": archived, "summaries": summaries}

    def run_consistency_check(self, *, apply_changes: bool = True) -> Dict[str, List[tuple[str, str]]]:
        duplicates: List[tuple[str, str]] = []
        conflicts: List[tuple[str, str]] = []
        seen_text: Dict[str, str] = {}
        truth_index: Dict[str, Dict[str, Any]] = {}
        for record in self._vector_store.iter_records():
            record_id = record.get("id")
            text = (record.get("text") or "").strip().lower()
            metadata = dict(record.get("metadata", {}))
            if text:
                if text in seen_text:
                    duplicates.append((record_id, seen_text[text]))
                    if apply_changes:
                        self._vector_store.update_metadata(record_id, {"duplicate_of": seen_text[text]})
                else:
                    seen_text[text] = record_id
            statement = metadata.get("statement")
            if statement is not None and "truth" in metadata:
                key = str(statement).strip().lower()
                truth = metadata.get("truth")
                if key in truth_index and truth_index[key]["truth"] != truth:
                    conflicts.append((record_id, truth_index[key]["id"]))
                    if apply_changes:
                        self._vector_store.update_metadata(record_id, {"conflict_with": truth_index[key]["id"], "conflict_flag": True})
                        self._vector_store.update_metadata(truth_index[key]["id"], {"conflict_with": record_id, "conflict_flag": True})
                else:
                    truth_index[key] = {"id": record_id, "truth": truth}
        if apply_changes and (duplicates or conflicts):
            self._vector_store._persist()  # type: ignore[attr-defined]
        return {"duplicates": duplicates, "conflicts": conflicts}

    def run_maintenance(self, *, max_hot: Optional[int] = None, idle_seconds: Optional[float] = None) -> Dict[str, Any]:
        consolidation = self.consolidate()
        heat = self._vector_store.run_heat_maintenance(max_hot=max_hot, idle_seconds=idle_seconds)
        return {"consolidation": consolidation, "heat": heat}

    def get_short_term(self) -> List[Dict[str, Any]]:
        return [self._export_memory(memory) for memory in self._short_term.values()]

    def get_working_memory(self) -> List[Dict[str, Any]]:
        return [self._export_memory(memory) for memory in self._working.values()]

    # ------------------------------------------------------------------ internal helpers
    def _add_short_term(self, memory: Dict[str, Any]) -> None:
        self._short_term[memory["id"]] = memory
        while len(self._short_term) > self._short_term_limit:
            self._short_term.popitem(last=False)

    def _promote_to_working(self, memory_id: str) -> None:
        memory = self._short_term.pop(memory_id, None)
        if memory is None:
            memory = self._working.get(memory_id)
        if memory is None:
            return
        memory["stage"] = "working"
        memory["metadata"]["memory_stage"] = "working"
        self._working[memory_id] = memory
        if len(self._working) > self._working_limit:
            oldest_id, oldest = min(self._working.items(), key=lambda item: item[1]["last_access"])
            self._working.pop(oldest_id, None)
            if oldest["importance"] >= self._importance_threshold or oldest["usage"] >= self._usage_threshold:
                self._promote_to_long_term(oldest_id, oldest)

    def _promote_to_long_term(self, memory_id: str, memory: Optional[Dict[str, Any]] = None) -> None:
        if memory is None:
            memory = self._short_term.pop(memory_id, None) or self._working.pop(memory_id, None)
        else:
            self._short_term.pop(memory_id, None)
            self._working.pop(memory_id, None)
        if memory is None:
            return
        text = memory["text"]
        metadata = dict(memory["metadata"])
        metadata.update(
            {
                "memory_stage": "long_term",
                "importance": memory["importance"],
                "usage": memory["usage"],
                "created_at": memory["created_at"],
                "last_access": memory["last_access"],
                "source": metadata.get("source", "memory_lifecycle"),
            }
        )
        record_id = self._vector_store.add_text(text, metadata=metadata, record_id=memory_id)
        self._vector_store.update_metadata(record_id, {})
        memory["metadata"].update(metadata)
        self._vector_store._persist()  # type: ignore[attr-defined]
        memory["stage"] = "long_term"
        memory["vector_id"] = record_id
        self._long_term_index[record_id] = memory
        self._write_text_log(record_id, text, metadata)
        if memory.get("symbols"):
            self._write_symbol_log(record_id, memory["symbols"])

    def _match_score(self, memory: Dict[str, Any], query_lower: str) -> float:
        if not query_lower:
            return 1.0
        text_lower = memory["text"].lower()
        if query_lower in text_lower:
            return 1.0
        summary = memory["payload"].summary.lower()
        if query_lower in summary:
            return 0.8
        return 0.0

    def _build_result(self, memory: Dict[str, Any], score: float) -> Dict[str, Any]:
        memory["usage"] += 1
        memory["last_access"] = time.time()
        return {
            "stage": memory["stage"],
            "score": score,
            "memory": self._export_memory(memory),
        }

    def _export_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        payload = memory["payload"]
        return {
            "id": memory["id"],
            "stage": memory["stage"],
            "summary": payload.summary,
            "task_id": payload.task_id,
            "messages": payload.messages,
            "metadata": dict(memory["metadata"]),
            "importance": memory["importance"],
            "usage": memory["usage"],
            "last_access": memory["last_access"],
            "vector_id": memory.get("vector_id"),
        }

    def run_summary_consolidation(self, *, force: bool = False, timestamp: Optional[float] = None) -> List[str]:
        now = time.time() if timestamp is None else float(timestamp)
        summaries = self._run_summary_generation(now, force=force)
        if summaries:
            self._vector_store._persist()  # type: ignore[attr-defined]
        return summaries

    def _run_summary_generation(self, now: float, *, force: bool = False) -> List[str]:
        self._prune_summary_queue()
        if not self._summary_queue:
            return []

        since_last = now - self._last_summary_time
        if not force and since_last < self._summary_rate_limit:
            if len(self._summary_queue) < self._summary_batch_size:
                return []

        batch_size = min(len(self._summary_queue), self._summary_batch_size)
        batch_ids = self._summary_queue[:batch_size]
        batch_memories: List[Dict[str, Any]] = []
        processed_ids: List[str] = []
        for memory_id in batch_ids:
            memory = self._lookup_memory(memory_id)
            processed_ids.append(memory_id)
            if memory is None:
                continue
            batch_memories.append(memory)

        if not batch_memories:
            self._summary_queue = self._summary_queue[len(processed_ids) :]
            return []

        try:
            summary_result = self._summarizer([m["payload"] for m in batch_memories])
        except Exception:
            logger.exception("Failed to summarise memory batch")
            self._summary_queue = self._summary_queue[len(processed_ids) :]
            self._last_summary_time = now
            return []

        summary_text = str(summary_result.get("text", "")).strip()
        if not summary_text:
            self._summary_queue = self._summary_queue[len(processed_ids) :]
            self._last_summary_time = now
            return []

        key_facts_raw = summary_result.get("key_facts", []) or []
        key_facts = [str(fact).strip() for fact in key_facts_raw if str(fact).strip()]
        key_facts = key_facts[:50]

        source_ids = [memory["id"] for memory in batch_memories]
        summary_metadata = dict(summary_result.get("metadata", {}))
        summary_metadata.setdefault("memory_stage", "long_term")
        summary_metadata.setdefault("source", "memory_consolidation")
        summary_metadata.setdefault("type", "episodic_summary")
        summary_metadata["source_memory_ids"] = source_ids
        summary_metadata["batch_size"] = len(batch_memories)
        summary_metadata["key_facts"] = key_facts
        summary_metadata["created_at"] = now
        summary_metadata["last_access"] = now
        summary_metadata.setdefault("importance", max(self._importance_threshold, 0.6))
        summary_metadata.setdefault("usage", 0)
        summary_metadata.setdefault("weight", 1.0)
        summary_metadata["importance"] = float(summary_metadata.get("importance", 0.6))
        summary_metadata["usage"] = int(summary_metadata.get("usage", 0))
        summary_metadata["weight"] = float(summary_metadata.get("weight", 1.0))

        record_id = self._vector_store.add_text(summary_text, metadata=summary_metadata)

        summary_payload = ExperiencePayload(
            task_id=summary_metadata.get("task_id", "memory_summary"),
            summary=summary_text,
            messages=[],
            metadata=dict(summary_metadata),
        )
        raw_symbols = summary_result.get("symbols")
        if isinstance(raw_symbols, dict):
            symbols = dict(raw_symbols)
        elif raw_symbols:
            symbols = {"items": list(raw_symbols)}
        else:
            symbols = {}
        summary_entry = {
            "id": record_id,
            "payload": summary_payload,
            "text": summary_text,
            "metadata": dict(summary_metadata),
            "symbols": symbols,
            "importance": float(summary_metadata.get("importance", 1.0)),
            "usage": 0,
            "stage": "long_term",
            "vector_id": record_id,
            "created_at": now,
            "last_access": now,
        }
        self._long_term_index[record_id] = summary_entry

        self._write_text_log(record_id, summary_text, summary_metadata)
        if summary_entry["symbols"]:
            self._write_symbol_log(record_id, summary_entry["symbols"])

        for memory in batch_memories:
            metadata = memory.setdefault("metadata", {})
            consolidated = metadata.setdefault("consolidated_summaries", [])
            if record_id not in consolidated:
                consolidated.append(record_id)
            memory["last_access"] = now
            vector_id = memory.get("vector_id")
            if vector_id:
                self._vector_store.update_metadata(vector_id, {"consolidated_summaries": consolidated, "last_access": now})

        self._summary_queue = self._summary_queue[len(processed_ids) :]
        self._last_summary_time = now
        return [record_id]

    def _prune_summary_queue(self) -> None:
        if not self._summary_queue:
            return
        filtered: List[str] = []
        seen: set[str] = set()
        for memory_id in self._summary_queue:
            if memory_id in seen:
                continue
            seen.add(memory_id)
            memory = self._lookup_memory(memory_id)
            if memory is None:
                continue
            metadata = memory.get("metadata", {})
            if metadata.get("consolidated_summaries"):
                continue
            filtered.append(memory_id)
        self._summary_queue = filtered

    def _lookup_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        if memory_id in self._short_term:
            return self._short_term[memory_id]
        if memory_id in self._working:
            return self._working[memory_id]
        if memory_id in self._long_term_index:
            return self._long_term_index[memory_id]
        return None

    def _default_summarizer(self, payloads: List[ExperiencePayload]) -> Dict[str, Any]:
        key_facts: List[str] = []
        summary_lines: List[str] = []
        for payload in payloads:
            text = payload.summary.strip() if payload.summary else ""
            if not text:
                text = format_experience_payload(payload)
            text = text.strip()
            if not text:
                continue
            summary_lines.append(text)
            if text not in key_facts:
                key_facts.append(text)
        summary_text = "\n".join(summary_lines).strip()
        if not summary_text:
            summary_text = "Aggregated experience lacked textual content."
        symbols = {"key_facts": key_facts[:10]} if key_facts else {}
        return {
            "text": summary_text,
            "metadata": {"summary_strategy": "concatenate"},
            "key_facts": key_facts,
            "symbols": symbols,
        }

    def _write_text_log(self, record_id: str, text: str, metadata: Dict[str, Any]) -> None:
        entry = {
            "id": record_id,
            "timestamp": time.time(),
            "metadata": {**metadata, "memory_stage": "short_term"},
            "text": text,
        }
        with self._text_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _write_symbol_log(self, record_id: str, symbols: Dict[str, Any]) -> None:
        entry = {
            "id": record_id,
            "timestamp": time.time(),
            "symbols": symbols,
        }
        with self._symbol_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


__all__ = ["MemoryLifecycleManager"]