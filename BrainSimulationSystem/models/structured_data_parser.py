"""
Structured data ingestion utilities for the perception stack.

The :class:`StructuredDataParser` normalises heterogeneous structured sources
such as CSV tables, JSON payloads, or ad-hoc row dictionaries into fact triples
that can be inserted into the lightweight :class:`~BrainSimulationSystem.models.
knowledge_graph.KnowledgeGraph`.  When possible it also derives compact
embeddings describing numeric and textual statistics so downstream multimodal
modules can align structured knowledge with other sensory channels.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from BrainSimulationSystem.models.knowledge_graph import KnowledgeGraph, Triple


_LOGGER = logging.getLogger(__name__)


@dataclass
class StructuredDataBatch:
    """Container returned by :class:`StructuredDataParser.parse`."""

    triples: List[Triple] = field(default_factory=list)
    records: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "triples": list(self.triples),
            "records": [dict(record) for record in self.records],
            "metadata": dict(self.metadata),
            "embeddings": {key: value.tolist() for key, value in self.embeddings.items()},
        }


class StructuredDataParser:
    """Normalise structured payloads into triples and embeddings."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.default_relation = str(self.config.get("default_relation", "has_attribute"))
        self.subject_field = self.config.get("subject_field") or "id"
        self.object_field = self.config.get("object_field")
        self.separator = str(self.config.get("separator", ","))
        self.max_columns = int(self.config.get("max_columns", 64))
        self.return_records = bool(self.config.get("return_records", True))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def parse(
        self,
        payload: Any,
        *,
        source: Optional[Dict[str, Any]] = None,
    ) -> StructuredDataBatch:
        """Return triples + embeddings for ``payload``."""

        source = dict(source or {})
        records = self._coerce_records(payload, source)
        triples = self._records_to_triples(records, source)
        embeddings = self._compute_embeddings(records, triples)
        metadata = {
            "rows": len(records),
            "columns": sorted({key for record in records for key in record.keys()}),
            "source": source.get("name") or source.get("path") or source.get("format"),
        }
        return StructuredDataBatch(
            triples=triples,
            records=records if self.return_records else [],
            metadata=metadata,
            embeddings=embeddings,
        )

    def ingest(
        self,
        graph: KnowledgeGraph,
        payload: Any,
        *,
        source: Optional[Dict[str, Any]] = None,
        provenance: Optional[str] = None,
    ) -> StructuredDataBatch:
        """Parse ``payload`` and upsert results into ``graph``."""

        batch = self.parse(payload, source=source)
        if not graph or not batch.triples:
            return batch

        default_meta = {
            "provenance": provenance or source.get("name") if isinstance(source, dict) else provenance,
            "columns": batch.metadata.get("columns"),
            "rows": batch.metadata.get("rows"),
        }
        if batch.embeddings:
            default_meta["embedding"] = next(iter(batch.embeddings.values())).tolist()

        graph.upsert_triples(batch.triples, default_metadata=default_meta)
        return batch

    # ------------------------------------------------------------------ #
    # Record coercion helpers
    # ------------------------------------------------------------------ #
    def _coerce_records(
        self,
        payload: Any,
        source: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if payload is None:
            return []

        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict):
                return [self._sanitize_record(record) for record in payload if isinstance(record, dict)]
            if payload and isinstance(payload[0], (list, tuple)):
                return self._rows_to_records(payload, source)

        if isinstance(payload, dict):
            if all(isinstance(value, (list, tuple)) for value in payload.values()):
                return self._columns_to_records(payload)
            return [self._sanitize_record(payload)]

        if isinstance(payload, (pathlib.Path, str)):
            text = self._read_text(payload, source)
            if text is not None:
                return self._text_to_records(text, source)

        if isinstance(payload, io.IOBase):
            text = payload.read()
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="ignore")
            return self._text_to_records(str(text or ""), source)

        if isinstance(payload, numpy_like := getattr(payload, "tolist", None)):
            try:
                sequence = numpy_like()
                if isinstance(sequence, list):
                    return self._coerce_records(sequence, source)
            except TypeError:
                pass

        return []

    def _rows_to_records(
        self,
        rows: Sequence[Sequence[Any]],
        source: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []
        header = source.get("header")
        if not header:
            header = rows[0]
            rows = rows[1:]
        header = [str(col) for col in header][: self.max_columns]
        records: List[Dict[str, Any]] = []
        for row in rows:
            row_dict = {header[idx]: row[idx] for idx in range(min(len(header), len(row)))}
            records.append(self._sanitize_record(row_dict))
        return records

    def _columns_to_records(self, columns: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
        keys = list(columns.keys())[: self.max_columns]
        length = max((len(columns[key]) for key in keys), default=0)
        records: List[Dict[str, Any]] = []
        for idx in range(length):
            record: Dict[str, Any] = {}
            for key in keys:
                values = columns.get(key)
                if idx < len(values):
                    record[str(key)] = values[idx]
            records.append(self._sanitize_record(record))
        return records

    def _text_to_records(self, text: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = text.strip()
        if not text:
            return []
        fmt = (source.get("format") or "").lower()
        if not fmt:
            fmt = self._infer_text_format(text)

        if fmt in {"json", "jsonl"}:
            try:
                if fmt == "jsonl":
                    blobs = [json.loads(line) for line in text.splitlines() if line.strip()]
                else:
                    blobs = json.loads(text)
            except json.JSONDecodeError as exc:
                _LOGGER.debug("JSON parsing failed (%s), falling back to CSV: %s", fmt, exc)
                blobs = None
            if blobs is not None:
                if isinstance(blobs, dict):
                    if isinstance(blobs.get("records"), list):
                        blobs = blobs["records"]
                if isinstance(blobs, list):
                    return [self._sanitize_record(obj) for obj in blobs if isinstance(obj, dict)]
                if isinstance(blobs, dict):
                    return [self._sanitize_record(blobs)]

        delimiter = self.separator if fmt != "tsv" else "\t"
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        return [self._sanitize_record(row) for row in reader]

    def _read_text(self, payload: Union[str, pathlib.Path], source: Dict[str, Any]) -> Optional[str]:
        path = pathlib.Path(str(payload))
        if path.exists():
            source.setdefault("path", str(path))
            source.setdefault("format", path.suffix.lstrip("."))
            try:
                return path.read_text(encoding="utf-8")
            except OSError as exc:
                _LOGGER.debug("Failed to read structured data file %s: %s", path, exc)
                return None
        return str(payload)

    @staticmethod
    def _infer_text_format(text: str) -> str:
        stripped = text.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            return "json"
        if "\t" in text and "," not in text:
            return "tsv"
        return "csv"

    @staticmethod
    def _sanitize_record(record: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for key, value in record.items():
            if value is None:
                continue
            clean[str(key)] = value
        return clean

    # ------------------------------------------------------------------ #
    # Triple + embedding helpers
    # ------------------------------------------------------------------ #
    def _records_to_triples(
        self,
        records: Sequence[Dict[str, Any]],
        source: Dict[str, Any],
    ) -> List[Triple]:
        triples: List[Triple] = []
        relation_overrides = source.get("relations") or {}
        default_relation = relation_overrides.get("*", self.default_relation)
        for idx, record in enumerate(records):
            subject = str(
                record.get(self.subject_field)
                or record.get("subject")
                or record.get("entity")
                or f"row_{idx}"
            )
            for key, value in record.items():
                if key == self.subject_field:
                    continue
                if value in (None, "", []):
                    continue
                relation = str(relation_overrides.get(key, default_relation if key == self.object_field else key))
                triples.append((subject, relation, self._stringify(value)))
        return triples

    def _compute_embeddings(
        self,
        records: Sequence[Dict[str, Any]],
        triples: Sequence[Triple],
    ) -> Dict[str, np.ndarray]:
        if not records:
            return {}

        numeric_values: List[float] = []
        token_counts: List[float] = []
        for record in records:
            for value in record.values():
                if isinstance(value, (int, float)):
                    numeric_values.append(float(value))
                elif isinstance(value, str):
                    token_counts.append(float(len(value.split())))

        summary = np.array(
            [
                float(len(records)),
                float(len(triples)),
                float(np.mean(numeric_values) if numeric_values else 0.0),
                float(np.std(numeric_values) if numeric_values else 0.0),
                float(np.mean(token_counts) if token_counts else 0.0),
                float(np.std(token_counts) if token_counts else 0.0),
            ],
            dtype=float,
        )
        column_vectors: Dict[str, List[float]] = {}
        for record in records:
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    column_vectors.setdefault(key, []).append(float(value))

        embeddings = {"summary": summary}
        for key, values in column_vectors.items():
            embeddings[f"column::{key}"] = np.array(
                [
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(np.min(values)),
                    float(np.max(values)),
                ],
                dtype=float,
            )
        return embeddings

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, (str, int, float)):
            return str(value)
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value if item is not None)
        return json.dumps(value, ensure_ascii=False, default=str)


__all__ = [
    "StructuredDataBatch",
    "StructuredDataParser",
]

