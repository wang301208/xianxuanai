"""Utilities for augmenting the knowledge graph from external sources."""

from __future__ import annotations

import csv
import json
import logging
import pathlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import urlopen

from BrainSimulationSystem.models.knowledge_graph import KnowledgeGraph


Triple = Tuple[str, str, str]


def load_external_sources(
    graph: KnowledgeGraph,
    sources: Iterable[Dict[str, object]],
    *,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Load triples from configured sources into the provided knowledge graph."""

    log = logger or logging.getLogger(__name__)
    total_added = 0
    for source in sources:
        try:
            triples = _load_source(source, log)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning("Failed to load knowledge source %s: %s", source, exc)
            continue
        if triples:
            graph.add_many(triples)
            total_added += len(triples)
    return total_added


def _load_source(source: Dict[str, object], logger: logging.Logger) -> List[Triple]:
    source_type = str(source.get("type", "file")).lower()
    fmt = str(source.get("format", "json")).lower()
    if source_type == "file":
        path = pathlib.Path(str(source.get("path", ""))).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"knowledge source file not found: {path}")
        return _parse_file(path, fmt, source, logger)
    if source_type == "http":
        url = str(source.get("url"))
        timeout = float(source.get("timeout", 10.0))
        try:
            with urlopen(url, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
        except URLError as exc:
            raise ConnectionError(f"Failed to fetch knowledge source {url}: {exc}") from exc
        return _parse_payload(payload, fmt, source, logger)
    logger.warning("Unsupported knowledge source type '%s'", source_type)
    return []


def _parse_file(path: pathlib.Path, fmt: str, source: Dict[str, object], logger: logging.Logger) -> List[Triple]:
    if fmt in {"json", "jsonl"}:
        if fmt == "jsonl":
            triples: List[Triple] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    triples.extend(_extract_triples(payload, source))
            return triples
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return _extract_triples(payload, source)
    if fmt == "csv":
        triples_csv: List[Triple] = []
        subject_key = str(source.get("subject_field", "subject"))
        predicate_key = str(source.get("predicate_field", "predicate"))
        object_key = str(source.get("object_field", "object"))
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                triples_csv.extend(
                    _from_mapping(row, subject_key, predicate_key, object_key)
                )
        return triples_csv
    if fmt in {"txt", "triples"}:
        triples_txt: List[Triple] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = [part.strip() for part in line.strip().split(",") if part.strip()]
                if len(parts) == 3:
                    triples_txt.append((parts[0], parts[1], parts[2]))
        return triples_txt
    logger.warning("Unsupported knowledge format '%s'", fmt)
    return []


def _parse_payload(payload: str, fmt: str, source: Dict[str, object], logger: logging.Logger) -> List[Triple]:
    if fmt in {"json", "jsonl"}:
        if fmt == "jsonl":
            return _extract_triples([json.loads(line) for line in payload.splitlines() if line.strip()], source)
        return _extract_triples(json.loads(payload), source)
    if fmt == "csv":
        subject_key = str(source.get("subject_field", "subject"))
        predicate_key = str(source.get("predicate_field", "predicate"))
        object_key = str(source.get("object_field", "object"))
        triples_csv: List[Triple] = []
        rows = csv.DictReader(payload.splitlines())
        for row in rows:
            triples_csv.extend(_from_mapping(row, subject_key, predicate_key, object_key))
        return triples_csv
    if fmt in {"txt", "triples"}:
        triples_txt: List[Triple] = []
        for line in payload.splitlines():
            parts = [part.strip() for part in line.strip().split(",") if part.strip()]
            if len(parts) == 3:
                triples_txt.append((parts[0], parts[1], parts[2]))
        return triples_txt
    logger.warning("Unsupported knowledge format '%s'", fmt)
    return []


def _extract_triples(data: object, source: Dict[str, object]) -> List[Triple]:
    triples: List[Triple] = []
    if isinstance(data, dict):
        if "triples" in data and isinstance(data["triples"], (list, tuple)):
            for entry in data["triples"]:
                triples.extend(_extract_triples(entry, source))
        else:
            subject_key = str(source.get("subject_field", "subject"))
            predicate_key = str(source.get("predicate_field", "predicate"))
            object_key = str(source.get("object_field", "object"))
            triples.extend(_from_mapping(data, subject_key, predicate_key, object_key))
    elif isinstance(data, (list, tuple)):
        if len(data) == 3 and all(isinstance(part, (str, int, float)) for part in data):
            triples.append(tuple(str(part) for part in data))  # type: ignore[arg-type]
        else:
            for entry in data:
                triples.extend(_extract_triples(entry, source))
    return triples


def _from_mapping(
    mapping: Dict[str, object],
    subject_key: str,
    predicate_key: str,
    object_key: str,
) -> List[Triple]:
    if subject_key in mapping and predicate_key in mapping and object_key in mapping:
        subject = str(mapping[subject_key])
        predicate = str(mapping[predicate_key])
        obj = str(mapping[object_key])
        return [(subject, predicate, obj)]
    return []
