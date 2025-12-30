"""
Knowledge ingestion manager for periodic and curiosity-driven updates.

The manager reuses the existing :func:`load_external_sources` helper for
configured sources while also supporting lightweight topic-based injections to
keep the knowledge graph fresh without manual intervention.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from BrainSimulationSystem.knowledge.source_loader import load_external_sources
from BrainSimulationSystem.models.knowledge_graph import KnowledgeGraph, Triple


@dataclass
class TopicRequest:
    topic: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_ingested: float = 0.0


class KnowledgeIngestionManager:
    """Coordinate scheduled knowledge ingestion and validation."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]],
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        cfg = config or {}
        self.refresh_interval = float(cfg.get("refresh_interval", 3600.0))
        self.topic_interval = float(cfg.get("topic_interval", 900.0))
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.base_sources: List[Dict[str, Any]] = list(cfg.get("sources", []))
        self.topic_template: Dict[str, Any] = cfg.get("topic_template", {})
        self.last_refresh = 0.0
        self.pending_topics: Dict[str, TopicRequest] = {}
        self.min_triples_per_topic = int(cfg.get("min_triples_per_topic", 1))

    # ------------------------------------------------------------------ #
    def request_topic(self, topic: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        topic = str(topic).strip()
        if not topic:
            return
        entry = self.pending_topics.get(topic)
        if entry is None:
            entry = TopicRequest(topic=topic, metadata=dict(metadata or {}))
            self.pending_topics[topic] = entry
        else:
            entry.metadata.update(metadata or {})

    # ------------------------------------------------------------------ #
    def tick(
        self,
        current_time: float,
        graph: KnowledgeGraph,
        constraints: Optional[Iterable[Any]] = None,
    ) -> Dict[str, Any]:
        summary = {
            "added": 0,
            "topics": [],
        }
        if graph is None:
            return summary

        if current_time - self.last_refresh >= self.refresh_interval:
            try:
                added = load_external_sources(graph, self.base_sources, logger=self.logger)
                summary["added"] += added
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.warning("Base knowledge ingestion failed: %s", exc)
            else:
                self.last_refresh = current_time

        for topic, request in list(self.pending_topics.items()):
            if current_time - request.last_ingested < self.topic_interval:
                continue
            triples = self._topic_to_triples(topic, request.metadata, current_time)
            if not triples:
                continue
            graph.add_many(triples)
            summary["added"] += len(triples)
            summary["topics"].append(topic)
            request.last_ingested = current_time

        if constraints:
            try:
                validation = graph.check_constraints(constraints)
                summary["constraints"] = validation
            except Exception:  # pragma: no cover - defensive fallback
                pass
        return summary

    # ------------------------------------------------------------------ #
    def _topic_to_triples(
        self,
        topic: str,
        metadata: Dict[str, Any],
        now: float,
    ) -> List[Triple]:
        triples: List[Triple] = []
        source = metadata.get("source") or self.topic_template.get("source", "auto_ingest")
        timestamp = metadata.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        focus = metadata.get("focus") or self.topic_template.get("focus", "general")
        summary = metadata.get("summary") or f"pending research on {topic}"

        triples.append((topic, "source", source))
        triples.append((topic, "updated_at", timestamp))
        triples.append((topic, "focus", str(focus)))
        triples.append((topic, "status", "pending_verification"))
        triples.append((topic, "summary", summary))

        tags = metadata.get("tags") or self.topic_template.get("tags", [])
        for tag in tags:
            triples.append((topic, "tag", str(tag)))

        return triples[: max(self.min_triples_per_topic, len(triples))]


__all__ = ["KnowledgeIngestionManager"]

