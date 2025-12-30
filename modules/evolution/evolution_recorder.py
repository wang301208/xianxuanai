from __future__ import annotations

"""Utilities for logging evolution steps to an external knowledge store."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from backend.autogpt.autogpt.core.knowledge_graph.graph_store import GraphStore
    from backend.autogpt.autogpt.core.knowledge_graph.ontology import (
        EntityType,
        RelationType,
    )
except Exception:  # pragma: no cover - optional dependency
    GraphStore = None  # type: ignore
    EntityType = None  # type: ignore
    RelationType = None  # type: ignore

from backend.knowledge.registry import get_graph_store_instance

from .self_evolving_cognition import EvolutionRecord


class EvolutionKnowledgeRecorder:
    """Persist evolution steps to disk and the internal knowledge graph."""

    def __init__(
        self,
        log_path: Path | str = Path("results") / "evolution_knowledge.jsonl",
        graph_store: Optional[GraphStore] = None,
        enable_graph: bool = True,
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph: Optional[GraphStore]
        graph_capable = enable_graph and GraphStore is not None
        if graph_capable:
            try:
                self.graph = graph_store or get_graph_store_instance()
            except Exception:
                self.graph = None
        else:
            self.graph = None
        self._last_version: Optional[int] = None

    # ------------------------------------------------------------------
    def record(
        self,
        record: EvolutionRecord,
        previous_architecture: Optional[Dict[str, float]] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record ``record`` along with an optional architecture diff."""

        payload = self._build_payload(record, previous_architecture, annotations)
        with self.log_path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")

        if self.graph is not None:
            self._write_to_graph(payload)
        self._last_version = record.version

    # ------------------------------------------------------------------
    def _build_payload(
        self,
        record: EvolutionRecord,
        previous_architecture: Optional[Dict[str, float]],
        annotations: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        data = {
            "version": record.version,
            "performance": record.performance,
            "architecture": record.architecture,
            "metrics": record.metrics,
        }
        if previous_architecture is not None:
            data["delta"] = _diff_architectures(previous_architecture, record.architecture)
        if annotations:
            data["annotations"] = annotations
        return data

    # ------------------------------------------------------------------
    def _write_to_graph(self, payload: Dict[str, Any]) -> None:
        if self.graph is None or EntityType is None or RelationType is None:
            return
        node_id = f"evolution:version:{payload['version']}"
        try:
            self.graph.add_node(
                node_id,
                EntityType.CONCEPT,
                performance=float(payload["performance"]),
                metrics=json.dumps(payload.get("metrics", {}), ensure_ascii=False),
                delta=json.dumps(payload.get("delta", {}), ensure_ascii=False),
                annotations=json.dumps(payload.get("annotations", {}), ensure_ascii=False),
            )
            if self._last_version is not None:
                prev_id = f"evolution:version:{self._last_version}"
                self.graph.add_edge(prev_id, node_id, RelationType.RELATED_TO)
        except Exception:
            # Graph persistence is best effort; ignore failures.
            return


def _diff_architectures(
    old_arch: Dict[str, float],
    new_arch: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    delta: Dict[str, Dict[str, float]] = {}
    keys = set(old_arch.keys()) | set(new_arch.keys())
    for key in sorted(keys):
        old_val = float(old_arch.get(key, 0.0))
        new_val = float(new_arch.get(key, 0.0))
        if abs(old_val - new_val) > 1e-9:
            delta[key] = {"old": old_val, "new": new_val, "change": new_val - old_val}
    return delta

