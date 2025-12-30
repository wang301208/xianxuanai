"""
Lightweight knowledge graph module.

Provides basic storage for fact triples plus utilities for logical assertions,
consistency checks, and query lookup. Designed to operate without external
database dependencies while allowing optional hooks for larger systems.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union


Triple = Tuple[str, str, str]
TripleLike = Union[Triple, Tuple[str, str, str, Dict[str, Any]]]


@dataclass
class KnowledgeConstraint:
    """Represents a simple logical constraint over triples."""

    description: str
    required: List[Triple] = field(default_factory=list)
    forbidden: List[Triple] = field(default_factory=list)


class KnowledgeGraph:
    """In-memory knowledge graph supporting basic consistency checks."""

    def __init__(self) -> None:
        self.triples: Set[Triple] = set()
        self.metadata: Dict[Triple, Dict[str, Any]] = {}
        self.index_spo: Dict[str, Dict[str, Set[str]]] = {}
        self.index_pos: Dict[str, Dict[str, Set[str]]] = {}
        self.index_osp: Dict[str, Dict[str, Set[str]]] = {}

    # ------------------------------------------------------------------ #
    # Mutators
    # ------------------------------------------------------------------ #
    def add(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        triple = (subject, predicate, obj)
        existed = triple in self.triples
        if not existed:
            self.triples.add(triple)
            self.index_spo.setdefault(subject, {}).setdefault(predicate, set()).add(obj)
            self.index_pos.setdefault(predicate, {}).setdefault(obj, set()).add(subject)
            self.index_osp.setdefault(obj, {}).setdefault(subject, set()).add(predicate)

        if metadata is not None:
            self.metadata[triple] = dict(metadata)
        elif not existed and triple in self.metadata:
            # Ensure stale metadata from previous lifecycle is cleared.
            self.metadata.pop(triple, None)

    def add_many(self, triples: Iterable[TripleLike]) -> None:
        for entry in triples:
            if len(entry) == 4 and isinstance(entry[3], dict):
                subject, predicate, obj, meta = entry  # type: ignore[misc]
                self.add(str(subject), str(predicate), str(obj), metadata=meta)
            elif len(entry) == 3:
                subject, predicate, obj = entry  # type: ignore[misc]
                self.add(str(subject), str(predicate), str(obj))
            else:  # pragma: no cover - defensive guard
                raise ValueError(f"Unsupported triple format: {entry!r}")

    def remove(self, subject: str, predicate: str, obj: str) -> None:
        triple = (subject, predicate, obj)
        if triple not in self.triples:
            return
        self.triples.remove(triple)
        self.index_spo.get(subject, {}).get(predicate, set()).discard(obj)
        self.index_pos.get(predicate, {}).get(obj, set()).discard(subject)
        self.index_osp.get(obj, {}).get(subject, set()).discard(predicate)
        self.metadata.pop(triple, None)

    def upsert_triples(
        self,
        triples: Iterable[Triple],
        *,
        default_metadata: Optional[Dict[str, Any]] = None,
        per_triple_metadata: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
    ) -> int:
        """
        Insert or update ``triples`` and attached metadata.

        Returns
        -------
        int
            Number of newly inserted triples (excluding updates).
        """

        inserted = 0
        metadata_iter: Optional[Iterator[Optional[Dict[str, Any]]]] = (
            iter(per_triple_metadata) if per_triple_metadata is not None else None
        )

        for triple in triples:
            subject, predicate, obj = triple
            existed = self.exists(subject, predicate, obj)
            metadata_payload: Dict[str, Any] = {}
            if default_metadata:
                metadata_payload.update(default_metadata)
            if metadata_iter is not None:
                try:
                    extra = next(metadata_iter)
                except StopIteration:
                    metadata_iter = None
                else:
                    if extra:
                        metadata_payload.update(extra)
            self.add(subject, predicate, obj, metadata=metadata_payload or None)
            if not existed:
                inserted += 1
        return inserted

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def exists(self, subject: str, predicate: str, obj: str) -> bool:
        return (subject, predicate, obj) in self.triples

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> Set[Triple]:
        if subject is None and predicate is None and obj is None:
            return set(self.triples)

        results: Set[Triple] = set()

        if subject is not None:
            preds = self.index_spo.get(subject, {})
            if predicate is not None:
                objs = preds.get(predicate, set())
                if obj is not None:
                    if obj in objs:
                        results.add((subject, predicate, obj))
                else:
                    results.update((subject, predicate, o) for o in objs)
            else:
                for p, objs in preds.items():
                    if obj is None:
                        results.update((subject, p, o) for o in objs)
                    elif obj in objs:
                        results.add((subject, p, obj))
            return results

        if predicate is not None:
            objs = self.index_pos.get(predicate, {})
            if obj is not None:
                subjects = objs.get(obj, set())
                results.update((s, predicate, obj) for s in subjects)
            else:
                for o, subjects in objs.items():
                    results.update((s, predicate, o) for s in subjects)
            return results

        if obj is not None:
            subjects = self.index_osp.get(obj, {})
            for s, preds in subjects.items():
                if predicate is None:
                    results.update((s, p, obj) for p in preds)
                elif predicate in preds:
                    results.add((s, predicate, obj))
        return results

    def find_paths(
        self,
        start: str,
        *,
        target: Optional[str] = None,
        predicates: Optional[Iterable[str]] = None,
        max_depth: int = 3,
        max_paths: int = 5,
        bidirectional: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Return up to ``max_paths`` paths originating from ``start`` within ``max_depth`` hops.

        Each returned entry contains:
            - ``nodes``: ordered list of visited nodes starting with ``start``.
            - ``triples``: list of dictionaries ``{\"triple\": (s, p, o), \"direction\": str}``.
              ``direction`` is ``"forward"`` when the triple aligns with the traversal direction
              and ``"backward"`` when the search followed the triple in reverse (object -> subject).
            - ``target``: the last node in the path (duplicate of ``nodes[-1]``).
            - ``source``: always ``"knowledge_graph"`` for easier provenance tracking.
        """

        start_node = str(start)
        predicate_filter: Optional[Set[str]] = set(map(str.lower, predicates or [])) or None

        def _passes_filter(predicate: str) -> bool:
            if predicate_filter is None:
                return True
            return predicate.lower() in predicate_filter

        results: List[Dict[str, Any]] = []
        if start_node not in self.index_spo and start_node not in self.index_osp:
            return results

        queue: deque[Tuple[str, List[Dict[str, Any]], List[str]]] = deque()
        queue.append((start_node, [], [start_node]))
        visited: Set[str] = {start_node}

        while queue and len(results) < max_paths:
            node, path_triples, node_sequence = queue.popleft()
            depth = len(path_triples)

            neighbors: List[Tuple[Triple, str, str]] = []
            for predicate, objects in self.index_spo.get(node, {}).items():
                if not _passes_filter(predicate):
                    continue
                for obj in objects:
                    neighbors.append(((node, predicate, obj), obj, "forward"))

            if bidirectional:
                reverse_entries = self.index_osp.get(node, {})
                for subj, predicates_map in reverse_entries.items():
                    if isinstance(predicates_map, dict):
                        predicates_iter = predicates_map.keys()
                    else:
                        predicates_iter = predicates_map
                    for predicate in predicates_iter:
                        if not _passes_filter(predicate):
                            continue
                        neighbors.append(((subj, predicate, node), subj, "backward"))

            for triple, neighbor, direction in neighbors:
                new_path = path_triples + [{"triple": triple, "direction": direction}]
                new_nodes = node_sequence + [neighbor]

                if target is None or neighbor == target:
                    results.append(
                        {
                            "nodes": new_nodes,
                            "triples": new_path,
                            "target": neighbor,
                            "source": "knowledge_graph",
                        }
                    )
                    if len(results) >= max_paths:
                        break

                if len(new_path) >= max_depth:
                    continue

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_path, new_nodes))
            # end neighbor loop

        return results[:max_paths]

    # ------------------------------------------------------------------ #
    # Constraints and reasoning aids
    # ------------------------------------------------------------------ #
    def check_constraints(self, constraints: Iterable[KnowledgeConstraint]) -> Dict[str, bool]:
        """Return mapping `description -> satisfied`."""
        results: Dict[str, bool] = {}
        for constraint in constraints:
            satisfied = True
            for triple in constraint.required:
                if triple not in self.triples:
                    satisfied = False
                    break
            if satisfied:
                for triple in constraint.forbidden:
                    if triple in self.triples:
                        satisfied = False
                        break
            results[constraint.description] = satisfied
        return results

    # ------------------------------------------------------------------ #
    # Action-specific constraint evaluation
    # ------------------------------------------------------------------ #
    def evaluate_action_constraints(
        self,
        action: Any,
        constraints: Iterable[KnowledgeConstraint],
    ) -> Dict[str, Any]:
        """Return detailed constraint satisfaction report for ``action``."""

        action_str = str(action)
        violations: List[Dict[str, Any]] = []
        details: Dict[str, Dict[str, Any]] = {}
        satisfied_all = True

        for constraint in constraints or []:
            missing: List[Tuple[str, str, str]] = []
            triggered: List[Tuple[str, str, str]] = []

            for triple in constraint.required:
                resolved = self._resolve_placeholder(triple, action_str)
                if not self._triple_satisfied(resolved):
                    missing.append(resolved)

            if not missing:
                for triple in constraint.forbidden:
                    resolved = self._resolve_placeholder(triple, action_str)
                    if self._triple_satisfied(resolved):
                        triggered.append(resolved)

            entry = {
                "required_met": not missing,
                "forbidden_clear": not triggered,
                "missing": [self._format_triple(t) for t in missing],
                "triggered": [self._format_triple(t) for t in triggered],
            }
            details[constraint.description] = entry
            if missing or triggered:
                satisfied_all = False
                violations.append(
                    {
                        "constraint": constraint.description,
                        "missing": entry["missing"],
                        "triggered": entry["triggered"],
                    }
                )

        return {
            "satisfied": satisfied_all,
            "violations": violations,
            "details": details,
        }

    # ------------------------------------------------------------------ #
    # Metadata helpers
    # ------------------------------------------------------------------ #
    def get_metadata(self, subject: str, predicate: str, obj: str) -> Dict[str, Any]:
        return dict(self.metadata.get((subject, predicate, obj), {}))

    def metadata_for(self, triple: Triple) -> Dict[str, Any]:
        return self.get_metadata(triple[0], triple[1], triple[2])

    def touch(self, subject: str, predicate: str, obj: str, *, now: float | None = None) -> Dict[str, Any]:
        """Mark a triple as accessed to support pruning/retention policies."""

        triple = (subject, predicate, obj)
        if triple not in self.triples:
            return {}
        stamp = float(time.time() if now is None else now)
        meta = dict(self.metadata.get(triple, {}))
        meta["usage"] = int(meta.get("usage", 0)) + 1
        meta["last_access"] = stamp
        meta.setdefault("created_at", stamp)
        self.metadata[triple] = meta
        return dict(meta)

    def prune(
        self,
        *,
        min_confidence: float = 0.0,
        min_usage: int = 0,
        max_age_s: float | None = None,
        now: float | None = None,
    ) -> int:
        """Remove low-value triples based on metadata (best-effort).

        Recognised metadata keys:
            - confidence: float in [0, 1]
            - usage: int
            - last_access / created_at: timestamps (seconds)
        """

        stamp = float(time.time() if now is None else now)
        removed = 0
        for triple in list(self.triples):
            meta = self.metadata.get(triple, {})
            try:
                confidence = float(meta.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            usage = int(meta.get("usage", 0) or 0)
            last_access = meta.get("last_access", meta.get("created_at"))
            age_ok = True
            if max_age_s is not None and last_access is not None:
                try:
                    age_ok = (stamp - float(last_access)) <= float(max_age_s)
                except (TypeError, ValueError):
                    age_ok = True
            if confidence < float(min_confidence) or usage < int(min_usage) or not age_ok:
                self.remove(triple[0], triple[1], triple[2])
                removed += 1
        return removed

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _resolve_placeholder(
        self,
        triple: Tuple[Any, Any, Any],
        action: str,
    ) -> Tuple[Any, Any, Any]:
        subject, predicate, obj = triple
        return (
            self._resolve_value(subject, action),
            self._resolve_value(predicate, action),
            self._resolve_value(obj, action),
        )

    @staticmethod
    def _resolve_value(value: Any, action: str) -> Any:
        if isinstance(value, str):
            cleaned = value.strip()
            if "{action}" in cleaned:
                return cleaned.replace("{action}", action)
            if cleaned in {"<action>", "ACTION", "action"}:
                return action
        return value

    def _triple_satisfied(self, triple: Tuple[Any, Any, Any]) -> bool:
        subject, predicate, obj = triple
        subject = None if subject in ("", None, "*") else subject
        obj = None if obj in ("", None, "*") else obj
        predicate = str(predicate)

        if subject is None and obj is None:
            return bool(self.query(predicate=predicate))
        if subject is None:
            return bool(self.query(predicate=predicate, obj=str(obj)))
        if obj is None:
            return bool(self.query(subject=str(subject), predicate=predicate))
        return self.exists(str(subject), predicate, str(obj))

    @staticmethod
    def _format_triple(triple: Tuple[Any, Any, Any]) -> Tuple[str, str, str]:
        return tuple(str(part) for part in triple)

    def related_entities(self, entity: str) -> Set[str]:
        neighbors: Set[str] = set()
        for predicate, objs in self.index_spo.get(entity, {}).items():
            neighbors.update(objs)
        for predicate, subjects in self.index_pos.items():
            for obj, subs in subjects.items():
                if entity in subs:
                    neighbors.add(obj)
        return neighbors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triples": list(self.triples),
            "metadata": [
                {
                    "triple": [subject, predicate, obj],
                    "metadata": dict(meta),
                }
                for (subject, predicate, obj), meta in self.metadata.items()
                if meta
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        graph = cls()
        for triple in data.get("triples", []):
            if isinstance(triple, (list, tuple)) and len(triple) == 3:
                graph.add(str(triple[0]), str(triple[1]), str(triple[2]))
        for entry in data.get("metadata", []):
            triple = entry.get("triple")
            meta = entry.get("metadata") or {}
            if (
                isinstance(triple, (list, tuple))
                and len(triple) == 3
                and isinstance(meta, dict)
            ):
                graph.add(str(triple[0]), str(triple[1]), str(triple[2]), metadata=meta)
        return graph
