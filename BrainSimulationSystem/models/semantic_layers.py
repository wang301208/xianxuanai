"""
Layered semantic representation utilities.

This module introduces a lightweight separation between local (per-turn) concept
graphs and the long-term semantic network, together with a hippocampal-style
integrator that decides when to consolidate new knowledge.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from BrainSimulationSystem.models.language_processing import SemanticNetwork


class LocalSemanticField:
    """Isolated concept graph capturing the semantics of the current turn."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.relations: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}

    def reset(self) -> None:
        self.nodes.clear()
        self.relations.clear()
        self.metadata.clear()

    def add_node(self, concept: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        if concept not in self.nodes:
            self.nodes[concept] = {
                "activation": 0.0,
                "count": 0,
                "attributes": dict(attributes or {}),
            }
        else:
            if attributes:
                self.nodes[concept]["attributes"].update(attributes)

    def add_relation(
        self,
        head: str,
        dependent: str,
        relation: str,
        strength: float = 1.0,
    ) -> None:
        key = (head, dependent)
        existing = self.relations.get(key)
        if existing:
            existing["strength"] = max(existing["strength"], strength)
            existing["types"].add(relation)
        else:
            self.relations[key] = {
                "types": {relation},
                "strength": strength,
            }

    def activate(self, concept: str, amount: float) -> None:
        if concept not in self.nodes:
            self.add_node(concept)
        self.nodes[concept]["activation"] = min(1.0, self.nodes[concept]["activation"] + amount)
        self.nodes[concept]["count"] = self.nodes[concept].get("count", 0) + 1

    def summarise(self, top_n: int = 6) -> Dict[str, Any]:
        activations = sorted(
            ((concept, data["activation"]) for concept, data in self.nodes.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        top_concepts = [concept for concept, _ in activations[:top_n]]
        relation_types = Counter()
        for info in self.relations.values():
            relation_types.update(info["types"])
        summary = {
            "top_concepts": top_concepts,
            "relation_counts": dict(relation_types),
            "node_count": len(self.nodes),
            "relation_count": len(self.relations),
        }
        self.metadata["summary"] = summary
        return summary


class HippocampalIntegrator:
    """Decide when to consolidate local semantic knowledge into the global network."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        params = params or {}
        self.commit_threshold = float(params.get("commit_threshold", 0.25))
        self.repetition_threshold = int(params.get("repetition_threshold", 2))
        self.max_relations = int(params.get("max_relations", 24))

    def should_commit(
        self,
        local_field: LocalSemanticField,
        confidence: float,
    ) -> bool:
        if confidence >= self.commit_threshold:
            return True
        for concept, info in local_field.nodes.items():
            if info.get("count", 0) >= self.repetition_threshold and info.get("activation", 0.0) > 0.15:
                return True
        return False

    def consolidate(
        self,
        local_field: LocalSemanticField,
        global_network: SemanticNetwork,
        importance: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        committed = {"nodes": [], "relations": []}
        importance = importance or []
        importance_set = set(importance)

        # Add nodes
        for concept, info in local_field.nodes.items():
            if concept not in global_network.nodes:
                global_network.add_node(concept, info.get("attributes"))
            global_network.activate_concept(concept, amount=min(0.3, info.get("activation", 0.0) + 0.1))
            committed["nodes"].append(concept)

        # Add relations (bounded)
        rel_count = 0
        for (head, dep), info in local_field.relations.items():
            if rel_count >= self.max_relations:
                break
            strength = info.get("strength", 1.0)
            for relation in info.get("types", []):
                weight = strength + (0.2 if head in importance_set or dep in importance_set else 0.0)
                if head in global_network.nodes and dep in global_network.nodes:
                    global_network.add_relation(head, dep, relation, weight)
                    rel_count += 1
                    committed["relations"].append((head, dep, relation))
        return committed


class SemanticLayerManager:
    """Coordinate local and global semantic representations."""

    def __init__(
        self,
        global_network: SemanticNetwork,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        params = params or {}
        self.global_network = global_network
        self.local_field = LocalSemanticField()
        self.hippocampus = HippocampalIntegrator(params.get("hippocampus", {}))
        self.default_activation = float(params.get("default_activation", 0.18))

    def start_turn(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.local_field.reset()
        if metadata:
            self.local_field.metadata.update(metadata)

    def ingest_semantic_info(self, semantic_info: Any) -> None:
        if semantic_info is None:
            return
        key_terms = getattr(semantic_info, "key_terms", []) or []
        relations = getattr(semantic_info, "relations", []) or []
        activation_map = getattr(semantic_info, "activation_map", {}) or {}

        for term in key_terms:
            self.local_field.add_node(term, {"origin": "semantic_info"})
            self.local_field.activate(term, self.default_activation)

        for concept, activation in activation_map.items():
            self.local_field.add_node(concept)
            self.local_field.activate(concept, float(activation))

        for relation in relations:
            head = relation.get("head")
            dependent = relation.get("dependent")
            relation_type = relation.get("relation")
            if head and dependent and relation_type:
                self.local_field.add_relation(head, dependent, relation_type)

        self.local_field.summarise()

    def add_focus_terms(self, terms: Iterable[str]) -> None:
        for term in terms:
            self.local_field.add_node(term, {"origin": "focus"})
            self.local_field.activate(term, self.default_activation * 1.5)

    def commit_if_needed(self, confidence: float, importance: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
        if not self.hippocampus.should_commit(self.local_field, confidence):
            return None
        return self.hippocampus.consolidate(self.local_field, self.global_network, importance)

    def get_local_summary(self) -> Dict[str, Any]:
        return self.local_field.metadata.get("summary", {})
