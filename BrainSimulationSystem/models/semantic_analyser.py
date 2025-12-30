"""
Internal semantic analysis utilities for concept activation and relation mining.

This module translates token sequences into activations on the project's
SemanticNetwork, extracts high-salience concepts, and heuristically infers
lightweight semantic relations. It is designed to replace external LLM-based
keyword/entity extraction with fully local logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from BrainSimulationSystem.models.language_processing import SemanticNetwork


@dataclass
class SemanticAnalysisResult:
    key_terms: List[str]
    entities: List[str]
    relations: List[Dict[str, str]]
    activation_map: Dict[str, float]
    concept_stats: Dict[str, Dict[str, Any]]
    suggested_actions: List[str]
    confidence: float


class SemanticAnalyser:
    """
    Map language tokens into semantic network activations and derive key concepts.
    """

    def __init__(self, network: SemanticNetwork, params: Optional[Dict[str, Any]] = None) -> None:
        self.network = network
        self.params = params or {}
        self.top_k = int(self.params.get("top_k", 6))
        self.activation_threshold = float(self.params.get("activation_threshold", 0.18))
        self.activation_weights = {
            "V": float(self.params.get("verb_activation", 0.35)),
            "Gerund": float(self.params.get("gerund_activation", 0.32)),
            "N": float(self.params.get("noun_activation", 0.28)),
            "ProperNoun": float(self.params.get("proper_activation", 0.32)),
            "Adj": float(self.params.get("adj_activation", 0.22)),
            "Adv": float(self.params.get("adv_activation", 0.18)),
        }
        self.default_activation = float(self.params.get("default_activation", 0.14))

        base_rel = {
            "subject": 0.6,
            "object": 0.6,
            "modifier": 0.35,
            "attribute": 0.4,
            "aux": 0.25,
            "root": 0.5,
        }
        override_rel = self.params.get("relation_strengths") or {}
        base_rel.update({k: float(v) for k, v in override_rel.items()})
        self.relation_strengths = base_rel

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def analyse(
        self,
        text: str,
        tokens: Sequence[str],
        normalized_tokens: Sequence[str],
        parse_result: Dict[str, Any],
    ) -> SemanticAnalysisResult:
        """
        Analyse a token sequence, update the semantic network and extract insights.

        Parameters
        ----------
        text:
            Original text input.
        tokens:
            Token list aligned with parse indices (original casing stripped of punctuation).
        normalized_tokens:
            Lower-case tokens aligned to ``tokens`` (empty string when token was discarded).
        parse_result:
            Output of ``SyntaxProcessor.parse_sentence`` providing POS tags and dependency arcs.
        """

        pos_tags: List[str] = parse_result.get("meta", {}).get("pos_tags", [])
        dependency = parse_result.get("meta", {}).get("dependency", {})
        arcs: Sequence[Tuple[int, int, str]] = dependency.get("arcs", []) or []

        index_to_concept: Dict[int, str] = {}
        concept_stats: Dict[str, Dict[str, Any]] = {}
        entities: List[str] = []
        seen_for_activation: set[str] = set()

        for idx, concept in enumerate(normalized_tokens):
            if not concept:
                continue
            pos = pos_tags[idx] if idx < len(pos_tags) else "N"
            index_to_concept[idx] = concept

            if concept not in self.network.nodes:
                self.network.add_node(concept, {"pos": pos})
            else:
                self.network.nodes[concept]["attributes"].setdefault("pos", pos)

            if concept not in seen_for_activation:
                seen_for_activation.add(concept)
                self.network.activate_concept(concept, self._activation_for_pos(pos))

            stats = concept_stats.setdefault(concept, {"count": 0, "pos": pos})
            stats["count"] += 1
            stats["activation"] = float(self.network.nodes[concept]["activation"])
            stats["pos"] = pos

            original = tokens[idx] if idx < len(tokens) else ""
            if original and original[:1].isupper() and original.lower() != concept:
                if original not in entities:
                    entities.append(original)
                self.network.nodes[concept]["attributes"]["entity"] = True

        relations: List[Dict[str, str]] = []
        suggested_actions: set[str] = set()

        for head_idx, dep_idx, relation in arcs:
            dep_concept = index_to_concept.get(dep_idx)
            if not dep_concept:
                continue

            if head_idx == -1:
                relations.append({"head": "ROOT", "dependent": dep_concept, "relation": relation})
                continue

            head_concept = index_to_concept.get(head_idx)
            if not head_concept:
                continue

            strength = self.relation_strengths.get(relation, 0.3)
            self._ensure_relation(head_concept, dep_concept, relation, strength)
            relations.append({"head": head_concept, "dependent": dep_concept, "relation": relation})

            head_pos = pos_tags[head_idx] if head_idx < len(pos_tags) else ""
            if relation in {"subject", "object"} and head_pos in {"V", "Gerund"}:
                suggested_actions.add(f"{head_concept} -> {dep_concept} ({relation})")

        activation_map = {
            concept: float(round(info.get("activation", 0.0), 4))
            for concept, info in concept_stats.items()
        }

        sorted_concepts = sorted(activation_map.items(), key=lambda item: item[1], reverse=True)
        key_terms = [
            concept
            for concept, score in sorted_concepts
            if score >= self.activation_threshold
        ][: self.top_k]
        if not key_terms and sorted_concepts:
            key_terms = [concept for concept, _ in sorted_concepts[: self.top_k]]

        max_activation = sorted_concepts[0][1] if sorted_concepts else 0.0
        coverage = len(index_to_concept) / max(1, len(tokens))
        confidence = float(
            min(
                0.95,
                0.35 + 0.4 * coverage + 0.25 * min(max_activation, 1.0),
            )
        )

        return SemanticAnalysisResult(
            key_terms=key_terms,
            entities=entities,
            relations=relations,
            activation_map=activation_map,
            concept_stats=concept_stats,
            suggested_actions=sorted(suggested_actions),
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _activation_for_pos(self, pos: str) -> float:
        return self.activation_weights.get(pos, self.default_activation)

    def _ensure_relation(self, head: str, dependent: str, relation: str, strength: float) -> None:
        if head not in self.network.nodes:
            self.network.add_node(head, {"origin": "semantic"})
        if dependent not in self.network.nodes:
            self.network.add_node(dependent, {"origin": "semantic"})
        existing = self.network.relations.get(head, {}).get(dependent)
        if not existing or existing.get("strength", 0.0) < strength:
            self.network.add_relation(head, dependent, relation, strength)
