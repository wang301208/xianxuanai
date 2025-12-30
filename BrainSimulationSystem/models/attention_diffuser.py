"""
Attention diffuser for semantic network priming.

This module nudges the semantic network towards concepts that are relevant
according to working memory and the current lexical evidence.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from BrainSimulationSystem.models.language_processing import SemanticNetwork


class AttentionDiffuser:
    """Apply targeted activation boosts before comprehension."""

    def __init__(self, params: Optional[Dict[str, float]] = None) -> None:
        params = params or {}
        self.memory_boost = float(params.get("memory_boost", 0.12))
        self.token_boost = float(params.get("token_boost", 0.08))
        self.pending_boost = float(params.get("pending_boost", 0.18))
        self.focus_boost = float(params.get("focus_boost", 0.22))
        self.neighbour_boost = float(params.get("neighbour_boost", 0.12))
        self.suppression_factor = float(params.get("suppression_factor", 0.75))
        self.top_n = int(params.get("top_n", 12))

    def apply(
        self,
        network: SemanticNetwork,
        memory_state: Dict[str, Any],
        normalized_tokens: Sequence[str],
        focus_terms: Optional[Sequence[str]] = None,
        directives: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._boost_terms(network, memory_state.get("key_terms", []), self.memory_boost)
        self._boost_terms(network, normalized_tokens, self.token_boost)
        self._boost_terms(network, memory_state.get("pending_actions", []), self.pending_boost)
        combined_focus: List[str] = []
        if focus_terms:
            combined_focus.extend(focus_terms)
        if directives:
            combined_focus.extend(directives.get("semantic_focus") or [])
            self._boost_terms(
                network,
                directives.get("goal_snapshot", []),
                self.memory_boost * 0.8,
            )
        if combined_focus:
            self._apply_focus(network, combined_focus)
        self._suppress_unfocused(network)

    def _boost_terms(
        self,
        network: SemanticNetwork,
        terms: Iterable[str],
        amount: float,
    ) -> None:
        for term in terms:
            if not term:
                continue
            if term not in network.nodes:
                network.add_node(term, {"origin": "attention"})
            network.activate_concept(term, amount=amount)

    def _apply_focus(self, network: SemanticNetwork, focus_terms: Sequence[str]) -> None:
        for term in focus_terms:
            if not term:
                continue
            if term not in network.nodes:
                network.add_node(term, {"origin": "focus"})
            network.activate_concept(term, amount=self.focus_boost)
            neighbours = network.get_related_concepts(term) if hasattr(network, "get_related_concepts") else []
            for neighbour in neighbours:
                if neighbour not in network.nodes:
                    network.add_node(neighbour, {"origin": "focus_neighbour"})
                network.activate_concept(neighbour, amount=self.neighbour_boost)

    def _suppress_unfocused(self, network: SemanticNetwork) -> None:
        if self.top_n <= 0 or self.suppression_factor >= 1.0:
            return
        activations = [
            (concept, info.get("activation", 0.0))
            for concept, info in network.nodes.items()
        ]
        if not activations:
            return
        activations.sort(key=lambda item: item[1], reverse=True)
        focus_set = {concept for concept, _ in activations[: self.top_n]}
        for concept, info in network.nodes.items():
            if concept in focus_set:
                continue
            current = info.get("activation", 0.0)
            info["activation"] = current * self.suppression_factor

    def refine_focus(
        self,
        network: SemanticNetwork,
        focus_terms: Sequence[str],
    ) -> None:
        if not focus_terms:
            return
        self._apply_focus(network, focus_terms)
        self._suppress_unfocused(network)
