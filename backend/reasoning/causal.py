from __future__ import annotations

"""Causal and counterfactual reasoning helpers."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .multi_hop import MultiHopAssociator
from .interfaces import CausalReasoner, CounterfactualReasoner


@dataclass
class CausalEdge:
    target: str
    weight: float = 1.0
    relation: str = "causes"
    description: str | None = None


class CausalGraph:
    """Directed causal graph with optional weights and descriptions."""

    def __init__(self) -> None:
        self.children: Dict[str, List[CausalEdge]] = {}

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        weight: float = 1.0,
        relation: str = "causes",
        description: str | None = None,
    ) -> None:
        edge = CausalEdge(target=target, weight=weight, relation=relation, description=description)
        self.children.setdefault(source, []).append(edge)

    @classmethod
    def from_graph(
        cls,
        graph: dict[str, Iterable[str]],
        *,
        default_weight: float = 1.0,
    ) -> "CausalGraph":
        causal_graph = cls()
        for source, targets in graph.items():
            for target in targets:
                causal_graph.add_edge(source, target, weight=default_weight)
        return causal_graph

    def find_path(self, source: str, target: str) -> List[str]:
        associator = MultiHopAssociator({node: [edge.target for edge in edges] for node, edges in self.children.items()})
        return associator.find_path(source, target)

    def downstream(self, source: str, *, depth: int = 3) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        frontier: List[Tuple[str, float, int]] = [(source, 1.0, 0)]
        visited: Dict[str, float] = {}
        while frontier:
            node, strength, level = frontier.pop(0)
            if level >= depth:
                continue
            for edge in self.children.get(node, []):
                new_strength = strength * edge.weight
                prev = visited.get(edge.target)
                if prev is None or new_strength > prev:
                    visited[edge.target] = new_strength
                    results.append((edge.target, new_strength))
                frontier.append((edge.target, new_strength, level + 1))
        return results

    def simulate_intervention(self, source: str, target: str) -> str:
        path = self.find_path(source, target)
        if not path:
            return f"No causal chain connects {source} to {target}."
        chain = " -> ".join(path)
        return (
            f"Intervening on {source} breaks the chain {chain}, "
            f"reducing the likelihood of {target}."
        )


class KnowledgeGraphCausalReasoner(CausalReasoner):
    """Check causal relationships using a knowledge graph."""

    def __init__(self, graph: dict[str, Iterable[str]]):
        self.associator = MultiHopAssociator(graph)
        self.graph = CausalGraph.from_graph(graph)

    def check_causality(self, cause: str, effect: str) -> Tuple[bool, Iterable[str]]:
        path = self.associator.find_path(cause, effect)
        return bool(path), path

    def predict_effects(self, cause: str, *, depth: int = 2) -> List[Tuple[str, float]]:
        """Return downstream effects reachable from ``cause`` along with strengths."""

        return self.graph.downstream(cause, depth=depth)

    def intervention(self, cause: str, effect: str) -> str:
        """Describe the counterfactual impact of removing ``cause`` on ``effect``."""

        return self.graph.simulate_intervention(cause, effect)


class CounterfactualGraphReasoner(CounterfactualReasoner):
    """Simple counterfactual reasoning based on causal paths."""

    def __init__(self, causal_reasoner: KnowledgeGraphCausalReasoner):
        self.causal_reasoner = causal_reasoner

    def evaluate_counterfactual(self, cause: str, effect: str) -> str:
        exists, path = self.causal_reasoner.check_causality(cause, effect)
        if not exists:
            return f"{cause} has no causal effect on {effect}."
        chain = " -> ".join(path)
        intervention = self.causal_reasoner.intervention(cause, effect)
        return f"Without {cause}, {effect} would not occur via {chain}. {intervention}"

