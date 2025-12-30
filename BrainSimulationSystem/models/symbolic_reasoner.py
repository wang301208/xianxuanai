"""
Lightweight symbolic reasoning engine.

Supports rule definition, forward chaining, and simple plan validation checks.
Rules are expressed as implications over predicates (represented as triples).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .knowledge_graph import KnowledgeGraph, Triple


Predicate = Triple


@dataclass
class Rule:
    """A simple Horn-like rule: if all antecedents hold, infer the consequent."""

    name: str
    antecedents: List[Predicate] = field(default_factory=list)
    consequent: Predicate = field(default_factory=lambda: ("", "", ""))


class SymbolicReasoner:
    """Minimal symbolic reasoner that works atop the knowledge graph."""

    def __init__(self, knowledge: Optional[KnowledgeGraph] = None) -> None:
        self.knowledge = knowledge or KnowledgeGraph()
        self.rules: Dict[str, Rule] = {}

    def add_rule(self, rule: Rule) -> None:
        self.rules[rule.name] = rule

    def infer(self, iterations: int = 10) -> Set[Predicate]:
        """
        Perform forward chaining for a limited number of iterations.

        Returns the set of newly inferred predicates.
        """
        inferred: Set[Predicate] = set()
        for _ in range(iterations):
            progress = False
            for rule in self.rules.values():
                if rule.consequent in self.knowledge.triples or rule.consequent in inferred:
                    continue
                if all(pred in self.knowledge.triples for pred in rule.antecedents):
                    inferred.add(rule.consequent)
                    self.knowledge.add(*rule.consequent)
                    progress = True
            if not progress:
                break
        return inferred

    def explain(self, predicate: Predicate) -> Optional[List[str]]:
        """
        Provide a naive explanation path for how a predicate was derived.
        Returns a list of rule names used, or None if not derivable via stored rules.
        """
        if predicate in self.knowledge.triples:
            for rule in self.rules.values():
                if rule.consequent == predicate:
                    satisfied = [pred in self.knowledge.triples for pred in rule.antecedents]
                    if all(satisfied):
                        return [rule.name]
            return []
        return None

    def validate_plan(self, plan_steps: Iterable[Predicate], constraints: Iterable[Predicate]) -> Dict[str, bool]:
        """
        Validate each plan step and constraint against the current knowledge graph.
        Returns mapping predicate->bool indicating whether each requirement holds.
        """
        results: Dict[str, bool] = {}
        for predicate in plan_steps:
            results[str(predicate)] = predicate in self.knowledge.triples
        for predicate in constraints:
            results[f"constraint:{predicate}"] = predicate in self.knowledge.triples
        return results

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None):
        return self.knowledge.query(subject=subject, predicate=predicate, obj=obj)

    def to_dict(self) -> Dict[str, any]:
        return {
            "knowledge": self.knowledge.to_dict(),
            "rules": [
                {
                    "name": rule.name,
                    "antecedents": list(rule.antecedents),
                    "consequent": rule.consequent,
                }
                for rule in self.rules.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "SymbolicReasoner":
        knowledge = KnowledgeGraph.from_dict(data.get("knowledge", {}))
        reasoner = cls(knowledge)
        for rule_data in data.get("rules", []):
            reasoner.add_rule(
                Rule(
                    name=rule_data.get("name", "rule"),
                    antecedents=[tuple(a) for a in rule_data.get("antecedents", [])],
                    consequent=tuple(rule_data.get("consequent", ("", "", ""))),
                )
            )
        return reasoner
