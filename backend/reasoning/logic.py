"""Lightweight logic programming utilities for symbolic reasoning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple

Graph = Dict[str, Iterable[str]]


@dataclass(frozen=True)
class LogicRule:
    """A Horn-clause style implication ``body -> head``."""

    head: str
    body: Tuple[str, ...]
    description: str = ""

    def is_applicable(self, known: Set[str]) -> bool:
        return all(atom in known for atom in self.body)


class LogicProgram:
    """Very small forward chaining engine used by :class:`SymbolicReasoner`."""

    def __init__(self, *, max_iterations: int = 32) -> None:
        self._facts: Set[str] = set()
        self._rules: List[LogicRule] = []
        self._max_iterations = max(1, max_iterations)
        self._proofs: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def add_fact(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        self._facts.add(fact)
        self._proofs.setdefault(fact, [f"fact: {fact}"])

    def add_facts(self, facts: Iterable[str]) -> None:
        for fact in facts:
            self.add_fact(fact)

    def add_rule(self, rule: LogicRule) -> None:
        self._rules.append(rule)

    def add_rules(self, rules: Iterable[LogicRule]) -> None:
        for rule in rules:
            self.add_rule(rule)

    def clone(self) -> "LogicProgram":
        clone = LogicProgram(max_iterations=self._max_iterations)
        clone._facts = set(self._facts)
        clone._rules = list(self._rules)
        clone._proofs = {k: list(v) for k, v in self._proofs.items()}
        return clone

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer(self) -> Dict[str, List[str]]:
        """Run forward chaining and return mapping ``fact -> proof steps``."""

        known = set(self._facts)
        proofs = {atom: list(path) for atom, path in self._proofs.items()}
        iterations = 0
        changed = True
        while changed and iterations < self._max_iterations:
            changed = False
            iterations += 1
            for rule in self._rules:
                if rule.head in known:
                    continue
                if rule.is_applicable(known):
                    known.add(rule.head)
                    description = rule.description or " & ".join(rule.body)
                    explanation: List[str] = [
                        f"applied: {description} -> {rule.head}"
                    ]
                    for atom in rule.body:
                        explanation.extend(proofs.get(atom, [atom]))
                    proofs[rule.head] = explanation
                    changed = True
        return proofs

    def query(self, goal: str) -> Tuple[bool, List[str]]:
        goal = goal.strip()
        if not goal:
            return False, []
        proofs = self.infer()
        if goal in proofs:
            return True, proofs[goal]
        return False, []

    def satisfies(self, hypothesis: Sequence[str]) -> bool:
        proofs = self.infer()
        return all(atom in proofs for atom in hypothesis)

    # ------------------------------------------------------------------
    # Graph ingestion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def fact_from_edge(source: str, target: str, relation: str = "rel") -> str:
        return f"{relation}({source},{target})"

    @classmethod
    def from_graph(
        cls,
        graph: Graph,
        *,
        relation: str = "rel",
        max_iterations: int = 32,
    ) -> "LogicProgram":
        program = cls(max_iterations=max_iterations)
        for source, targets in graph.items():
            for target in targets:
                program.add_fact(cls.fact_from_edge(source, target, relation))
        return program

    def extend_from_graph(self, graph: Graph, *, relation: str = "rel") -> None:
        for source, targets in graph.items():
            for target in targets:
                self.add_fact(self.fact_from_edge(source, target, relation))

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------
    def temporary_context(
        self,
        *,
        additional_facts: Iterable[str] | None = None,
        additional_rules: Iterable[LogicRule] | None = None,
    ) -> "LogicProgram":
        clone = self.clone()
        if additional_facts:
            clone.add_facts(additional_facts)
        if additional_rules:
            clone.add_rules(additional_rules)
        return clone


def iter_graph_edges(graph: Graph) -> Iterator[Tuple[str, str]]:
    for source, targets in graph.items():
        for target in targets:
            yield source, target
