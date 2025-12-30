from __future__ import annotations

"""Symbolic reasoning with chain-of-thought generation."""

from typing import Iterable, List, Sequence, Tuple

from .decision_engine import ActionPlan, DecisionEngine
from .multi_hop import MultiHopAssociator
from .causal import KnowledgeGraphCausalReasoner, CounterfactualGraphReasoner
from .logic import LogicProgram, LogicRule


class SymbolicReasoner:
    """Perform symbolic reasoning over a knowledge graph."""

    def __init__(
        self,
        graph: dict[str, Iterable[str]],
        decision_engine: DecisionEngine | None = None,
        logic_program: LogicProgram | None = None,
    ):
        self.graph = graph
        self.associator = MultiHopAssociator(graph)
        self.decision_engine = decision_engine or DecisionEngine()
        self.causal = KnowledgeGraphCausalReasoner(graph)
        self.counterfactual = CounterfactualGraphReasoner(self.causal)
        self.logic = logic_program or LogicProgram.from_graph(graph)

    def chain_of_thought(self, start: str, goal: str) -> List[str]:
        """Generate intermediate reasoning steps from ``start`` to ``goal``."""

        path = self.associator.find_path(start, goal)
        return [f"{path[i]} -> {path[i + 1]}" for i in range(len(path) - 1)]

    def reason(self, start: str, goal: str) -> Tuple[str, List[str]]:
        """Return the conclusion and chain-of-thought."""

        steps = self.chain_of_thought(start, goal)
        conclusion = goal if steps else start
        if steps:
            plans = [ActionPlan(action=s, utility=1.0, cost=0.0, rationale=s) for s in steps]
            # Decision engine evaluates the steps; we ignore the choice but keep integration
            self.decision_engine.select_optimal_action(plans)
            return conclusion, steps

        holds, proof = self.logic.query(goal)
        if holds:
            return goal, proof
        return conclusion, []

    # ------------------------------------------------------------------
    # Logic integration
    # ------------------------------------------------------------------
    def register_rule(self, head: str, body: Sequence[str], *, description: str = "") -> None:
        """Register a Horn-clause rule in the internal logic program."""

        self.logic.add_rule(LogicRule(head=head, body=tuple(body), description=description))

    def register_fact(self, fact: str) -> None:
        """Inject an additional fact into the logic program."""

        self.logic.add_fact(fact)

    def prove(
        self,
        goal: str,
        *,
        premises: Sequence[str] | None = None,
        rules: Sequence[LogicRule] | None = None,
    ) -> Tuple[bool, List[str]]:
        """Attempt to prove ``goal`` given optional additional premises and rules."""

        program = self.logic.temporary_context(
            additional_facts=premises,
            additional_rules=rules,
        )
        return program.query(goal)

    def detect_contradictions(self, statements: Sequence[str]) -> List[str]:
        """Return a list of contradictory statements (``x`` vs ``not x``)."""

        positives = {s for s in statements if not s.lower().startswith("not ")}
        negatives = {s[4:].strip() for s in statements if s.lower().startswith("not ")}
        conflicts = sorted(positives.intersection(negatives))
        return conflicts

    def validate_constraints(
        self,
        constraints: Sequence[str],
        *,
        premises: Sequence[str] | None = None,
        rules: Sequence[LogicRule] | None = None,
    ) -> Tuple[bool, List[str]]:
        """Check whether the supplied ``constraints`` hold under ``premises``."""

        program = self.logic.temporary_context(
            additional_facts=premises,
            additional_rules=rules,
        )
        holds = program.satisfies(constraints)
        missing = [c for c in constraints if not program.query(c)[0]]
        return holds, missing

    def explain_causality(self, cause: str, effect: str) -> Tuple[bool, Iterable[str]]:
        """Expose causal links between ``cause`` and ``effect``."""

        return self.causal.check_causality(cause, effect)

    def evaluate_counterfactual(self, cause: str, effect: str) -> str:
        """Provide a counterfactual explanation for ``effect`` if ``cause`` changes."""

        return self.counterfactual.evaluate_counterfactual(cause, effect)

