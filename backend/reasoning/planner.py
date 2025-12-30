from __future__ import annotations

"""Planning utilities for chaining reasoning steps.

This module exposes :class:`ReasoningPlanner` which can infer conclusions for
single statements via :meth:`infer` or perform batched inference over multiple
statements with :meth:`infer_batch`. Batched inference gathers evidence for all
statements concurrently and populates the planner's cache, serving as the
backend for :meth:`chain`.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Sequence, Tuple

from .interfaces import KnowledgeSource, Solver
from .decision_engine import ActionPlan, DecisionEngine
from .solvers import NeuroSymbolicSolver, RuleProbabilisticSolver


class ReasoningPlanner:
    """Plan and execute reasoning steps with caching and explanations."""

    def __init__(
        self,
        knowledge_sources: List[KnowledgeSource] | None = None,
        solver: Solver | None = None,
        solver_config: Dict[str, object] | None = None,
        decision_engine: DecisionEngine | None = None,
    ):
        self.knowledge_sources = knowledge_sources or []

        if solver is None and solver_config:
            name = str(solver_config.get("name"))
            params = solver_config.get("params", {})
            if name == "neuro_symbolic":
                solver = NeuroSymbolicSolver(**params)
            elif name == "rule_probabilistic":
                solver = RuleProbabilisticSolver(**params)

        self.solver = solver
        self.decision_engine = decision_engine
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.history: List[Dict[str, object]] = []

    def infer(self, statement: str) -> Tuple[str, float]:
        """Infer a conclusion for ``statement`` leveraging all knowledge sources."""
        if statement in self.cache:
            return self.cache[statement]
        evidence: List[str] = []
        for source in self.knowledge_sources:
            evidence.extend(source.query(statement))
        if self.solver:
            conclusion, probability = self.solver.infer(statement, evidence)
        else:
            conclusion, probability = statement, 1.0
        self.cache[statement] = (conclusion, probability)
        self.history.append(
            {
                "statement": statement,
                "conclusion": conclusion,
                "probability": probability,
                "evidence": evidence,
            }
        )
        return conclusion, probability

    def infer_batch(self, statements: Sequence[str]) -> List[Tuple[str, float]]:
        """Infer conclusions for ``statements`` concurrently.

        Evidence for all uncached statements is gathered in parallel from all
        knowledge sources. Inference is then performed (also in parallel if a
        solver is available) and results are stored in the planner cache and
        history.
        """

        ordered = list(statements)
        to_process: List[str] = []
        for stmt in ordered:
            if stmt not in self.cache:
                to_process.append(stmt)

        if to_process:
            evidence_map: Dict[str, List[str]] = {s: [] for s in to_process}

            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(source.query, stmt): (source, stmt)
                    for source in self.knowledge_sources
                    for stmt in to_process
                }
                for future, (_, stmt) in futures.items():
                    evidence_map[stmt].extend(list(future.result()))

            if self.solver:
                with ThreadPoolExecutor() as executor:
                    inf_futures = {
                        executor.submit(self.solver.infer, stmt, evidence_map[stmt]): stmt
                        for stmt in to_process
                    }
                    for future, stmt in inf_futures.items():
                        conclusion, probability = future.result()
                        self.cache[stmt] = (conclusion, probability)
                        self.history.append(
                            {
                                "statement": stmt,
                                "conclusion": conclusion,
                                "probability": probability,
                                "evidence": evidence_map[stmt],
                            }
                        )
            else:
                for stmt in to_process:
                    conclusion, probability = stmt, 1.0
                    self.cache[stmt] = (conclusion, probability)
                    self.history.append(
                        {
                            "statement": stmt,
                            "conclusion": conclusion,
                            "probability": probability,
                            "evidence": evidence_map[stmt],
                        }
                    )

        return [self.cache[s] for s in ordered]

    def chain(self, statements: Iterable[str]) -> List[Tuple[str, float]]:
        """Infer conclusions for multiple statements using batched inference."""
        return self.infer_batch(list(statements))

    def explain(self) -> List[Dict[str, object]]:
        """Return a trace of all reasoning steps performed so far."""
        return list(self.history)

    def plan(self, options: Iterable[Dict[str, float | str]]) -> Tuple[str, str]:
        """Select the optimal action among ``options`` using the decision engine.

        Each option should provide an ``action`` string and ``cost``. If ``utility``
        is omitted, it is estimated via :meth:`infer` on the action string. The
        chosen action and its rationale are returned.
        """

        if not self.decision_engine:
            raise ValueError("No decision engine available")

        plans: List[ActionPlan] = []
        for option in options:
            action = str(option["action"])
            cost = float(option.get("cost", 0))
            if "utility" in option:
                utility = float(option["utility"])
                rationale = str(option.get("rationale", ""))
            else:
                conclusion, probability = self.infer(action)
                utility = probability
                rationale = str(option.get("rationale", conclusion))
            plans.append(ActionPlan(action=action, utility=utility, cost=cost, rationale=rationale))

        return self.decision_engine.select_optimal_action(plans)
