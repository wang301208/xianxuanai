"""
Hybrid planning strategy combining symbolic reasoning with heuristic/RL scoring.

This module is intentionally lightweight yet extensible: it leverages the knowledge
graph and symbolic reasoner for constraint checking while exposing hooks for LLM or
RL-based scoring of candidate plans. When advanced backends are unavailable it falls
back to simple heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .knowledge_graph import KnowledgeGraph, KnowledgeConstraint
from .symbolic_reasoner import Rule, SymbolicReasoner
from BrainSimulationSystem.planning.rl_features import build_step_observation

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover
    PPO = None  # type: ignore[assignment]


@dataclass
class PlannerConfig:
    """Configuration for the hybrid planner."""

    max_candidates: int = 5
    rl_model_path: Optional[str] = None
    llm_enabled: bool = False  # placeholder for future integration
    heuristic_weight: float = 0.5
    sequence_enabled: bool = True
    sequence_max_depth: int = 4
    sequence_max_actions: int = 12


class HybridPlanner:
    """Generate candidate plans using reasoning plus heuristic/RL evaluation."""

    def __init__(
        self,
        knowledge: KnowledgeGraph,
        reasoner: SymbolicReasoner,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.knowledge = knowledge
        self.reasoner = reasoner
        raw_config = config or {}
        known = {f.name for f in fields(PlannerConfig)}
        filtered = {k: v for k, v in raw_config.items() if k in known}
        self.config = PlannerConfig(**filtered)
        self._rl_policy = None

        if self.config.rl_model_path and PPO is not None:
            try:  # pragma: no cover - requires trained policy
                self._rl_policy = PPO.load(self.config.rl_model_path)
            except Exception:
                self._rl_policy = None

    def generate_plan(
        self,
        context: Dict[str, Any],
        goals: Sequence[str],
        options: Sequence[Any],
        constraints: Optional[Iterable[KnowledgeConstraint]] = None,
    ) -> Dict[str, Any]:
        """Return a plan dictionary containing candidate steps and evaluations."""

        steps = self._propose_steps(context, goals, options)
        evaluations = self._evaluate_steps(steps, constraints)

        ranked = sorted(zip(steps, evaluations), key=lambda item: item[1]["score"], reverse=True)
        top_steps = [step for step, _ in ranked[: self.config.max_candidates]]
        sequence = self._build_plan_sequence(goals)

        return {
            "candidates": top_steps,
            "evaluations": [info for _, info in ranked[: self.config.max_candidates]],
            "constraints": (constraints or []),
            "sequence": sequence,
        }

    def _build_plan_sequence(self, goals: Sequence[str]) -> List[str]:
        if not self.config.sequence_enabled:
            return []

        goal_nodes = [str(goal).strip() for goal in goals if str(goal).strip()]
        if not goal_nodes:
            return []

        dependency_predicates = ("requires", "needs", "depends_on", "subtask_of", "part_of", "step")
        forward_predicates = ("precedes", "enables")

        max_depth = max(0, int(self.config.sequence_max_depth))
        max_actions = max(0, int(self.config.sequence_max_actions))

        nodes = set(goal_nodes)
        edges: set[tuple[str, str]] = set()

        queue: List[tuple[str, int]] = [(goal, 0) for goal in goal_nodes]
        visited = set(goal_nodes)
        goal_set = set(goal_nodes)

        while queue:
            node, depth = queue.pop(0)
            if max_depth and depth >= max_depth:
                continue

            # Dependency edges: node depends on obj -> obj must happen before node.
            for predicate in dependency_predicates:
                for _, __, obj in self.knowledge.query(subject=node, predicate=predicate):
                    dep = str(obj).strip()
                    if not dep or dep == node:
                        continue
                    nodes.add(dep)
                    edges.add((dep, node))
                    if dep not in visited:
                        visited.add(dep)
                        queue.append((dep, depth + 1))

            # Forward edges: predecessor -> node and node -> successor (avoid expanding beyond goals).
            for predicate in forward_predicates:
                for subj, __, ___ in self.knowledge.query(predicate=predicate, obj=node):
                    pred = str(subj).strip()
                    if not pred or pred == node:
                        continue
                    nodes.add(pred)
                    edges.add((pred, node))
                    if pred not in visited:
                        visited.add(pred)
                        queue.append((pred, depth + 1))

                if node in goal_set:
                    continue
                for _, __, obj in self.knowledge.query(subject=node, predicate=predicate):
                    succ = str(obj).strip()
                    if not succ or succ == node:
                        continue
                    nodes.add(succ)
                    edges.add((node, succ))
                    if succ not in visited:
                        visited.add(succ)
                        queue.append((succ, depth + 1))

        ordered = self._toposort(nodes, edges)
        if max_actions and len(ordered) > max_actions:
            ordered = ordered[-max_actions:]
        return ordered

    @staticmethod
    def _toposort(nodes: set[str], edges: set[tuple[str, str]]) -> List[str]:
        indegree: Dict[str, int] = {n: 0 for n in nodes}
        outgoing: Dict[str, set[str]] = {n: set() for n in nodes}
        for src, dst in edges:
            if src not in nodes or dst not in nodes or src == dst:
                continue
            if dst in outgoing[src]:
                continue
            outgoing[src].add(dst)
            indegree[dst] = indegree.get(dst, 0) + 1

        ready = sorted([n for n, deg in indegree.items() if deg == 0])
        result: List[str] = []
        while ready:
            node = ready.pop(0)
            result.append(node)
            for dep in sorted(outgoing.get(node, set())):
                indegree[dep] = indegree.get(dep, 0) - 1
                if indegree[dep] == 0:
                    ready.append(dep)
            ready.sort()

        # Cycle fallback: preserve as much deterministic ordering as possible.
        if len(result) < len(nodes):
            remaining = sorted(nodes.difference(result))
            result.extend(remaining)
        return result

    def _propose_steps(
        self,
        context: Dict[str, Any],
        goals: Sequence[str],
        options: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []

        for option in options:
            step = {
                "action": option,
                "justification": [],
            }
            for goal in goals:
                query = self.reasoner.query(subject=str(option), predicate="supports", obj=str(goal))
                if query:
                    step["justification"].append(f"{option} supports {goal}")
            if not step["justification"]:
                step["justification"].append("fallback")
            steps.append(step)

        if not steps and goals:
            for goal in goals:
                related = self.knowledge.query(predicate="enables", obj=str(goal))
                for subject, _, _ in related:
                    steps.append({"action": subject, "justification": [f"{subject} enables {goal}"]})

        if not steps:
            steps.append({"action": "reflect", "justification": ["no viable options"]})

        return steps

    def _evaluate_steps(
        self,
        steps: List[Dict[str, Any]],
        constraints: Optional[Iterable[KnowledgeConstraint]],
    ) -> List[Dict[str, Any]]:
        evaluations: List[Dict[str, Any]] = []

        for step in steps:
            heuristic_score = self._heuristic_score(step)
            constraint_info = self.knowledge.evaluate_action_constraints(
                step.get("action"),
                constraints or [],
            )
            violation_count = len(constraint_info["violations"])
            constraint_penalty = 0.0 if constraint_info["satisfied"] else 0.2 + 0.1 * violation_count
            metadata = {
                "heuristic_score": heuristic_score,
                "constraint_penalty": constraint_penalty,
                "total_steps": len(steps),
                "goal_matches": sum(
                    1 for justification in step.get("justification", []) if "supports" in justification
                ),
            }
            observation = build_step_observation(step, constraint_info, metadata)
            rl_score = self._rl_score(observation) if self._rl_policy else 0.0

            score = (
                self.config.heuristic_weight * heuristic_score
                + (1 - self.config.heuristic_weight) * rl_score
                - constraint_penalty
            )
            evaluations.append(
                {
                    "action": step["action"],
                    "score": float(score),
                    "heuristic": float(heuristic_score),
                    "rl_score": float(rl_score),
                    "constraint_satisfied": constraint_info["satisfied"],
                    "constraint_violations": constraint_info["violations"],
                    "constraint_details": constraint_info["details"],
                    "observation": observation.tolist(),
                }
            )
        return evaluations

    def _heuristic_score(self, step: Dict[str, Any]) -> float:
        justifications = step.get("justification", [])
        base = min(len(justifications), 3) * 0.2
        if step["action"] == "reflect":
            base -= 0.1
        return base

    def _rl_score(self, observation: np.ndarray) -> float:  # pragma: no cover - requires RL model
        if self._rl_policy is None:
            return 0.0
        observation = np.asarray(observation, dtype=np.float32)
        action, _ = self._rl_policy.predict(observation, deterministic=True)
        return float(np.mean(action))
