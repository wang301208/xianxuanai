"""Feature engineering utilities for HybridPlanner reinforcement learning."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

PLANNER_OBSERVATION_DIM = 12


def _count_matching(justifications: Iterable[str], keyword: str) -> int:
    keyword_lower = keyword.lower()
    return sum(1 for entry in justifications if keyword_lower in entry.lower())


def build_step_observation(
    step: Dict[str, Any],
    constraint_info: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Build a fixed-length observation vector describing a planner step.

    Parameters
    ----------
    step:
        Planner step dictionary with keys such as ``action`` and ``justification``.
    constraint_info:
        Result from :meth:`KnowledgeGraph.evaluate_action_constraints`.
    metadata:
        Additional scalar hints (e.g. heuristic score, goal counts) gathered during evaluation.

    Returns
    -------
    numpy.ndarray
        A 12-dimensional feature vector normalised to [-1, 1] where possible.
    """

    metadata = metadata or {}
    justifications = list(step.get("justification", []))
    justification_count = len(justifications)
    supported = _count_matching(justifications, "supports")
    enables = _count_matching(justifications, "enables")
    fallback_flag = 1.0 if any("fallback" in j.lower() for j in justifications) else 0.0

    constraint_info = constraint_info or {}
    satisfied = 1.0 if constraint_info.get("satisfied", True) else 0.0
    violations = constraint_info.get("violations", []) or []
    violation_count = float(len(violations))

    details = constraint_info.get("details", {}) or {}
    if details:
        coverage_vals = []
        for entry in details.values():
            required_met = 1.0 if entry.get("required_met", False) else 0.0
            forbidden_clear = 1.0 if entry.get("forbidden_clear", False) else 0.0
            coverage_vals.append(0.5 * (required_met + forbidden_clear))
        constraint_coverage = float(np.mean(coverage_vals))
    else:
        constraint_coverage = 1.0

    heuristic_score = float(metadata.get("heuristic_score", 0.0))
    constraint_penalty = float(metadata.get("constraint_penalty", 0.0))
    total_steps = int(metadata.get("total_steps", 1))
    goal_matches = int(metadata.get("goal_matches", 0))
    goal_count = int(metadata.get("goal_count", max(goal_matches, 1)))

    action = str(step.get("action", ""))
    action_length_norm = min(len(action), 48) / 48.0 if action else 0.0

    observation = np.zeros(PLANNER_OBSERVATION_DIM, dtype=np.float32)
    observation[0] = min(justification_count, 8) / 8.0
    observation[1] = supported / max(1, justification_count)
    observation[2] = enables / max(1, justification_count)
    observation[3] = fallback_flag
    observation[4] = satisfied
    observation[5] = min(violation_count, 4.0) / 4.0
    observation[6] = float(np.clip(constraint_coverage, 0.0, 1.0))
    observation[7] = float(np.clip(heuristic_score / 0.6, -1.0, 1.0))
    observation[8] = float(action_length_norm)
    observation[9] = min(total_steps, 8) / 8.0
    observation[10] = goal_matches / max(1, goal_count)
    observation[11] = float(np.clip(constraint_penalty, 0.0, 1.0))

    observation = np.clip(observation, -1.0, 1.0)
    return observation
