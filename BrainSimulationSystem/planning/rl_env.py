"""Gym environment for training HybridPlanner reinforcement learning policies."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import gym

    _GYMNASIUM_STYLE = gym.__name__ == "gymnasium"
except Exception:  # pragma: no cover
    import gymnasium as gym  # type: ignore

    _GYMNASIUM_STYLE = True

from BrainSimulationSystem.models.hybrid_planner import HybridPlanner
from BrainSimulationSystem.models.knowledge_graph import KnowledgeConstraint, KnowledgeGraph
from BrainSimulationSystem.models.symbolic_reasoner import SymbolicReasoner
from BrainSimulationSystem.planning.rl_features import PLANNER_OBSERVATION_DIM, build_step_observation


class PlannerRankingEnv(gym.Env):
    """Environment that scores planner candidates and rewards correct ranking."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        planner_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        max_options: int = 6,
    ) -> None:
        super().__init__()
        self._seed_value = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._rng = np.random.default_rng(self._seed_value)
        self.max_options = max(2, max_options)

        planner_cfg = deepcopy(planner_config or {})
        planner_cfg["rl_model_path"] = None  # ensure no policy is loaded during training
        self._planner_cfg = planner_cfg
        self.knowledge = KnowledgeGraph()
        self.reasoner = SymbolicReasoner(self.knowledge)
        self.planner = HybridPlanner(self.knowledge, self.reasoner, planner_cfg)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(PLANNER_OBSERVATION_DIM,),
            dtype=np.float32,
        )

        self.goals: List[str] = []
        self.options: List[str] = []
        self.constraints: List[KnowledgeConstraint] = []
        self.steps: List[Dict[str, Any]] = []
        self.observations: List[np.ndarray] = []
        self.target_scores: List[float] = []
        self.best_index: int = 0
        self.predicted_scores: List[float] = []
        self.current_index: int = 0

        self._reset_episode()

    def reset(  # type: ignore[override]
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        if seed is not None:
            self._seed_value = int(seed)
            self._rng = np.random.default_rng(self._seed_value)
        obs = self._reset_episode()
        if _GYMNASIUM_STYLE:
            return obs, {}
        return obs

    def step(self, action):  # type: ignore[override]
        score = float(np.asarray(action).mean())
        self.predicted_scores.append(score)
        self.current_index += 1

        terminated = self.current_index >= len(self.observations)
        reward = 0.0

        if terminated:
            if self.predicted_scores:
                best_guess = int(np.argmax(self.predicted_scores))
            else:
                best_guess = 0
            reward = 1.0 if best_guess == self.best_index else -1.0
            obs = np.zeros(PLANNER_OBSERVATION_DIM, dtype=np.float32)
        else:
            obs = self.observations[self.current_index]

        info: Dict[str, Any] = {
            "best_index": self.best_index,
            "options": list(self.options),
            "goals": list(self.goals),
        }

        if _GYMNASIUM_STYLE:
            return obs, reward, terminated, False, info
        return obs, reward, terminated, info

    # ------------------------------------------------------------------ #
    # Episode generation
    # ------------------------------------------------------------------ #
    def _reset_episode(self) -> np.ndarray:
        self.knowledge = KnowledgeGraph()
        self.reasoner = SymbolicReasoner(self.knowledge)
        planner_cfg = deepcopy(self._planner_cfg)
        planner_cfg["rl_model_path"] = None
        self.planner = HybridPlanner(self.knowledge, self.reasoner, planner_cfg)

        goal_count = int(self._rng.integers(1, 4))
        option_count = int(self._rng.integers(2, self.max_options + 1))

        self.goals = [f"goal_{i}" for i in range(goal_count)]
        self.options = [f"action_{i}" for i in range(option_count)]
        self.constraints = self._sample_constraints()

        self._populate_knowledge()

        context = {"urgency": float(self._rng.random()), "resource_level": float(self._rng.random())}
        self.steps = self.planner._propose_steps(context, self.goals, self.options)

        if not self.steps:
            self.steps = [{"action": "reflect", "justification": ["no viable options"]}]

        self.observations = []
        self.target_scores = []
        total_steps = len(self.steps)

        for step in self.steps:
            constraint_info = self.knowledge.evaluate_action_constraints(step.get("action"), self.constraints)
            heuristic_score = self.planner._heuristic_score(step)
            violation_count = len(constraint_info["violations"])
            constraint_penalty = 0.0 if constraint_info["satisfied"] else 0.2 + 0.1 * violation_count
            score = self.planner.config.heuristic_weight * heuristic_score - constraint_penalty

            metadata = {
                "heuristic_score": heuristic_score,
                "constraint_penalty": constraint_penalty,
                "total_steps": total_steps,
                "goal_matches": _count_goal_matches(step, self.goals),
                "goal_count": len(self.goals),
            }
            obs = build_step_observation(step, constraint_info, metadata)
            self.observations.append(obs)
            self.target_scores.append(float(score))

        if self.target_scores:
            self.best_index = int(np.argmax(self.target_scores))
        else:
            self.best_index = 0

        self.predicted_scores = []
        self.current_index = 0
        return self.observations[0]

    def _populate_knowledge(self) -> None:
        for option in self.options:
            supported_goals = self._rng.choice([True, False], size=len(self.goals))
            for goal, supported in zip(self.goals, supported_goals):
                if supported or self._rng.random() < 0.3:
                    self.knowledge.add(option, "supports", goal)

            if self._rng.random() < 0.4:
                target_goal = self._rng.choice(self.goals)
                self.knowledge.add(option, "enables", target_goal)

            if self._rng.random() < 0.3:
                conflict = f"conflict_{self._rng.integers(0, 5)}"
                self.knowledge.add(option, "conflicts", conflict)

    def _sample_constraints(self) -> List[KnowledgeConstraint]:
        constraints: List[KnowledgeConstraint] = []
        num_constraints = int(self._rng.integers(0, 3))

        for _ in range(num_constraints):
            goal = self._rng.choice(self.goals) if self.goals else "goal_0"
            description = f"require_support_{goal}"
            required = [( "{action}", "supports", goal )]
            forbidden: List[Tuple[str, str, str]] = []
            if self._rng.random() < 0.5:
                forbidden_conflict = f"conflict_{self._rng.integers(0, 5)}"
                forbidden.append(("{action}", "conflicts", forbidden_conflict))
            constraints.append(KnowledgeConstraint(description=description, required=required, forbidden=forbidden))

        return constraints


def _count_goal_matches(step: Dict[str, Any], goals: List[str]) -> int:
    justifications = step.get("justification", [])
    count = 0
    for entry in justifications:
        for goal in goals:
            if goal in entry:
                count += 1
    return count
