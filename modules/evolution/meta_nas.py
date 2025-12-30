"""Meta-NAS controller: NAS + meta-learning (bandit) for operator selection.

This module implements a lightweight NAS loop that:
- generates candidate architectures via a mutation operator library
- evaluates them with a supplied fitness function
- uses a multi-armed bandit to *learn* which mutation operators are effective

The intent is to complement the existing GA-based approach with a controller
that can adapt its mutation strategy over time (meta-learning).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .mutation_operators import (
    MutationContext,
    MutationOperatorLibrary,
    default_operator_library,
)


@dataclass
class BanditArmStats:
    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.pulls <= 0:
            return 0.0
        return self.reward_sum / float(self.pulls)


class UCBBanditSelector:
    """UCB1 bandit for selecting mutation operators."""

    def __init__(
        self,
        arms: Sequence[str],
        *,
        exploration_c: float = 1.4,
    ) -> None:
        if not arms:
            raise ValueError("arms must be non-empty")
        self._arms = sorted({str(a) for a in arms})
        self.exploration_c = float(exploration_c)
        self.stats: Dict[str, BanditArmStats] = {name: BanditArmStats() for name in self._arms}
        self.total_pulls = 0

    def select(self) -> str:
        # Cold start: try each arm once (deterministic order).
        for name in self._arms:
            if self.stats[name].pulls == 0:
                return name

        total = max(1, self.total_pulls)
        log_total = math.log(float(total))
        best_name = self._arms[0]
        best_ucb = -float("inf")
        for name in self._arms:
            arm = self.stats[name]
            mean = arm.mean_reward
            bonus = self.exploration_c * math.sqrt(log_total / float(arm.pulls))
            ucb = mean + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_name = name
        return best_name

    def update(self, name: str, reward: float) -> None:
        key = str(name)
        if key not in self.stats:
            raise KeyError(f"Unknown arm '{name}'")
        arm = self.stats[key]
        arm.pulls += 1
        arm.reward_sum += float(reward)
        self.total_pulls += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_pulls": int(self.total_pulls),
            "arms": {
                name: {"pulls": int(stat.pulls), "mean_reward": float(stat.mean_reward)}
                for name, stat in self.stats.items()
            },
        }


@dataclass(frozen=True)
class NASCandidate:
    generation: int
    operator: str
    score: float
    reward: float
    genes: Dict[str, float]


class MetaNASController:
    """Search controller that combines NAS operator library with a bandit."""

    def __init__(
        self,
        *,
        operator_library: MutationOperatorLibrary | None = None,
        bandit: UCBBanditSelector | None = None,
        postprocess: Optional[Callable[[Dict[str, float]], None]] = None,
        seed: int | None = None,
        population_size: int = 16,
        generations: int = 4,
        reward_baseline: str = "best",  # "best" or "parent"
    ) -> None:
        self.operator_library = operator_library or default_operator_library()
        self.bandit = bandit or UCBBanditSelector(self.operator_library.names())
        self.postprocess = postprocess
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self.population_size = max(1, int(population_size))
        self.generations = max(1, int(generations))
        self.reward_baseline = str(reward_baseline or "best")

    def search(
        self,
        seed_genome: Mapping[str, float],
        fitness_fn: Callable[[Dict[str, float]], float],
        *,
        context: MutationContext | None = None,
    ) -> Tuple[Dict[str, float], float, List[NASCandidate]]:
        """Return (best_genome, best_score, history)."""

        parent = dict(seed_genome)
        if self.postprocess is not None:
            try:
                self.postprocess(parent)
            except Exception:
                pass
        best = dict(parent)
        best_score = float(fitness_fn(best))
        history: List[NASCandidate] = []

        for gen in range(self.generations):
            for _ in range(self.population_size):
                op_name = self.bandit.select()
                candidate = self.operator_library.mutate(parent, name=op_name, rng=self._rng, context=context)
                if self.postprocess is not None:
                    try:
                        self.postprocess(candidate)
                    except Exception:
                        pass
                score = float(fitness_fn(candidate))
                if self.reward_baseline == "parent":
                    baseline = float(fitness_fn(parent))
                else:
                    baseline = best_score
                reward = score - baseline
                self.bandit.update(op_name, reward)
                history.append(
                    NASCandidate(
                        generation=int(gen),
                        operator=str(op_name),
                        score=float(score),
                        reward=float(reward),
                        genes=dict(candidate),
                    )
                )
                if score > best_score:
                    best = dict(candidate)
                    best_score = float(score)

            parent = dict(best)

        return best, best_score, history


__all__ = [
    "BanditArmStats",
    "UCBBanditSelector",
    "NASCandidate",
    "MetaNASController",
]

