from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Mapping

from modules.evolution.meta_nas import MetaNASController, UCBBanditSelector
from modules.evolution.mutation_operators import MutationContext, MutationOperator, MutationOperatorLibrary
from modules.evolution.evolving_cognitive_architecture import EvolvingCognitiveArchitecture


class _SetXOperator(MutationOperator):
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self._value = float(value)

    def mutate(self, genome: Mapping[str, float], *, rng: random.Random, context: MutationContext | None = None) -> Dict[str, float]:
        out = dict(genome)
        out["x"] = float(self._value)
        return out


def test_ucb_bandit_tries_each_arm_then_prefers_higher_reward():
    bandit = UCBBanditSelector(["bad", "good"], exploration_c=0.0)
    assert bandit.select() == "bad"
    bandit.update("bad", reward=-1.0)
    assert bandit.select() == "good"
    bandit.update("good", reward=1.0)
    # exploration_c=0 -> always exploit.
    assert bandit.select() == "good"


def test_meta_nas_controller_selects_better_operator_and_improves_score():
    ops = MutationOperatorLibrary([_SetXOperator("bad", -1.0), _SetXOperator("good", 1.0)])
    bandit = UCBBanditSelector(ops.names(), exploration_c=0.0)
    controller = MetaNASController(
        operator_library=ops,
        bandit=bandit,
        seed=0,
        population_size=2,
        generations=2,
        reward_baseline="best",
    )

    def fitness_fn(genome: Dict[str, float]) -> float:
        # maximum at x == 1
        x = float(genome.get("x", 0.0))
        return -abs(x - 1.0)

    best, best_score, history = controller.search({"x": 0.0}, fitness_fn, context=MutationContext())
    assert best["x"] == 1.0
    assert best_score == 0.0
    assert history


def test_evolving_cognitive_architecture_can_use_meta_nas_controller():
    ops = MutationOperatorLibrary([_SetXOperator("good", 1.0)])
    controller = MetaNASController(operator_library=ops, seed=0, population_size=1, generations=1)

    def fitness_fn(genome: Dict[str, float]) -> float:
        return float(genome.get("x", 0.0))

    evolver = EvolvingCognitiveArchitecture(fitness_fn, nas_controller=controller)
    best = evolver.evolve_architecture({"x": 0.0}, performance_feedback=0.0)
    assert best["x"] == 1.0
    # The controller path annotates history with mutation tags.
    assert any(isinstance(item, dict) and item.get("mutation") == "meta_nas_best" for item in evolver.history)

