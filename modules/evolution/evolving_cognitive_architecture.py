"""Tools for evolving simple cognitive architectures.

This module defines :class:`EvolvingCognitiveArchitecture` which uses a small
GeneticAlgorithm implementation to iteratively improve a representation of an
"architecture".  The architecture is modelled as a mapping from parameter names
to floating point values.  Evolution proceeds by mutating the best known
architecture and keeping track of performance feedback for each generation.

The implementation is intentionally lightweight and self contained so it can be
used in environments without heavy dependencies.  It is not meant to be a
state-of-the-art optimiser but rather a minimal example that demonstrates how
an evolutionary process can be integrated in the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING
import random

if TYPE_CHECKING:  # pragma: no cover - optional controller
    from .meta_nas import MetaNASController


@dataclass
class GAConfig:
    """Configuration parameters for :class:`GeneticAlgorithm`."""

    population_size: int = 20
    generations: int = 5
    mutation_rate: float = 0.3
    mutation_sigma: float = 0.1


class GeneticAlgorithm:
    """A tiny genetic algorithm working on dictionaries of floats.

    The algorithm keeps the best individual seen so far and generates each new
    population by mutating that individual.  This hill-climbing approach keeps
    the implementation small while still yielding consistent improvements for
    smooth fitness landscapes.
    """

    def __init__(
        self,
        fitness_fn: Callable[[Dict[str, float]], float],
        config: GAConfig | None = None,
        seed: int | None = None,
        post_mutation: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.config = config or GAConfig()
        self._rng = random.Random(seed) if seed is not None else random
        self._post_mutation = post_mutation

    # ------------------------------------------------------------------
    def _mutate(self, individual: Dict[str, float]) -> None:
        for key in individual:
            if self._rng.random() < self.config.mutation_rate:
                individual[key] += self._rng.gauss(0.0, self.config.mutation_sigma)
        if self._post_mutation is not None:
            self._post_mutation(individual)

    # ------------------------------------------------------------------
    def evolve(
        self, seed: Dict[str, float]
    ) -> Tuple[Dict[str, float], float, List[Tuple[Dict[str, float], float]]]:
        """Evolve a new architecture starting from ``seed``.

        Returns the best architecture, its fitness and a history of evaluated
        individuals for introspection.
        """

        best = seed.copy()
        best_score = self.fitness_fn(best)
        history: List[Tuple[Dict[str, float], float]] = [(best.copy(), best_score)]

        # Initial population: mutated copies of the seed.
        population: List[Dict[str, float]] = []
        for _ in range(self.config.population_size):
            ind = seed.copy()
            self._mutate(ind)
            population.append(ind)

        for _ in range(self.config.generations):
            scored = [(ind, self.fitness_fn(ind)) for ind in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            best_candidate, best_candidate_score = scored[0]
            history.append((best_candidate.copy(), best_candidate_score))
            if best_candidate_score > best_score:
                best, best_score = best_candidate.copy(), best_candidate_score
            # Create next generation by mutating the current best candidate.
            population = []
            for _ in range(self.config.population_size):
                child = best.copy()
                self._mutate(child)
                population.append(child)

        return best, best_score, history


class EvolvingCognitiveArchitecture:
    """Maintain and evolve a simple architecture using a genetic algorithm."""

    def __init__(
        self,
        fitness_fn: Callable[[Dict[str, float]], float],
        ga: GeneticAlgorithm | None = None,
        post_mutation: Optional[Callable[[Dict[str, float]], None]] = None,
        nas_controller: Optional["MetaNASController"] = None,
    ) -> None:
        self.ga = ga or GeneticAlgorithm(fitness_fn, post_mutation=post_mutation)
        self.fitness_fn = fitness_fn
        self.history: List[Dict[str, float] | Tuple[Dict[str, float], float]] = []
        self.nas_controller = nas_controller

    # ------------------------------------------------------------------
    def evolve_architecture(
        self, architecture: Dict[str, float], performance_feedback: float
    ) -> Dict[str, float]:
        """Evolve ``architecture`` using the supplied ``performance_feedback``.

        The current architecture and feedback are recorded in ``history``.  A
        new architecture is generated by running the genetic algorithm and the
        evaluations performed by the GA are appended to the history as well.
        The best architecture discovered by the GA is returned.
        """

        # Record the current state before evolving.
        self.history.append({**architecture, "performance": performance_feedback})

        if self.nas_controller is not None:
            try:
                from .mutation_operators import MutationContext

                ctx = MutationContext(extra={"performance_feedback": float(performance_feedback)})
                best, best_score, nas_history = self.nas_controller.search(
                    architecture,
                    self.fitness_fn,
                    context=ctx,
                )
                for candidate in nas_history:
                    self.history.append(
                        {
                            **candidate.genes,
                            "performance": float(candidate.score),
                            "mutation": str(candidate.operator),
                            "generation": float(candidate.generation),
                            "reward": float(candidate.reward),
                        }
                    )
                self.history.append({**best, "performance": float(best_score), "mutation": "meta_nas_best"})
                return best
            except Exception:
                # Fall back to baseline GA when the controller fails.
                pass

        best, _, ga_history = self.ga.evolve(architecture)
        # Extend the evolution history with GA evaluations.
        for arch, score in ga_history:
            self.history.append({**arch, "performance": score})
        return best
