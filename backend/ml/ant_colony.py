"""Basic Ant Colony Optimization (ACO) implementation.

This module provides a simple ACO algorithm for solving small Travelling
Salesman Problems (TSP). It follows the classic approach with pheromone
initialisation, probabilistic solution construction, local pheromone updates
and global updates using the best solution of each iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class ACOParameters:
    """Parameters controlling the behaviour of the ACO algorithm."""

    alpha: float = 1.0  # Influence of pheromone values
    beta: float = 2.0   # Influence of heuristic information
    rho: float = 0.5    # Pheromone evaporation rate
    q: float = 1.0      # Constant used in global update
    tau0: float | None = None  # Initial pheromone level
    seed: int | None = None


class AntColony:
    """Ant Colony Optimization for symmetric TSP instances."""

    def __init__(
        self,
        distance_matrix: Sequence[Sequence[float]],
        n_ants: int,
        n_iterations: int,
        params: ACOParameters | None = None,
    ) -> None:
        self.distance = np.asarray(distance_matrix, dtype=float)
        if self.distance.shape[0] != self.distance.shape[1]:
            raise ValueError("distance_matrix must be square")
        self.n_nodes = self.distance.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.params = params or ACOParameters()
        self.rng = np.random.default_rng(self.params.seed)

        # Initialise pheromone and heuristic matrices
        self.tau0 = (
            self.params.tau0
            if self.params.tau0 is not None
            else 1.0 / (self.distance.mean() * self.n_nodes)
        )
        self.pheromone = np.full_like(self.distance, self.tau0, dtype=float)
        with np.errstate(divide="ignore"):
            self.heuristic = 1.0 / self.distance
        self.heuristic[self.distance == 0] = 0.0

    # ------------------------------------------------------------------
    # Solution construction
    # ------------------------------------------------------------------
    def _choose_next_node(self, current: int, unvisited: set[int]) -> int:
        options = list(unvisited)
        tau = self.pheromone[current, options] ** self.params.alpha
        eta = self.heuristic[current, options] ** self.params.beta
        probs = tau * eta
        total = probs.sum()
        if total <= 0:
            return int(self.rng.choice(options))
        return int(self.rng.choice(options, p=probs / total))

    def _construct_solution(self, start: int) -> Tuple[List[int], float]:
        path = [start]
        unvisited = set(range(self.n_nodes))
        unvisited.remove(start)
        cost = 0.0
        current = start
        while unvisited:
            nxt = self._choose_next_node(current, unvisited)
            cost += self.distance[current, nxt]
            self._local_update(current, nxt)
            current = nxt
            path.append(current)
            unvisited.remove(current)
        cost += self.distance[current, start]
        self._local_update(current, start)
        path.append(start)
        return path, cost

    # ------------------------------------------------------------------
    # Pheromone updates
    # ------------------------------------------------------------------
    def _local_update(self, i: int, j: int) -> None:
        new_value = (1.0 - self.params.rho) * self.pheromone[i, j] + self.params.rho * self.tau0
        self.pheromone[i, j] = self.pheromone[j, i] = new_value

    def _global_update(self, best_path: List[int], best_cost: float) -> None:
        self.pheromone *= 1.0 - self.params.rho
        deposit = self.params.q / best_cost
        for a, b in zip(best_path[:-1], best_path[1:]):
            self.pheromone[a, b] += deposit
            self.pheromone[b, a] += deposit

    # ------------------------------------------------------------------
    def run(self) -> Tuple[List[int], float]:
        """Execute the algorithm and return the best path and its cost."""

        best_path: List[int] | None = None
        best_cost = float("inf")
        for _ in range(self.n_iterations):
            for _ in range(self.n_ants):
                start = int(self.rng.integers(self.n_nodes))
                path, cost = self._construct_solution(start)
                if cost < best_cost:
                    best_path, best_cost = path, cost
            if best_path is not None:
                self._global_update(best_path, best_cost)
        assert best_path is not None
        return best_path, best_cost


__all__ = ["ACOParameters", "AntColony"]
