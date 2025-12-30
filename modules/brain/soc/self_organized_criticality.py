"""Self-organized criticality models for adaptive neural networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AvalancheDynamics:
    """Handle avalanche propagation in a network.

    The model is intentionally simple: when a neuron's activity exceeds the
    threshold it ``topples`` and distributes a fixed amount of activity to its
    neighbours according to the provided weight matrix. This process continues
    recursively until all neurons are below threshold.
    """

    threshold: float = 1.0

    def propagate(self, state: np.ndarray, weights: np.ndarray) -> int:
        """Propagate activity through the network.

        Parameters
        ----------
        state: np.ndarray
            Current activity state of each node.
        weights: np.ndarray
            Connection weight matrix. Rows should sum to the desired branching
            ratio. ``weights[i, j]`` is the proportion of activity sent from
            node ``i`` to ``j`` when ``i`` topples.

        Returns
        -------
        int
            Total number of topplings that occurred (i.e. avalanche size).
        """

        avalanche_size = 0
        active = list(np.where(state >= self.threshold)[0])
        while active:
            idx = active.pop()
            # Topple node
            avalanche_size += 1
            state[idx] -= self.threshold
            state += self.threshold * weights[idx]

            # Find newly active nodes
            newly_active = np.where(state >= self.threshold)[0]
            for j in newly_active:
                if j not in active:
                    active.append(j)
        return avalanche_size


class ScaleFreeTopology:
    """Generate a simple scale-free network using preferential attachment."""

    def __init__(self, num_nodes: int, m: int = 2, seed: Optional[int] = None):
        self.num_nodes = num_nodes
        self.m = max(1, m)
        self.random = np.random.default_rng(seed)

    def generate(self) -> np.ndarray:
        """Return a weight matrix with scale-free degree distribution."""
        n = self.num_nodes
        m = self.m

        # Start with a small fully connected network of m+1 nodes
        edges = {i: set(j for j in range(m + 1) if j != i) for i in range(m + 1)}
        degrees = np.array([len(edges[i]) for i in range(m + 1)], dtype=float)

        for new_node in range(m + 1, n):
            targets = self.random.choice(
                np.arange(new_node), size=m, replace=False, p=degrees[:new_node] / degrees[:new_node].sum()
            )
            edges[new_node] = set(targets)
            degrees = np.append(degrees, 0.0)
            for t in targets:
                edges[t].add(new_node)
                degrees[t] += 1
                degrees[new_node] += 1

        # Build weight matrix
        weights = np.zeros((n, n), dtype=float)
        for i, neighbours in edges.items():
            if neighbours:
                # random initial weights
                w = self.random.random(len(neighbours))
                w /= w.sum()
                for j, wij in zip(neighbours, w):
                    weights[i, j] = wij
        return weights


class SelfOrganizedCriticality:
    """Self-organised critical network combining topology and avalanches."""

    def __init__(
        self,
        num_nodes: int,
        threshold: float = 1.0,
        adapt_rate: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.threshold = threshold
        self.adapt_rate = adapt_rate
        self.random = np.random.default_rng(seed)

        topology = ScaleFreeTopology(num_nodes, seed=seed)
        self.base_weights = topology.generate()  # rows normalised to 1
        self.state = np.zeros(num_nodes, dtype=float)
        self.weight_scale = 1.0
        self.avalanche = AvalancheDynamics(threshold)

    def update_network(self, activity: np.ndarray) -> int:
        """Update network with external ``activity`` and propagate avalanches.

        Parameters
        ----------
        activity: np.ndarray
            External input added to each node's activity.

        Returns
        -------
        int
            Size of the avalanche triggered by this update (0 if none).
        """

        if activity.shape[0] != self.num_nodes:
            raise ValueError("Activity vector size does not match number of nodes")

        self.state += activity
        avalanche_size = 0
        if np.any(self.state >= self.threshold):
            weights = self.base_weights * self.weight_scale
            avalanche_size = self.avalanche.propagate(self.state, weights)
            self._adapt_weights(avalanche_size)
        return avalanche_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _adapt_weights(self, avalanche_size: int) -> None:
        """Adapt connection strengths to keep the system near criticality."""

        # Aim for avalanches of around ten percent of the network size. This
        # empirically keeps the branching ratio near unity and encourages a
        # heavy–tailed distribution of avalanche magnitudes.
        target = self.num_nodes * 0.1
        if avalanche_size > target:
            self.weight_scale *= 1.0 - self.adapt_rate
        elif 0 < avalanche_size < target / 2:
            self.weight_scale *= 1.0 + self.adapt_rate
        self.weight_scale = float(np.clip(self.weight_scale, 0.01, 10.0))
