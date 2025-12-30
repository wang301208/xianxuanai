"""Simplified Grover search simulation.

This module provides a lightweight implementation of Grover's search
algorithm.  It avoids heavy quantum computing dependencies by operating on
numpy arrays and simulating amplitude amplification directly.  The goal is to
offer a mock quantum search routine suitable for unit tests and benchmarks
that compare against classical search strategies.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def grover_search(items: Sequence[T], oracle: Callable[[T], bool]) -> T:
    """Return the item satisfying *oracle* using Grover's algorithm.

    Parameters
    ----------
    items:
        Sequence of items to search.  The sequence length should be a power of
        two for the simulation to mimic a true quantum system, but this
        function will operate on arbitrary lengths for convenience.
    oracle:
        Function returning ``True`` for the desired item.

    The implementation simulates the standard single-solution Grover search
    by iteratively applying an oracle phase flip followed by the diffusion
    operator.  The number of iterations is chosen based on the size of the
    search space assuming a single matching item.
    """

    n = len(items)
    if n == 0:
        raise ValueError("items sequence is empty")

    amplitudes = np.ones(n, dtype=float) / math.sqrt(n)
    marked = np.array([1 if oracle(it) else 0 for it in items], dtype=int)
    m = int(marked.sum())
    if m == 0:
        raise ValueError("oracle does not mark any item")

    iterations = int(np.floor(math.pi / 4 * math.sqrt(n / m)))
    for _ in range(iterations):
        # Oracle phase inversion
        amplitudes = np.where(marked, -amplitudes, amplitudes)
        # Diffusion around the mean
        mean = amplitudes.mean()
        amplitudes = 2 * mean - amplitudes

    probabilities = np.abs(amplitudes) ** 2
    index = int(np.argmax(probabilities))
    return items[index]


__all__ = ["grover_search"]
