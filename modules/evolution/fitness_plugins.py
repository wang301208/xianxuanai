from __future__ import annotations

"""Plugin registry for genetic algorithm fitness metrics.

Users can register new metrics via the :func:`register` decorator or load
predefined metrics by name through a configuration file.  Each metric is a
callable that accepts an individual (sequence of floats) and returns a fitness
value.  The genetic algorithm combines multiple metrics using weights.
"""

from typing import Callable, Dict, List, Sequence, Tuple
from pathlib import Path
import yaml

# Registry mapping metric names to callables
_REGISTRY: Dict[str, Callable[[Sequence[float]], float]] = {}


def register(name: str) -> Callable[[Callable[[Sequence[float]], float]], Callable[[Sequence[float]], float]]:
    """Register a fitness metric under ``name``."""

    def decorator(fn: Callable[[Sequence[float]], float]) -> Callable[[Sequence[float]], float]:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get(name: str) -> Callable[[Sequence[float]], float]:
    """Retrieve a registered metric by name."""

    return _REGISTRY[name]


def load_from_config(path: str | Path) -> List[Tuple[Callable[[Sequence[float]], float], float]]:
    """Load metrics and weights from a YAML configuration file.

    The file should contain a top-level ``metrics`` list with ``name`` and
    optional ``weight`` entries, e.g.::

        metrics:
          - name: minimize_response_time
            weight: 0.7
          - name: minimize_resource_consumption
            weight: 0.3

    Returns a list of ``(metric_fn, weight)`` tuples.
    """

    data = yaml.safe_load(Path(path).read_text())
    result: List[Tuple[Callable[[Sequence[float]], float], float]] = []
    for item in data.get("metrics", []):
        fn = get(item["name"])
        weight = float(item.get("weight", 1.0))
        result.append((fn, weight))
    return result


# ---------------------------------------------------------------------------
# Built-in metric templates

@register("minimize_response_time")
def minimize_response_time(individual: Sequence[float]) -> float:
    """Template metric minimizing the first gene (response time)."""

    return -abs(individual[0])


@register("minimize_resource_consumption")
def minimize_resource_consumption(individual: Sequence[float]) -> float:
    """Template metric minimizing the second gene (resource consumption)."""

    return -abs(individual[1])
