from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Any, Iterable

from .storage import load_history, DEFAULT_HISTORY_FILE


def optimize_params(
    algorithm: str,
    search_space: Dict[str, Iterable],
    metric: str = "score",
    history_file: Path = DEFAULT_HISTORY_FILE,
) -> Dict[str, Any]:
    """Return recommended parameters for ``algorithm``.

    Parameters are chosen by looking at past runs stored in ``history_file``. If
    runs for the given algorithm exist, the parameters with the highest value for
    the provided ``metric`` are returned. Otherwise a random sample from the
    provided ``search_space`` is used as a starting point.
    """
    best_params = None
    best_score = None
    for record in load_history(history_file):
        if record["algorithm"] != algorithm:
            continue
        score = record["metrics"].get(metric)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_params = record["params"]

    if best_params is not None:
        return best_params

    # Fallback to random selection
    return {name: random.choice(list(values)) for name, values in search_space.items()}


def log_run(
    algorithm: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    history_file: Path = DEFAULT_HISTORY_FILE,
) -> None:
    """Public wrapper for appending history."""
    from .storage import append_history

    append_history(algorithm, params, metrics, history_file)
