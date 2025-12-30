"""Collection of optimization algorithms with a unified interface."""

from importlib import import_module
from typing import Callable

_ALGORITHM_MODULES = {
    "ga": "ga",
    "pso": "pso",
    "aco": "aco",
    "random": "random_search",
    "evolution": "evolution_engine",
}


def _lazy_optimize(module_name: str) -> Callable:
    def _runner(*args, **kwargs):
        module = import_module(f"{__name__}.{module_name}")
        return module.optimize(*args, **kwargs)

    return _runner


ALGORITHMS = {name: _lazy_optimize(module) for name, module in _ALGORITHM_MODULES.items()}


def load_algorithm(name: str):
    """Dynamically import and return the algorithm module ``name``."""

    if name not in _ALGORITHM_MODULES:
        raise KeyError(f"Unknown algorithm: {name}")
    return import_module(f"{__name__}.{_ALGORITHM_MODULES[name]}")


def __getattr__(name: str):  # pragma: no cover - simple delegation
    if name in _ALGORITHM_MODULES.values():
        return import_module(f"{__name__}.{name}")
    raise AttributeError(name)


__all__ = ["ALGORITHMS", "load_algorithm"] + list(_ALGORITHM_MODULES.values())
