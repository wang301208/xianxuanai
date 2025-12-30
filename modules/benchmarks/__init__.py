"""Common benchmark problems for optimization algorithms."""

from .problems import Problem, Sphere, Rastrigin, ConstrainedQuadratic

PROBLEMS = {
    "sphere": Sphere,
    "rastrigin": Rastrigin,
    "constrained": ConstrainedQuadratic,
}

__all__ = ["Problem", "Sphere", "Rastrigin", "ConstrainedQuadratic", "PROBLEMS"]
