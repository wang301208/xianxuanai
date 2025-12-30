"""Convenience wrapper exposing Particle Swarm Optimization utilities."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.pso import PSOResult, pso, linear_schedule

__all__ = ["PSOResult", "pso", "linear_schedule"]
