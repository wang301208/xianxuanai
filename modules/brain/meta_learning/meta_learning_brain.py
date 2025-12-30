from __future__ import annotations

from typing import List

import numpy as np

from .core import FewShotTask, MAMLEngine
from .self_improvement_manager import SelfImprovementManager


class MetaLearningBrain:
    """Coordinate meta-learning with automatic goal updates."""

    def __init__(self, engine: MAMLEngine, manager: SelfImprovementManager) -> None:
        self.engine = engine
        self.manager = manager

    def _make_task(self, difficulty: float) -> FewShotTask:
        support_x = np.array([[1.0], [2.0]])
        query_x = np.array([[3.0], [4.0]])
        support_y = difficulty * support_x.squeeze()
        query_y = difficulty * query_x.squeeze()
        return FewShotTask(support_x, support_y, query_x, query_y)

    def run(self, cycles: int, feedback: float | None = None) -> List[float]:
        """Run autonomous improvement cycles.

        For each cycle a task is sampled based on the current goal, the
        engine learns, performance is evaluated and the goal updated.
        """

        goal = {"difficulty": 1, "target_loss": float("inf")}
        losses: List[float] = []
        for _ in range(cycles):
            task = self._make_task(float(goal.get("difficulty", 1)))
            epochs = 2 if losses and losses[-1] > goal["target_loss"] else 1
            history = self.engine.learn_to_learn([task], epochs=epochs)
            loss = history[-1]
            losses.append(loss)

            metrics = [loss]
            goal = self.manager.generate_goal(metrics, metrics, metrics, feedback)
        return losses
