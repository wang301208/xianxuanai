import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.meta_learning import (
    SelfImprovementManager,
    MetaLearningBrain,
    MAMLEngine,
)


def test_autonomous_improvement_cycle():
    manager = SelfImprovementManager()
    engine = MAMLEngine(input_dim=1)
    brain = MetaLearningBrain(engine, manager)

    losses = brain.run(cycles=20)
    assert losses[-1] < losses[0]
    assert len(manager.history) == 20
