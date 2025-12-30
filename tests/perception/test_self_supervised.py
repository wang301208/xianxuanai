import numpy as np

from BrainSimulationSystem.perception.self_supervised import (
    ContrastiveLearner,
    ContrastiveLearnerConfig,
)


def test_contrastive_learner_trains_from_buffer():
    cfg = ContrastiveLearnerConfig(feature_dim=16, projection_dim=4, batch_size=8, buffer_size=32)
    learner = ContrastiveLearner(cfg)
    rng = np.random.default_rng(0)

    for _ in range(40):
        learner.observe(rng.normal(size=cfg.feature_dim))

    assert learner.last_loss is not None
    initial_loss = learner.last_loss

    for _ in range(100):
        learner.observe(rng.normal(size=cfg.feature_dim))

    assert learner.last_loss <= initial_loss + 0.5  # ensure it updates without diverging
