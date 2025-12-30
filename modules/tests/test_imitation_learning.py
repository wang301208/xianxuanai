import sys
import os

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.learning.imitation import ImitationLearner


def test_imitation_learner_updates_distribution_and_losses():
    learner = ImitationLearner()
    demos = [
        {"actions": ["a", "a", "b"]},
        {"action": "b"},
        {"steps": [{"action": "a"}, {"action": "c"}]},
        {"steps": [{"foo": "bar"}]},  # ignored
        {},  # ignored
    ]

    stats = learner.train(demos)

    assert stats.samples == 3
    assert len(stats.losses) == 3
    assert learner.trained_steps == 3
    assert set(stats.action_distribution) == {"a", "b", "c"}
    assert abs(sum(stats.action_distribution.values()) - 1.0) < 1e-9

