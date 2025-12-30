"""Tests for reinforcement learning meta-learning utilities."""

from __future__ import annotations

import random
from typing import List, Tuple

import pytest

torch = pytest.importorskip("torch")

from backend.rl.agents import ActorCriticAgent
from backend.rl.meta_learning import (
    MetaLearner,
    MetaLearningConfig,
    SkillLibrary,
    SkillRecord,
)
from backend.rl.simulated_env import SimulatedTaskEnv, TaskConfig, TaskSuite, vectorize_observation


def _simple_suite() -> Tuple[TaskSuite, List[str], ActorCriticAgent]:
    task = TaskConfig(
        name="line-walk",
        grid_size=(2, 2),
        start=(0, 0),
        goal=(1, 0),
        action_set=["right", "wait"],
        max_steps=2,
        goal_reward=1.0,
    )
    suite = TaskSuite(tasks={task.name: task})
    env = SimulatedTaskEnv(task, rng=random.Random(42))
    obs = env.reset()
    features, _ = vectorize_observation(env, obs, suite.action_space)
    action_space = suite.action_space
    agent = ActorCriticAgent(len(features), len(action_space))
    return suite, action_space, agent


def test_skill_library_keeps_best_records() -> None:
    library = SkillLibrary(max_size=2)
    library.add_skill(
        SkillRecord(task_name="t1", actions=["a"], reward=0.5, success_rate=1.0)
    )
    library.add_skill(
        SkillRecord(task_name="t2", actions=["b"], reward=0.7, success_rate=1.0)
    )
    library.add_skill(
        SkillRecord(task_name="t3", actions=["c"], reward=0.1, success_rate=0.2)
    )

    assert len(library.skills) == 2
    rewards = [skill.reward for skill in library.skills]
    assert rewards == sorted(rewards, reverse=True)


class _DeterministicMetaLearner(MetaLearner):
    """Meta-learner with deterministic inner-loop updates for testing."""

    def _adapt_to_task(self, task):  # type: ignore[override]
        adapted = self._clone_agent()
        with torch.no_grad():
            for param in adapted.parameters():
                param.add_(0.05)
        actions = ["right", "wait"]
        reward = 1.0
        success_rate = 1.0
        return adapted, actions, reward, success_rate


def test_meta_learner_updates_base_agent_and_stores_skills() -> None:
    suite, action_space, agent = _simple_suite()
    config = MetaLearningConfig(
        meta_iterations=3,
        meta_batch_size=1,
        inner_steps=1,
        inner_learning_rate=0.01,
        meta_learning_rate=0.5,
        eval_episodes=1,
        skill_success_threshold=0.5,
        max_skill_library_size=5,
        seed=123,
    )

    meta = _DeterministicMetaLearner(
        agent,
        suite,
        config,
        action_space=action_space,
        rng=random.Random(123),
    )

    before = [param.detach().clone() for param in agent.parameters()]
    stats = meta.meta_train(log_interval=0)
    after = [param.detach() for param in agent.parameters()]

    assert len(stats) == 3
    assert len(meta.skill_library.skills) >= 1
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_meta_learner_maml_mode_updates_agent() -> None:
    suite, action_space, agent = _simple_suite()
    config = MetaLearningConfig(
        meta_iterations=2,
        meta_batch_size=1,
        inner_steps=2,
        inner_learning_rate=0.05,
        meta_learning_rate=0.1,
        eval_episodes=1,
        skill_success_threshold=0.5,
        max_skill_library_size=5,
        seed=321,
        algorithm="maml",
    )

    meta = MetaLearner(
        agent,
        suite,
        config,
        action_space=action_space,
        rng=random.Random(321),
    )

    before = [param.detach().clone() for param in agent.parameters()]
    stats = meta.meta_train(log_interval=0)
    after = [param.detach() for param in agent.parameters()]

    assert len(stats) == 2
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))
