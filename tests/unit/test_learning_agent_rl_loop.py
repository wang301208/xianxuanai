from __future__ import annotations

from typing import Any, Dict

import numpy as np

from modules.environment.loop import ActionPerceptionLoop
from modules.environment.simulator import GridWorldEnvironment
from modules.events import InMemoryEventBus
from modules.learning.learning_agent import LearningAgent, LearningAgentConfig, run_rl_training


def test_learning_agent_q_learning_updates_weights() -> None:
    agent = LearningAgent(
        LearningAgentConfig(
            algorithm="q_learning",
            actions=("a", "b"),
            state_dim=2,
            lr=0.1,
            gamma=0.0,
            epsilon=0.0,
            min_replay_size=1,
            batch_size=1,
            train_every_steps=1,
            updates_per_train=1,
            seed=123,
        )
    )
    state = np.array([1.0, 0.0], dtype=np.float32)
    next_state = np.array([0.0, 1.0], dtype=np.float32)
    agent.observe(state, 0, 1.0, next_state, True)

    assert agent._q_weights is not None
    assert float(agent._q_weights[0, 0]) > 0.0


def test_learning_agent_actor_critic_updates_weights() -> None:
    agent = LearningAgent(
        LearningAgentConfig(
            algorithm="actor_critic",
            actions=("a", "b"),
            state_dim=2,
            policy_lr=0.1,
            value_lr=0.1,
            gamma=0.0,
            epsilon=0.0,
            min_replay_size=1,
            batch_size=1,
            train_every_steps=1,
            updates_per_train=1,
            seed=123,
        )
    )
    state = np.array([1.0, 0.0], dtype=np.float32)
    next_state = np.array([0.0, 1.0], dtype=np.float32)
    agent.observe(state, 0, 1.0, next_state, True)

    assert agent._actor_weights is not None
    assert agent._value_weights is not None
    assert float(agent._actor_weights[0, 0]) != 0.0
    assert float(agent._value_weights[0]) != 0.0


class _DummySemanticBridge:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def process(self, _snapshot: Any, *, agent_id: Any = None, cycle_index: Any = None, ingest: bool = False):
        self.calls.append({"agent_id": agent_id, "cycle_index": cycle_index, "ingest": ingest})

        class _Output:
            semantic_annotations: Dict[str, Dict[str, Any]] = {}
            knowledge_facts: list[Dict[str, Any]] = []
            fused_embedding = [0.25, 0.75]
            modality_embeddings: Dict[str, list[float]] = {}

        return _Output()


def test_action_perception_loop_step_and_process_emits_perception_event() -> None:
    bus = InMemoryEventBus()
    env = GridWorldEnvironment(width=2, height=2, max_steps=2)
    bridge = _DummySemanticBridge()
    loop = ActionPerceptionLoop(event_bus=bus, environment=env, semantic_bridge=bridge)

    received: list[Dict[str, Any]] = []

    async def _handler(event: Dict[str, Any]) -> None:
        received.append(dict(event))

    bus.subscribe("environment.perception", _handler)

    reset_event = loop.process_reset(
        agent_id="agent",
        task_id="t0",
        cycle=0,
        publish=True,
        broadcast_workspace=False,
    )
    bus.join()
    assert reset_event is not None
    assert received and received[-1]["status"] == "environment_reset"

    step_event = loop.step_and_process(
        agent_id="agent",
        task_id="t0",
        cycle=1,
        command="move_east",
        arguments={},
        publish=True,
        broadcast_workspace=False,
    )
    bus.join()
    assert step_event is not None
    assert step_event["status"] == "environment_feedback"
    assert "metadata" in step_event
    assert "reward" in step_event["metadata"]
    assert "done" in step_event["metadata"]
    assert received and received[-1]["status"] == "environment_feedback"


def test_run_rl_training_smoke() -> None:
    bus = InMemoryEventBus()
    env = GridWorldEnvironment(width=2, height=2, max_steps=3)
    loop = ActionPerceptionLoop(event_bus=bus, environment=env, semantic_bridge=_DummySemanticBridge())
    agent = LearningAgent(
        LearningAgentConfig(
            algorithm="q_learning",
            actions=("move_north", "move_south", "move_west", "move_east"),
            state_dim=2,
            min_replay_size=1,
            batch_size=1,
            train_every_steps=1,
            updates_per_train=1,
            gamma=0.0,
            epsilon=0.0,
            seed=7,
        )
    )
    reports = run_rl_training(loop=loop, agent=agent, episodes=2, max_steps=2, publish_events=False)
    assert len(reports) == 2

