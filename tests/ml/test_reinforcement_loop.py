
from typing import Any, Dict, List, Tuple

import numpy as np

from BrainSimulationSystem.environment.base import EnvironmentAdapter, EnvironmentController, ObservationTransformer
from BrainSimulationSystem.environment.policy_bridge import HierarchicalPolicyBridge
from BrainSimulationSystem.motivation.curiosity_bridge import CuriosityStimulusEncoder
import random

from backend.ml.deep_rl import DQNAgent, DQNConfig, TORCH_AVAILABLE
from backend.ml.experience_collector import ActiveCuriositySelector
from backend.ml.rl_loop import ReinforcementLearningLoop, ReinforcementTrainerConfig
from modules.learning import EpisodeRecord, ExperienceHub


class DummySim:
    def __init__(self) -> None:
        self._step = 0

    def initialize(self) -> None:  # pragma: no cover - no-op
        pass

    def reset(self, **_kwargs):
        self._step = 0
        return {"state": [0.0, 0.0], "reward": 0.0}

    def step(self, action):
        self._step += 1
        obs = {"state": [float(self._step), float(self._step) * 0.5]}
        reward = 1.0 if action == "forward" else 0.1
        done = self._step >= 3
        info = {"usage": {"cpu": 0.1, "memory": 0.2}, "success": done}
        return obs, reward, done, info

    def close(self) -> None:  # pragma: no cover - no-op
        pass


def decision_fn(_packet):
    return {"intent": "explore", "confidence": 0.9}


class TabularAgent:
    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = 0.2
        self.gamma = 0.9
        self.q_table: Dict[Tuple[float, ...], np.ndarray] = {}

    def _state_key(self, state: np.ndarray) -> Tuple[float, ...]:
        return tuple(np.round(state, 3).tolist())

    def select_action(self, state: np.ndarray):
        key = self._state_key(state)
        q_values = self.q_table.setdefault(key, np.zeros(self.action_dim))
        if random.random() < 0.3:
            action = random.randrange(self.action_dim)
        else:
            action = int(np.argmax(q_values))
        self._last = (key, action)
        return action, {}

    def observe(self, state, action, reward, next_state, done):
        key, act = self._last
        next_key = self._state_key(next_state)
        q_values = self.q_table.setdefault(key, np.zeros(self.action_dim))
        next_q = self.q_table.setdefault(next_key, np.zeros(self.action_dim))
        target = reward + self.gamma * (0.0 if done else np.max(next_q))
        q_values[act] += self.alpha * (target - q_values[act])

    def update(self):
        return {"tabular_updates": len(self.q_table)}


class StubSelfLearningBrain:
    def __init__(self) -> None:
        self.samples: List[Dict[str, Any]] = []

    def curiosity_driven_learning(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        self.samples.append(sample)
        return {"cpu": sample.get("usage", {}).get("cpu", 0.0)}


class StubMemoryBridge:
    def __init__(self) -> None:
        self.episodes: List[EpisodeRecord] = []

    def record_episode(self, episode, *, metrics=None, curiosity_samples=None):
        self.episodes.append(episode)
        return "mem-id"


class StubCuriosityEngine:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.updates: List[tuple[float, str]] = []

    def compute_integrated_curiosity(self, stimulus: Dict[str, Any]) -> float:
        self.calls.append(stimulus)
        return 0.5

    def update_social_parameters(self, reward: float, social_type: str) -> None:
        self.updates.append((reward, social_type))


def test_reinforcement_loop_generates_experience(tmp_path):
    sim = DummySim()
    adapter = EnvironmentAdapter(sim, transformer=ObservationTransformer(state_key="state"))
    controller = EnvironmentController(adapter)
    bridge = HierarchicalPolicyBridge(
        decision_fn=decision_fn,
        action_space=["forward", "turn"],
        feature_dim=8,
    )

    if TORCH_AVAILABLE:
        dqn_config = DQNConfig(batch_size=2, warmup_steps=2, buffer_size=32, update_interval=1)
        agent = DQNAgent(state_dim=8, action_dim=2, config=dqn_config)
    else:
        agent = TabularAgent(state_dim=8, action_dim=2)
    hub = ExperienceHub(tmp_path / "hub")
    collected = []

    def trainer(samples):
        collected.extend(samples)

    curiosity = ActiveCuriositySelector(reward_threshold=-1.0)
    memory_bridge = StubMemoryBridge()
    loop = ReinforcementLearningLoop(
        env=controller,
        policy_bridge=bridge,
        rl_agent=agent,
        experience_hub=hub,
        policy_trainer=trainer,
        curiosity_selector=curiosity,
        self_learning_brain=StubSelfLearningBrain(),
        config=ReinforcementTrainerConfig(max_steps_per_episode=3),
        memory_bridge=memory_bridge,
    )

    episode = loop.run_episode(task_id="dummy-task")
    assert episode.steps > 0
    assert hub.index_path.exists()
    data = hub.index_path.read_text(encoding="utf-8").strip().splitlines()
    assert data, "experience hub should contain at least one episode"
    assert collected, "policy trainer should receive curiosity-selected samples"
    assert memory_bridge.episodes, "memory bridge should record the processed episode"


def test_curiosity_reward_is_added(tmp_path):
    sim = DummySim()
    adapter = EnvironmentAdapter(sim, transformer=ObservationTransformer(state_key="state"))
    controller = EnvironmentController(adapter)
    bridge = HierarchicalPolicyBridge(decision_fn=decision_fn, action_space=["forward"], feature_dim=6)
    agent = TabularAgent(state_dim=6, action_dim=1)
    hub = ExperienceHub(tmp_path / "hub2")

    engine = StubCuriosityEngine()
    encoder = CuriosityStimulusEncoder()

    loop = ReinforcementLearningLoop(
        env=controller,
        policy_bridge=bridge,
        rl_agent=agent,
        experience_hub=hub,
        curiosity_selector=ActiveCuriositySelector(reward_threshold=-1.0),
        self_learning_brain=StubSelfLearningBrain(),
        config=ReinforcementTrainerConfig(max_steps_per_episode=1, curiosity_weight=0.2),
        curiosity_engine=engine,
        curiosity_encoder=encoder,
    )

    episode = loop.run_episode(task_id="curious-task")
    assert engine.calls, "curiosity engine should be invoked"
    assert episode.total_reward >= 0.2, "intrinsic reward should boost total reward"
