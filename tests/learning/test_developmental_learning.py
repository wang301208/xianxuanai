from BrainSimulationSystem.core.stage_manager import CurriculumStageManager
from BrainSimulationSystem.environment.base import PerceptionPacket
from BrainSimulationSystem.environment.interaction_loop import (
    ImitationReplayBuffer,
    InteractiveLanguageLoop,
)
from BrainSimulationSystem.environment.policy_bridge import HierarchicalPolicyBridge
from BrainSimulationSystem.learning.developmental_learning import DevelopmentalLearningController


class DummyEnvController:
    def __init__(self, *, episode_length: int = 6):
        self._episode_length = int(episode_length)
        self._step = 0

    def reset(self):
        self._step = 0
        return PerceptionPacket(state_vector=[0.0], rewards={"reward": 0.0}, info={"step": 0})

    def step(self, action):
        self._step += 1
        reward = 1.0 if action == "left" else 0.0
        terminated = self._step >= self._episode_length
        packet = PerceptionPacket(
            state_vector=[float(self._step)],
            rewards={"reward": reward},
            terminated=terminated,
            info={"step": self._step, "success": bool(reward)},
        )
        return packet, reward, terminated, packet.info


class DummyRLAgent:
    class Config:
        update_interval = 1

    def __init__(self):
        self.steps = 0
        self.config = DummyRLAgent.Config()
        self.observed = []
        self.update_calls = 0

    def select_action(self, obs_vector):
        return 1, {"predicted_reward": 0.0}

    def observe(self, state, action_idx, reward, next_state, done):
        self.observed.append((state, action_idx, reward, next_state, done))

    def update(self):
        self.update_calls += 1
        return {"updates": self.update_calls, "observed": len(self.observed)}


def _build_loop(*, agent: DummyRLAgent, episode_length: int = 6) -> InteractiveLanguageLoop:
    controller = DummyEnvController(episode_length=episode_length)
    policy_bridge = HierarchicalPolicyBridge(
        decision_fn=lambda _packet: {"intent": "move", "confidence": 0.9},
        action_space=["left", "right"],
        feature_dim=8,
    )
    return InteractiveLanguageLoop(
        controller,
        policy_bridge,
        rl_selector=lambda obs: agent.select_action(obs),
        replay_buffer=ImitationReplayBuffer(capacity=128),
    )


def test_infant_stage_uses_dense_mentor_and_delayed_offline_training():
    agent = DummyRLAgent()
    loop = _build_loop(agent=agent, episode_length=5)
    stage_manager = CurriculumStageManager(starting_stage="infant")
    controller = DevelopmentalLearningController(loop=loop, stage_manager=stage_manager, rl_agent=agent)

    first = controller.run_episode()
    sources = {t.action_source for t in first["episode"]["transitions"]}
    assert sources == {"mentor"}
    assert first["offline"] is None

    second = controller.run_episode()
    offline = second["offline"]
    assert offline is not None
    assert offline["status"] == "trained"
    assert offline["encoded"] > 0


def test_juvenile_stage_mixes_mentor_and_policy_actions():
    agent = DummyRLAgent()
    loop = _build_loop(agent=agent, episode_length=6)
    stage_manager = CurriculumStageManager(starting_stage="juvenile")
    controller = DevelopmentalLearningController(loop=loop, stage_manager=stage_manager, rl_agent=agent)

    report = controller.run_episode()
    sources = {t.action_source for t in report["episode"]["transitions"]}
    assert "mentor" in sources
    assert "policy" in sources
