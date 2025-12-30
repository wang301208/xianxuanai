from BrainSimulationSystem.environment.base import (
    EnvironmentAdapter,
    EnvironmentController,
    ObservationTransformer,
    PerceptionPacket,
)
from backend.world_model import WorldModel


class DummySim:
    def __init__(self) -> None:
        self._step = 0
        self.initialized = False

    def initialize(self) -> None:
        self.initialized = True

    def reset(self, **kwargs):
        self._step = 0
        return {"rgb": [[0.0]], "state": [0.0, 1.0]}

    def step(self, action):
        self._step += 1
        obs = {
            "rgb": [[float(self._step)]],
            "audio": [0.1 * self._step],
            "state": [float(self._step), float(self._step + 1)],
        }
        reward = 1.0
        terminated = self._step >= 2
        info = {"action": action}
        return obs, reward, terminated, info

    def close(self) -> None:
        self.initialized = False


def test_environment_controller_feeds_world_model():
    sim = DummySim()
    adapter = EnvironmentAdapter(
        sim,
        transformer=ObservationTransformer(state_key="state"),
    )
    world = WorldModel()
    controller = EnvironmentController(adapter, world_model=world, agent_id="robot")

    packet = controller.reset()
    assert isinstance(packet, PerceptionPacket)
    stored = world.get_visual_observation("robot")
    assert stored["image"] == [[0.0]]

    packet, reward, terminated, info = controller.step({"continuous": [0.0]})
    assert reward == 1.0
    assert packet.audio == [0.1]
    assert packet.state_vector == [1.0, 2.0]
    assert info["action"] == {"continuous": [0.0]}
    assert terminated is False


def test_observers_receive_packets():
    sim = DummySim()
    adapter = EnvironmentAdapter(sim)
    controller = EnvironmentController(adapter)
    received = []

    controller.register_observer(received.append)
    controller.reset()
    controller.step({"continuous": [0.0]})

    assert len(received) == 2
    assert isinstance(received[0], PerceptionPacket)


def test_environment_controller_run_loop_executes_until_terminated():
    sim = DummySim()
    adapter = EnvironmentAdapter(sim, transformer=ObservationTransformer(state_key="state"))
    controller = EnvironmentController(adapter)

    def agent(_packet: PerceptionPacket):
        return {"continuous": [0.0]}

    result = controller.run_loop(agent, max_steps=10)
    assert result["steps"] == 2
    assert result["total_reward"] == 2.0
    assert result["terminated"] is True
    assert result["errors"] == []


def test_environment_controller_run_loop_action_validator_can_replace_action():
    sim = DummySim()
    adapter = EnvironmentAdapter(sim, transformer=ObservationTransformer(state_key="state"))
    controller = EnvironmentController(adapter)

    def agent(_packet: PerceptionPacket):
        return {"continuous": [999.0]}

    def validator(action, _packet: PerceptionPacket):
        assert action == {"continuous": [999.0]}
        return False, "force_safe_action", {"continuous": [0.0]}

    result = controller.run_loop(agent, max_steps=2, action_validator=validator)
    assert result["steps"] == 2
    assert result["blocked_actions"] == 2
