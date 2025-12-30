import numpy as np

from BrainSimulationSystem.environment.base import PerceptionPacket
from BrainSimulationSystem.perception.vision import VisionPerceptionModule
from BrainSimulationSystem.perception.self_supervised import ContrastiveLearner, ContrastiveLearnerConfig


class FakeCortex:
    def __init__(self) -> None:
        self.last_shape = None

    def process_visual_input(self, visual_input):
        self.last_shape = getattr(visual_input, "data").shape
        return {
            "attention_map": np.zeros((64, 64)),
            "saliency_map": np.zeros((64, 64)),
        }


def random_image(h=64, w=64, c=3):
    return (np.random.rand(h, w, c) * 255).astype(np.uint8)


def test_vision_module_processes_packet_without_torch():
    cortex = FakeCortex()
    module = VisionPerceptionModule(cortex=cortex)
    packet = PerceptionPacket(vision=random_image(32, 32, 3))

    observation = module.process(packet)

    assert cortex.last_shape == module.config.input_size + (3,)
    assert "features" in observation
    assert observation["attention_map"].shape == (64, 64)


def test_vision_module_requires_vision():
    module = VisionPerceptionModule(cortex=FakeCortex())
    packet = PerceptionPacket(vision=None)
    try:
        module.process(packet)
    except ValueError as exc:
        assert "requires" in str(exc)
    else:  # pragma: no cover - should not happen
        assert False, "Expected ValueError when vision data missing"


def test_self_supervised_learner_receives_features():
    cortex = FakeCortex()
    module = VisionPerceptionModule(cortex=cortex)
    learner = ContrastiveLearner(
        ContrastiveLearnerConfig(feature_dim=64, projection_dim=8, batch_size=4)
    )
    module.attach_self_supervised(learner)

    for _ in range(5):
        packet = PerceptionPacket(vision=random_image(32, 32, 3))
        module.process(packet)

    assert learner.last_loss is not None
