from BrainSimulationSystem.config.stage_profiles import build_stage_config
from BrainSimulationSystem.environment.developmental_environments import (
    GridWorldEnvironment,
    build_stage_environment,
)


def test_infant_stage_environment_emits_basic_perception_packet():
    cfg = build_stage_config("infant")
    bundle = build_stage_environment(cfg)

    packet = bundle.controller.reset()
    assert packet.text is None
    assert packet.vision is not None
    assert tuple(packet.vision.shape) == (64, 64, 3)

    packet, reward, terminated, info = bundle.controller.step("touch")
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert info["focus_color"]


def test_juvenile_stage_environment_includes_teacher_text():
    cfg = build_stage_config("juvenile")
    bundle = build_stage_environment(cfg)

    packet = bundle.controller.reset()
    assert isinstance(packet.text, str)
    assert "touch" in packet.text.lower()

    packet, reward, terminated, info = bundle.controller.step("look_left")
    assert isinstance(packet.text, str)
    assert "instruction" in info
    assert isinstance(info["instruction"], str)
    assert reward >= 0.0


def test_adolescent_open_world_defaults_to_grid_world_when_unity_unconfigured():
    cfg = build_stage_config("adolescent")
    bundle = build_stage_environment(cfg)

    assert bundle.config.kind == "open_world"
    assert isinstance(bundle.environment, GridWorldEnvironment)
    assert bundle.action_space == GridWorldEnvironment.ACTION_SPACE

    packet = bundle.controller.reset()
    assert packet.vision is not None
    assert tuple(packet.vision.shape) == (192, 192, 3)
    assert packet.state_vector is not None

