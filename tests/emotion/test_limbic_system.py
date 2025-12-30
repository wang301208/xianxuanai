import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.limbic import EmotionModel, EmotionProcessor, HomeostasisController, LimbicSystem
from modules.brain.state import BrainRuntimeConfig, PersonalityProfile
from schemas.emotion import EmotionalState, EmotionType


def test_emotion_processor_vad_regression_polarity():
    processor = EmotionProcessor()
    happy_emotion, happy_dims, happy_context = processor.evaluate(
        "We achieved a wonderful success and feel proud and joyful!",
        {"safety": 0.7, "social": 0.6},
    )
    angry_emotion, angry_dims, angry_context = processor.evaluate(
        "This disaster makes me furious and afraid of the looming danger!",
        {"threat": 0.95, "novelty": 0.2},
    )

    assert happy_emotion == EmotionType.HAPPY
    assert happy_dims["valence"] > 0.3
    assert 0.0 <= happy_dims["arousal"] <= 1.0
    assert angry_emotion == EmotionType.ANGRY
    assert angry_dims["valence"] < -0.3
    assert angry_dims["arousal"] > happy_dims["arousal"]
    assert -1.0 <= angry_dims["dominance"] <= 1.0
    assert "model_activation" in happy_context
    assert "model_activation" in angry_context
    assert processor.last_inference is not None
    assert len(processor.last_inference["features"]) == EmotionModel.HASH_BUCKETS + 22
    assert len(processor.last_inference["logits"]) == 3


def test_emotion_processor_context_modulation():
    processor = EmotionProcessor()
    calm_emotion, calm_dims, _ = processor.evaluate(
        "The situation is manageable and calm.",
        {"safety": 0.8},
    )
    threat_emotion, threat_dims, _ = processor.evaluate(
        "The situation is manageable and calm.",
        {"threat": 0.9},
    )

    assert calm_emotion != EmotionType.ANGRY
    assert threat_dims["arousal"] > calm_dims["arousal"]
    assert threat_dims["valence"] < calm_dims["valence"]


def test_limbic_system_react_backward_compatibility():
    limbic = LimbicSystem()

    legacy_config = BrainRuntimeConfig(enable_multi_dim_emotion=False)
    legacy_state = limbic.react(
        "Steady and ordinary update.",
        context={"safety": 0.4},
        config=legacy_config,
    )
    assert list(legacy_state.dimensions.keys()) == ["valence"]
    assert "approach" in legacy_state.intent_bias
    assert "withdraw" in legacy_state.intent_bias
    assert pytest.approx(1.0) == sum(legacy_state.intent_bias.values())

    modern_state = limbic.react(
        "We secured a remarkable victory!",
        context={"social": 0.7, "novelty": 0.5},
        config=BrainRuntimeConfig(),
    )
    assert {"valence", "arousal", "dominance"}.issubset(modern_state.dimensions.keys())
    assert modern_state.intent_bias["approach"] > modern_state.intent_bias["withdraw"]
    assert 0.0 <= modern_state.intensity <= 1.0


def test_homeostasis_controller_tracks_vad_dimensions():
    controller = HomeostasisController(decay_rate=0.2)
    profile = PersonalityProfile(
        openness=0.5,
        conscientiousness=0.6,
        extraversion=0.6,
        agreeableness=0.7,
        neuroticism=0.2,
    )
    state = EmotionalState(
        emotion=EmotionType.SAD,
        intensity=0.4,
        dimensions={"valence": -0.5},
    )
    full_dimensions = {"valence": -0.5, "arousal": 0.7, "dominance": -0.4}
    regulated = controller.regulate(
        state,
        profile,
        {"threat": 0.6, "novelty": 0.2, "safety": 0.1},
        enable_decay=True,
        full_dimensions=full_dimensions,
    )

    assert controller.last_dimensions["arousal"] == pytest.approx(0.7, abs=1e-6)
    assert regulated.decay == pytest.approx(controller.decay_rate)
    assert 0.0 <= regulated.intensity <= 1.0
    assert controller.mood < 0
