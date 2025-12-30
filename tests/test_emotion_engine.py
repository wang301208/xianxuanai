import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from emotion_engine import EmotionEngine
from schemas.emotion import EmotionalState, EmotionType
from modules.brain.state import BrainRuntimeConfig


def test_emotion_engine_process_emotion():
    engine = EmotionEngine()
    state = engine.process_emotion("I feel good today!", context={"safety": 0.4, "novelty": 0.2})
    assert isinstance(state, EmotionalState)
    assert isinstance(state.emotion, EmotionType)
    assert 0.0 <= state.intensity <= 1.0
    assert "valence" in state.dimensions
    assert "arousal" in state.dimensions
    assert state.intent_bias.get("approach") is not None
    assert state.context_weights.get("safety", 0.0) >= 0.0
    assert state.decay >= 0.0


def test_emotion_engine_config_updates_runtime():
    config = BrainRuntimeConfig(enable_multi_dim_emotion=False)
    engine = EmotionEngine(config=config)
    state = engine.process_emotion("steady text", context={"safety": 0.1})
    assert list(state.dimensions.keys()) == ["valence"]
    engine.update_config(BrainRuntimeConfig(enable_multi_dim_emotion=True))
    richer_state = engine.process_emotion("steady text", context={"safety": 0.1})
    assert "arousal" in richer_state.dimensions
