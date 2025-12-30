import os
import sys

import pytest

# Ensure the repository root is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.emotion.advanced import (
    AdvancedEmotionalSystem,
    EmotionSpace,
)


def test_multimodal_input_processing():
    system = AdvancedEmotionalSystem()
    stimulus = {"text": "happy", "audio": [0.6, 0.2], "visual": [0.1, 0.3]}
    emotion = system.evaluate(stimulus)

    assert isinstance(emotion, EmotionSpace)
    assert emotion.valence > 0
    assert emotion.arousal > 0
    assert emotion.dominance > 0
    assert len(system.memory.log) == 1


def test_memory_storage():
    system = AdvancedEmotionalSystem()
    stimuli = [
        {"text": "good", "audio": [0.4], "visual": [0.2]},
        {"text": "bad", "audio": [0.1], "visual": [0.9]},
    ]
    for s in stimuli:
        system.evaluate(s)

    assert len(system.memory.log) == len(stimuli)
    for s, stored in zip(stimuli, system.memory.log):
        expected = system.network.forward(
            text=s["text"], audio=s["audio"], visual=s["visual"]
        )
        assert stored.valence == pytest.approx(expected[0])
        assert stored.arousal == pytest.approx(expected[1])
        assert stored.dominance == pytest.approx(expected[2])


def test_regulation_effects():
    def dampen_arousal(emotion: EmotionSpace) -> EmotionSpace:
        return EmotionSpace(emotion.valence, emotion.arousal * 0.5, emotion.dominance)

    system = AdvancedEmotionalSystem(regulation_strategies=[dampen_arousal])
    stimulus = {"text": "excited", "audio": [0.8], "visual": [0.5]}
    regulated = system.evaluate(stimulus)
    stored = system.memory.log[0]

    assert regulated.arousal == pytest.approx(stored.arousal * 0.5)
    assert regulated.valence == pytest.approx(stored.valence)
    assert regulated.dominance == pytest.approx(stored.dominance)
