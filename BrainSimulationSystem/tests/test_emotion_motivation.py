"""Tests for the emotion and motivation system."""

from __future__ import annotations

from BrainSimulationSystem.models.emotion_motivation import EmotionMotivationSystem


def test_emotion_motivation_reward_updates_valence():
    system = EmotionMotivationSystem({"baseline_valence": 0.0, "learning_rate": 0.5})
    state = system.update(reward=1.0, stimuli={})
    assert state["valence"] > 0
    assert system.get_state()["valence"] == state["valence"]


def test_emotion_motivation_generates_drive_for_goals():
    system = EmotionMotivationSystem({"motivation_gain": 0.8})
    state = system.update(reward=None, stimuli={"goals": ["explore"]})
    motivation = state["motivation"]
    assert motivation
    assert motivation.get("explore") is not None
