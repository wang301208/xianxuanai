import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.common.emotion import (
    EmotionAnalyzer,
    EmotionProfile,
    EmotionState,
    MLModel,
    adjust_response_style,
)


def test_keyword_model_classifies_voice():
    analyzer = EmotionAnalyzer()
    assert analyzer.analyze_voice(b"I love this") == "positive"


def test_emotion_analyzer_classifies_basic_sentiment():
    analyzer = EmotionAnalyzer()
    assert analyzer.analyze_text("I love this!") == "positive"
    assert analyzer.analyze_text("This is terrible") == "negative"
    assert analyzer.analyze_text("It is a table") == "neutral"


def test_emotion_profile_influences_classification_and_style():
    profile = EmotionProfile(positive_threshold=2, positive_suffix=":)", negative_prefix="Apologies:")
    analyzer = EmotionAnalyzer(profile=profile)
    state = EmotionState()

    state.update("good", analyzer)
    assert state.label == "neutral"  # not enough positive keywords

    state.update("good great", analyzer)
    assert state.label == "positive"

    response = adjust_response_style("Thanks", state, profile)
    assert response.endswith(":)")


def test_adjust_response_style_handles_multimodal_signals():
    analyzer = EmotionAnalyzer()
    state = EmotionState()
    state.update("good", analyzer)

    # Voice indicates negativity twice; majority vote yields negative response
    response = adjust_response_style("Hello", state, signals=["negative", "negative"])
    assert response.startswith("I'm sorry to hear that.")


def test_ml_model_classifies_text_with_mapping(monkeypatch):
    def fake_pipeline(task, model=None, device=None):
        assert device == "cpu"
        if task == "text-classification":
            
            def classifier(text):
                lowered = text.lower()
                if "love" in lowered:
                    return [{"label": "POSITIVE", "score": 0.9}]
                if "hate" in lowered:
                    return [{"label": "NEGATIVE", "score": 0.88}]
                return [{"label": "NEUTRAL", "score": 0.51}]

            return classifier
        raise AssertionError(f"unexpected task {task}")

    monkeypatch.setattr("modules.common.emotion.pipeline", fake_pipeline)
    model = MLModel(text_model="dummy", device="cpu")

    assert model.analyze_text("I love this") == "positive"
    assert model.analyze_text("I hate this") == "negative"
    assert model.analyze_text("Just a statement") == "neutral"


def test_ml_model_transcribes_voice_before_classifying(monkeypatch):
    def fake_pipeline(task, model=None, device=None):
        if task == "automatic-speech-recognition":

            def asr(_audio):
                return {"text": "I love this"}

            return asr
        if task == "text-classification":
            return lambda text: [{"label": "POSITIVE", "score": 0.93}]
        raise AssertionError(f"unexpected task {task}")

    monkeypatch.setattr("modules.common.emotion.pipeline", fake_pipeline)
    model = MLModel(text_model="dummy", asr_model="dummy-asr")

    assert model.analyze_voice(b"binary audio data") == "positive"


def test_ml_model_prefers_direct_audio_classifier(monkeypatch):
    def fake_pipeline(task, model=None, device=None):
        if task == "audio-classification":
            return lambda _audio: [{"label": "HAPPY", "score": 0.78}]
        if task == "text-classification":
            return lambda _text: [{"label": "NEUTRAL", "score": 0.51}]
        raise AssertionError(f"unexpected task {task}")

    monkeypatch.setattr("modules.common.emotion.pipeline", fake_pipeline)
    model = MLModel(
        text_model="dummy",
        audio_model="dummy-audio",
        label_mapping={"HAPPY": "positive", "NEUTRAL": "neutral"},
    )

    assert model.analyze_voice(b"binary audio data") == "positive"

