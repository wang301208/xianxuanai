import pytest

from modules.perception.semantic_bridge import ASRConfig, SemanticBridge


def _build_payload():
    return {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "features": {
            "energy": 0.6,
            "spectral_centroid": 2500.0,
            "spectral_flux": 0.35,
            "temporal_modulation": 0.08,
        },
        "metadata": {"frames": 1, "mels": 4},
        "raw_waveform": [0.01, -0.02, 0.03, -0.04],
    }


def test_audio_transcription_success(tmp_path):
    transcripts = []

    def stub_transcriber(audio_input, metadata):
        transcripts.append((audio_input, metadata))
        assert audio_input == _build_payload()["raw_waveform"]
        return "hello world"

    bridge = SemanticBridge(storage_root=tmp_path, asr_config=ASRConfig(provider=None, transcriber=stub_transcriber))

    result = bridge._process_audio(_build_payload(), "audio", "agent-1", 3)
    annotation = result["annotation"]
    facts = result["facts"]
    memory = result["memory"]

    assert annotation["transcript"] == "hello world"
    assert annotation["transcription_status"] == "success"
    assert annotation["metrics"]["energy"] == pytest.approx(0.6)
    assert any(f.predicate == "hasTranscript" for f in facts)

    assert memory.metadata["transcript"] == "hello world"
    assert memory.metadata["transcription_status"] == "success"
    assert any(msg["role"] == "transcript" for msg in memory.messages)
    assert "Transcript: \"hello world\"" in memory.summary

    assert transcripts, "Transcriber stub should have been invoked"


def test_audio_transcription_failure(tmp_path):
    def failing_transcriber(audio_input, metadata):
        raise RuntimeError("model missing")

    bridge = SemanticBridge(storage_root=tmp_path, asr_config=ASRConfig(provider=None, transcriber=failing_transcriber))

    result = bridge._process_audio(_build_payload(), "audio", "agent-1", 4)
    annotation = result["annotation"]
    facts = result["facts"]
    memory = result["memory"]

    assert "transcript" not in annotation
    assert annotation["transcription_status"] == "error"
    assert annotation["transcript_error"] == "model missing"

    assert "transcript" not in memory.metadata
    assert memory.metadata["transcription_status"] == "error"
    assert memory.metadata["transcript_error"] == "model missing"
    assert "Transcript:" not in memory.summary

    assert not any(f.predicate == "hasTranscript" for f in facts)
    assert any(f.predicate == "hasLabel" for f in facts)
