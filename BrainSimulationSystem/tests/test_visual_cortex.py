"""Tests for the visual cortex perception pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest import mock

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ..models.visual_cortex import VisualCortexModel
from ..models.auditory_cortex import AuditoryCortexModel
from ..models.perception import PerceptionProcess
from ..models.somatosensory_cortex import SomatosensoryCortex
from ..core.network.base import Layer, NeuralNetwork


class _TestNeuron:
    """Minimal neuron container used for perception tests."""

    def __init__(self) -> None:
        self.voltage = 0.0


class _DummyNetwork(NeuralNetwork):
    """Simple neural network stub exposing the minimal API used by PerceptionProcess."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layer_name = "sensory"
        neuron_ids = [0, 1, 2]
        self.layers[self.input_layer_name] = Layer(
            name=self.input_layer_name,
            neuron_type="excitatory",
            neurons=[_TestNeuron() for _ in neuron_ids],
            neuron_ids=neuron_ids,
        )
        self.neurons = {idx: _TestNeuron() for idx in neuron_ids}

    def step(self, dt=None):  # pragma: no cover - unused in these tests
        return super().step(dt)


def test_visual_cortex_model_numpy_backend_returns_embedding():
    """The numpy fallback backend should produce an embedding and feature maps."""

    model = VisualCortexModel({"backend": "numpy", "input_size": [32, 32]})
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

    output = model.process(image)

    assert "embedding" in output
    assert output["embedding"].size > 0
    assert "feature_maps" in output
    assert set(output["feature_maps"]) == {"edge_horizontal", "edge_vertical"}
    assert "confidence" in output


def test_perception_process_emits_vision_results_when_configured():
    """PerceptionProcess should attach vision output when vision is enabled."""

    network = _DummyNetwork()
    params = {
        "vision": {
            "enabled": True,
            "return_feature_maps": True,
            "model": {"backend": "numpy", "input_size": [16, 16]},
        }
    }
    perception = PerceptionProcess(network, params)

    image = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    result = perception.process({"image": image})

    assert "vision" in result, "Vision key should be present when vision processing runs"
    assert result["vision"]["embedding"].size > 0
    assert result["vision"]["backend"] in ("numpy", "torch")
    assert "confidence" in result["vision"]
    # No sensory data was provided, so the main perception output should remain empty.
    assert result["perception_output"] == []


def test_perception_process_without_vision_returns_basic_output():
    """If vision is disabled the process should still succeed gracefully."""

    network = _DummyNetwork()
    perception = PerceptionProcess(network, {"vision": {"enabled": False}})
    sensory_data = [0.1, 0.5, 0.9]

    result = perception.process({"sensory_data": sensory_data})

    assert result["perception_output"], "Sensory data should be propagated"
    assert "vision" not in result or "error" in result["vision"]



def test_auditory_cortex_numpy_backend_returns_embedding():
    model = AuditoryCortexModel({"backend": "numpy", "feature_dim": 64})
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    output = model.process(waveform)

    assert "embedding" in output
    assert output["embedding"].size > 0
    assert "feature_maps" in output
    assert "confidence" in output


def test_perception_process_handles_auditory_inputs():
    network = _DummyNetwork()
    params = {
        "auditory": {
            "enabled": True,
            "model": {"backend": "numpy", "feature_dim": 64},
        }
    }
    perception = PerceptionProcess(network, params)
    waveform = np.random.randn(800).astype(np.float32)

    result = perception.process({"audio": waveform, "audio_sample_rate": 8000})

    assert "auditory" in result
    assert result["auditory"]["embedding"].size > 0
    assert "confidence" in result["auditory"]


def test_perception_process_handles_somatosensory_inputs():
    network = _DummyNetwork()
    params = {
        "somatosensory": {"enabled": True}
    }
    perception = PerceptionProcess(network, params)
    sensors = {
        "pressure": [0.1, 0.4, 0.7],
        "temperature": [36.5, 36.6, 36.7],
    }

    result = perception.process({"somatosensory": sensors})

    assert "somatosensory" in result
    assert result["somatosensory"]["embedding"].size > 0
    assert "statistics" in result["somatosensory"]


def test_visual_torchvision_backend_mocked_embeddings():
    fake_vector = np.ones((512,), dtype=np.float32)
    fake_map = np.ones((1, 4), dtype=np.float32)

    def _fake_setup(self):
        self.status["active_backend"] = "torchvision"

    def _fake_process(self, image):
        return {
            "embedding": fake_vector.copy(),
            "feature_maps": {"resnet_embedding": fake_map.copy()},
            "confidence": 0.91,
        }

    with mock.patch.object(VisualCortexModel, "_setup_torchvision_backend", new=_fake_setup), \
        mock.patch.object(VisualCortexModel, "_process_torchvision", new=_fake_process):
        model = VisualCortexModel({"backend": "torchvision"})
        image = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        output = model.process(image)

    assert output["embedding"].shape == fake_vector.shape
    assert output["embedding"].dtype == np.float32
    assert output["feature_maps"]["resnet_embedding"].shape == fake_map.shape
    assert output["confidence"] == 0.91


def test_perception_process_warns_on_visual_backend_fallback():
    network = _DummyNetwork()
    params = {
        "vision": {"enabled": True, "model": {"backend": "clip"}}
    }

    with mock.patch.object(VisualCortexModel, "_setup_clip_backend", side_effect=RuntimeError("clip missing")):
        perception = PerceptionProcess(network, params)

    image = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    result = perception.process({"image": image})

    assert result["vision"]["backend"] == "numpy"
    assert result["vision"]["confidence"] <= 0.4
    assert "warning" in result["vision"]


def test_auditory_wav2vec_backend_mocked_embeddings():
    fake_vector = np.ones((32,), dtype=np.float32)
    fake_hidden = np.ones((4, 8), dtype=np.float32)

    def _fake_setup(self):
        self.status["active_backend"] = "wav2vec2"

    def _fake_process(self, waveform, sr):
        return {
            "embedding": fake_vector.copy(),
            "feature_maps": {"last_hidden_state": fake_hidden.copy()},
            "confidence": 0.9,
        }

    with mock.patch.object(AuditoryCortexModel, "_setup_wav2vec2_backend", new=_fake_setup), \
        mock.patch.object(AuditoryCortexModel, "_process_wav2vec2", new=_fake_process):
        model = AuditoryCortexModel({"backend": "wav2vec2"})
        waveform = np.random.randn(400).astype(np.float32)
        output = model.process(waveform, sample_rate=16_000)

    assert output["embedding"].shape == fake_vector.shape
    assert output["feature_maps"]["last_hidden_state"].shape == fake_hidden.shape
    assert output["confidence"] == 0.9


def test_perception_process_warns_on_auditory_backend_fallback():
    network = _DummyNetwork()
    params = {
        "auditory": {"enabled": True, "model": {"backend": "whisper"}}
    }

    with mock.patch.object(AuditoryCortexModel, "_setup_whisper_backend", side_effect=RuntimeError("whisper missing")):
        perception = PerceptionProcess(network, params)

    waveform = np.random.randn(3200).astype(np.float32)
    result = perception.process({"audio": waveform, "audio_sample_rate": 16_000})

    assert result["auditory"]["backend"] == "numpy"
    assert result["auditory"]["confidence"] <= 0.45
    assert "warning" in result["auditory"]


def test_perception_process_handles_structured_data_inputs():
    network = _DummyNetwork()
    params = {
        "structured": {"enabled": True, "parser": {"subject_field": "id"}},
    }
    perception = PerceptionProcess(network, params)
    structured_rows = [
        {"id": "entity-1", "label": "tree", "weight": 0.8},
        {"id": "entity-2", "label": "rock", "weight": 0.2},
    ]

    result = perception.process({"structured_data": {"payload": structured_rows, "source": {"name": "scene"}}})

    structured = result.get("structured")
    assert structured is not None
    assert structured["triples"]
    assert structured["metadata"]["rows"] == 2
    assert "summary_embedding" in structured


def test_perception_process_multimodal_fusion_combines_modalities():
    network = _DummyNetwork()
    params = {
        "structured": {"enabled": True},
        "multimodal_fusion": {"enabled": True},
    }
    perception = PerceptionProcess(network, params)
    rows = [
        {"id": "entity-1", "value": 0.5},
        {"id": "entity-2", "value": 0.9},
    ]
    language_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    result = perception.process(
        {
            "structured_data": rows,
            "language_embedding": language_embedding,
        }
    )

    fusion = result.get("multimodal_fusion")
    assert fusion is not None
    assert "embedding" in fusion
    assert len(fusion["modalities"]) >= 2


def test_perception_process_multimodal_fusion_respects_attention_directives():
    network = _DummyNetwork()
    params = {
        "structured": {"enabled": True},
        "multimodal_fusion": {"enabled": True},
    }
    perception = PerceptionProcess(network, params)
    rows = [
        {"id": "entity-1", "value": 0.1},
        {"id": "entity-2", "value": 0.2},
    ]
    language_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    result = perception.process(
        {
            "structured_data": rows,
            "language_embedding": language_embedding,
            "attention_directives": {"modality_weights": {"structured": 0.01, "language": 1.0}},
        }
    )

    fusion = result.get("multimodal_fusion")
    assert fusion is not None
    weights = fusion.get("attention_weights")
    assert isinstance(weights, dict)
    assert weights.get("language", 0.0) > weights.get("structured", 0.0)
