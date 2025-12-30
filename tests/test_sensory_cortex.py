import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.brain import VisualCortex, AuditoryCortex, SomatosensoryCortex
from modules.brain.neuromorphic.spiking_network import SpikingNeuralNetwork


@pytest.fixture
def checkerboard_image() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)


@pytest.fixture
def tone_signal() -> np.ndarray:
    t = np.linspace(0, 1, 1600, dtype=np.float32)
    carrier = 0.6 * np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 40 * t)
    modulation = np.linspace(0.2, 1.0, t.size, dtype=np.float32)
    return (carrier * modulation).astype(np.float32)


@pytest.fixture
def tactile_grid() -> np.ndarray:
    return np.array([[0.0, 0.4, 0.8], [0.2, 0.6, 1.0]], dtype=np.float32)


def test_visual_cortex_features(checkerboard_image: np.ndarray) -> None:
    cortex = VisualCortex()
    result = cortex.process(checkerboard_image)

    assert set(result.keys()) == {"v1", "v2", "v4", "mt"}
    for stage in result.values():
        assert isinstance(stage["vector"], list)
        assert isinstance(stage["features"], dict)
        assert stage["metadata"]["stage"] in {"V1", "V2", "V4", "MT"}

    v1_features = result["v1"]["features"]
    assert v1_features["edge_density"] == pytest.approx(0.546875, rel=1e-6)
    assert v1_features["edge_energy"] == pytest.approx(0.19950187, rel=1e-6)

    mt_features = result["mt"]["features"]
    assert mt_features["motion_energy"] == pytest.approx(0.02034884, rel=1e-6)
    assert mt_features["motion_bias"] == pytest.approx(0.0, abs=1e-6)


def test_auditory_cortex_features(tone_signal: np.ndarray) -> None:
    cortex = AuditoryCortex()
    result = cortex.process(tone_signal)

    assert set(result.keys()) == {"a1", "a2"}
    a1_features = result["a1"]["features"]
    assert a1_features["dominant_band"] == pytest.approx(1.0, abs=1e-6)
    assert a1_features["band_energy"] == pytest.approx(5.507319, rel=1e-6)

    a2_features = result["a2"]["features"]
    assert a2_features["temporal_variance"] == pytest.approx(2.1737657, rel=1e-6)
    assert a2_features["spectral_spread"] == pytest.approx(4.3831153, rel=1e-6)


def test_somatosensory_cortex_features(tactile_grid: np.ndarray) -> None:
    cortex = SomatosensoryCortex()
    result = cortex.process(tactile_grid)

    assert set(result.keys()) == {"s1"}
    s1_features = result["s1"]["features"]
    assert s1_features["central_pressure"] == pytest.approx(0.5714286, rel=1e-6)
    assert s1_features["edge_pressure"] == pytest.approx(0.5, rel=1e-6)
    assert s1_features["mean_pressure"] == pytest.approx(0.5, rel=1e-6)


def test_visual_cortex_neuromorphic_bridge(checkerboard_image: np.ndarray) -> None:
    snn = SpikingNeuralNetwork(2, weights=[[0.0, 1.0], [0.0, 0.0]])
    initial_weight = snn.synapses.weights[0][1]
    cortex = VisualCortex(spiking_backend=snn)
    result = cortex.process(checkerboard_image)

    assert "neuromorphic" in result
    payload = result["neuromorphic"]
    assert len(payload["spike_counts"]) == 2
    assert any(count > 0 for count in payload["spike_counts"])
    assert snn.synapses.weights[0][1] != initial_weight


def test_auditory_cortex_neuromorphic_bridge(tone_signal: np.ndarray) -> None:
    snn = SpikingNeuralNetwork(4, weights=np.ones((4, 4)).tolist())
    cortex = AuditoryCortex(spiking_backend=snn)
    result = cortex.process(tone_signal)

    assert "neuromorphic" in result
    payload = result["neuromorphic"]
    assert len(payload["spike_counts"]) == 4
    assert any(count > 0 for count in payload["spike_counts"])


def test_somatosensory_cortex_neuromorphic_bridge(tactile_grid: np.ndarray) -> None:
    snn = SpikingNeuralNetwork(3, weights=np.ones((3, 3)).tolist())
    cortex = SomatosensoryCortex(spiking_backend=snn)
    result = cortex.process(tactile_grid)

    assert "neuromorphic" in result
    payload = result["neuromorphic"]
    assert len(payload["spike_counts"]) == 3
    assert any(count > 0 for count in payload["spike_counts"])
