from __future__ import annotations

import numpy as np

from BrainSimulationSystem.core.enums import BrainRegion
from BrainSimulationSystem.core.network import create_full_brain_network


def test_full_brain_network_vision_sensory_sets_thalamic_activity():
    net = create_full_brain_network()
    v1_columns = net.brain_regions[BrainRegion.PRIMARY_VISUAL_CORTEX]["columns"]
    assert v1_columns

    image = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    result = net.update(0.1, external_inputs={"image": image})

    assert "sensory" in result
    assert "vision" in result["sensory"]
    assert "input_sources" in result
    assert "vision_system" in result["input_sources"]

    column_result = result["columns"][v1_columns[0]]
    thalamic_activity = column_result.get("thalamic_activity")
    assert thalamic_activity is not None
    assert np.asarray(thalamic_activity).size > 0


def test_full_brain_network_auditory_sensory_sets_thalamic_activity():
    net = create_full_brain_network()
    a1_columns = net.brain_regions[BrainRegion.PRIMARY_AUDITORY_CORTEX]["columns"]
    assert a1_columns

    t = np.linspace(0, 0.05, 800, endpoint=False)
    waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    result = net.update(0.1, external_inputs={"audio": waveform, "audio_sample_rate": 16_000})

    assert "sensory" in result
    assert "auditory" in result["sensory"]
    assert "input_sources" in result
    assert "auditory_system" in result["input_sources"]

    column_result = result["columns"][a1_columns[0]]
    thalamic_activity = column_result.get("thalamic_activity")
    assert thalamic_activity is not None
    assert np.asarray(thalamic_activity).size > 0

