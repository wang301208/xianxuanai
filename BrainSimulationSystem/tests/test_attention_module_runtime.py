from __future__ import annotations

import numpy as np

from BrainSimulationSystem.config.default_config import get_config
from BrainSimulationSystem.core.network import create_full_brain_network


def _build_attention_config(*, gain: float) -> dict:
    cfg = get_config()
    runtime_cfg = cfg.setdefault("runtime", {})
    runtime_cfg["attention_module"] = {
        "enabled": True,
        "min_attention_gain": float(gain),
        "max_attention_gain": float(gain),
        "phasic_gain": 0.0,
    }
    return cfg


def test_full_brain_network_attention_module_emits_norepinephrine():
    cfg = _build_attention_config(gain=1.0)
    net = create_full_brain_network(cfg)

    result = net.update(0.1, external_inputs={})

    neuromodulators = result.get("neuromodulators")
    assert isinstance(neuromodulators, dict)
    assert "norepinephrine" in neuromodulators
    assert isinstance(neuromodulators["norepinephrine"], float)


def test_full_brain_network_attention_gain_applies_to_sensory_pathways():
    cfg = _build_attention_config(gain=1.5)
    net = create_full_brain_network(cfg)

    net.update(0.1, external_inputs={})
    image = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    result = net.update(0.1, external_inputs={"image": image})

    assert result.get("sensory", {}).get("vision", {}).get("gain") == 1.5


def test_external_attention_gain_overrides_internal_gain():
    cfg = _build_attention_config(gain=1.5)
    net = create_full_brain_network(cfg)

    net.update(0.1, external_inputs={})
    image = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    result = net.update(0.1, external_inputs={"image": image, "attention_gain": 0.8})

    assert result.get("sensory", {}).get("vision", {}).get("gain") == 0.8

