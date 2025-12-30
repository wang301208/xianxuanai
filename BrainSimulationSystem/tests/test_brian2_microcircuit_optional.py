# -*- coding: utf-8 -*-

import pytest

from BrainSimulationSystem.models.enums import BrainRegion
from BrainSimulationSystem.models.regions import PhysiologicalBrainRegion


def test_brian2_microcircuit_smoke_if_installed():
    pytest.importorskip("brian2")

    region = PhysiologicalBrainRegion(
        BrainRegion.VISUAL_CORTEX,
        {
            "microcircuit": {
                "enabled": True,
                "model": "brian2",
                "preset": "single_region_spiking",
                "params": {
                    "seed": 7,
                    "neurons_per_region": 40,
                    "intra_connection_prob": 0.05,
                    "inter_connection_prob": 0.0,
                },
                "cfg": {
                    "input_gain": 50.0,
                    "target_rate_hz": 20.0,
                    "smooth_tau_ms": 10.0,
                    "brian2": {"resolution_ms": 0.1, "codegen_target": "numpy"},
                },
            }
        },
    )

    out = region.update(1.0, {"drive": 1.0}, {})
    micro = out.get("microcircuit")
    assert isinstance(micro, dict) and micro
    state = micro.get("state")
    assert isinstance(state, dict)
    assert state.get("framework") == "brian2"
    assert "spike_count" in state

