# -*- coding: utf-8 -*-

from BrainSimulationSystem.models.enums import BrainRegion
from BrainSimulationSystem.models.regions import PhysiologicalBrainRegion


def test_shadow_microcircuit_emits_compare_telemetry():
    region = PhysiologicalBrainRegion(
        BrainRegion.HIPPOCAMPUS,
        {
            "microcircuit": {
                "enabled": True,
                "model": "biophysical",
                "preset": "hippocampus_dg_ca3_ca1",
                "shadow": {"enabled": True, "model": "biophysical", "preset": "hippocampus_dg_ca3_ca1"},
                "shadow_compare": {"activation_abs": 1.0, "rate_hz_abs": 1e6},
            }
        },
    )

    assert region.microcircuit is not None

    out = region.update(1.0, {"DG": 0.6, "CA3": 0.4, "CA1": 0.2}, {"dopamine": 0.5})
    mc = out.get("microcircuit", {})
    assert isinstance(mc, dict)

    state = mc.get("state", {})
    assert isinstance(state, dict)

    telemetry = state.get("shadow_compare")
    assert isinstance(telemetry, dict)
    assert "primary" in telemetry
    assert "shadow" in telemetry or telemetry.get("errors")
    assert "diff" in telemetry or telemetry.get("errors")

