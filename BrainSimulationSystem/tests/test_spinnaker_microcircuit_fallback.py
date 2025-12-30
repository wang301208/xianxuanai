# -*- coding: utf-8 -*-

import importlib.util

import pytest

from BrainSimulationSystem.models.enums import BrainRegion
from BrainSimulationSystem.models.microcircuits import create_microcircuit_for_region
from BrainSimulationSystem.models.regions import PhysiologicalBrainRegion


def test_spinnaker_microcircuit_factory_requires_optional_dependency():
    if importlib.util.find_spec("spynnaker8") is not None:
        pytest.skip("spynnaker8 is installed; skipping environment-dependent runtime test")

    with pytest.raises(RuntimeError) as excinfo:
        create_microcircuit_for_region(
            BrainRegion.VISUAL_CORTEX,
            {"enabled": True, "model": "spinnaker", "preset": "single_region_spiking", "params": {}},
        )

    assert "spynnaker8" in str(excinfo.value).lower()


def test_spinnaker_region_gracefully_falls_back_when_missing_dependency():
    if importlib.util.find_spec("spynnaker8") is not None:
        pytest.skip("spynnaker8 is installed; skipping environment-dependent runtime test")

    region = PhysiologicalBrainRegion(
        BrainRegion.VISUAL_CORTEX,
        {"microcircuit": {"enabled": True, "model": "spinnaker", "preset": "single_region_spiking"}},
    )
    assert region.microcircuit is None

    out = region.update(1.0, {"drive": 1.0}, {})
    assert 0.0 <= float(out.get("activation", 0.0) or 0.0) <= 1.0
    assert out.get("microcircuit") == {}

