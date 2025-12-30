# -*- coding: utf-8 -*-

import importlib.util

import pytest

from BrainSimulationSystem.models.enums import BrainRegion
from BrainSimulationSystem.models.microcircuits import create_microcircuit_for_region
from BrainSimulationSystem.models.regions import PhysiologicalBrainRegion


def test_loihi_microcircuit_factory_requires_optional_dependency():
    if importlib.util.find_spec("nengo_loihi") is not None:
        pytest.skip("nengo_loihi is installed; skipping environment-dependent runtime test")

    with pytest.raises(RuntimeError) as excinfo:
        create_microcircuit_for_region(
            BrainRegion.BASAL_GANGLIA,
            {"enabled": True, "model": "loihi", "preset": "single_region_spiking", "params": {}},
        )

    assert "nengo_loihi" in str(excinfo.value).lower()


def test_loihi_region_gracefully_falls_back_when_missing_dependency():
    if importlib.util.find_spec("nengo_loihi") is not None:
        pytest.skip("nengo_loihi is installed; skipping environment-dependent runtime test")

    region = PhysiologicalBrainRegion(
        BrainRegion.BASAL_GANGLIA,
        {"microcircuit": {"enabled": True, "model": "loihi", "preset": "single_region_spiking"}},
    )
    assert region.microcircuit is None

    out = region.update(1.0, {"drive": 1.0}, {})
    assert 0.0 <= float(out.get("activation", 0.0) or 0.0) <= 1.0
    assert out.get("microcircuit") == {}

