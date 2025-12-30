# -*- coding: utf-8 -*-

import importlib.util

import pytest

from BrainSimulationSystem.models.enums import BrainRegion
from BrainSimulationSystem.models.microcircuits import ShadowMicrocircuit, create_microcircuit_for_region
from BrainSimulationSystem.models.regions import PhysiologicalBrainRegion


def test_pynn_microcircuit_factory_requires_optional_dependency():
    if importlib.util.find_spec("pyNN") is not None:
        pytest.skip("pyNN is installed; skipping environment-dependent runtime test")

    with pytest.raises(RuntimeError) as excinfo:
        create_microcircuit_for_region(
            BrainRegion.HIPPOCAMPUS,
            {
                "enabled": True,
                "model": "pynn",
                "preset": "hippocampus_dg_ca3_ca1",
                "cfg": {"pynn": {"backend": "nest"}},
                "params": {},
            },
        )

    assert "pynn" in str(excinfo.value).lower()


def test_pynn_region_gracefully_falls_back_when_missing_dependency():
    if importlib.util.find_spec("pyNN") is not None:
        pytest.skip("pyNN is installed; skipping environment-dependent runtime test")

    region = PhysiologicalBrainRegion(
        BrainRegion.HIPPOCAMPUS,
        {"microcircuit": {"enabled": True, "model": "pynn", "preset": "hippocampus_dg_ca3_ca1", "cfg": {"pynn": {"backend": "nest"}}}},
    )
    assert region.microcircuit is None

    out = region.update(1.0, {"drive": 1.0}, {})
    assert 0.0 <= float(out.get("activation", 0.0) or 0.0) <= 1.0
    assert out.get("microcircuit") == {}


def test_shadow_microcircuit_falls_back_when_shadow_backend_missing_dependency():
    if importlib.util.find_spec("pyNN") is not None:
        pytest.skip("pyNN is installed; skipping environment-dependent runtime test")

    region = PhysiologicalBrainRegion(
        BrainRegion.HIPPOCAMPUS,
        {
            "microcircuit": {
                "enabled": True,
                "model": "biophysical",
                "preset": "hippocampus_dg_ca3_ca1",
                "shadow": {"enabled": True, "model": "pynn", "preset": "hippocampus_dg_ca3_ca1", "cfg": {"pynn": {"backend": "nest"}}},
            }
        },
    )

    assert region.microcircuit is not None
    assert not isinstance(region.microcircuit, ShadowMicrocircuit)
    cfg = getattr(region.microcircuit, "cfg", {}) or {}
    assert "shadow_error" in cfg

