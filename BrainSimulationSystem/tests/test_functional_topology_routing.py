# -*- coding: utf-8 -*-

import asyncio

from BrainSimulationSystem.models.architecture import CognitiveArchitecture
from BrainSimulationSystem.models.enums import BrainRegion


def _run(coro):
    return asyncio.run(coro)


def _base_config(*, enable_topology: bool) -> dict:
    major_regions = [
        BrainRegion.PREFRONTAL_CORTEX,
        BrainRegion.MOTOR_CORTEX,
        BrainRegion.SOMATOSENSORY_CORTEX,
        BrainRegion.VISUAL_CORTEX,
        BrainRegion.AUDITORY_CORTEX,
        BrainRegion.PARIETAL_CORTEX,
        BrainRegion.TEMPORAL_CORTEX,
        BrainRegion.HIPPOCAMPUS,
        BrainRegion.AMYGDALA,
        BrainRegion.THALAMUS,
        BrainRegion.BASAL_GANGLIA,
        BrainRegion.CEREBELLUM,
    ]

    cfg = {
        "runtime": {
            "region_update": {"mode": "event_driven", "parallel": {"enabled": False}},
            "functional_topology": {"enabled": bool(enable_topology)},
        },
        "brain_regions": {region.value: {"microcircuit": False} for region in major_regions},
    }
    return cfg


def test_functional_topology_disabled_does_not_route_module_activity():
    architecture = CognitiveArchitecture(_base_config(enable_topology=False))

    res = _run(
        architecture.process_cognitive_cycle(
            1.0,
            {},
            {"module_activity": {"motor": 1.0}},
        )
    )
    assert res.get("event_driven", {}).get("regions_updated") == 0
    assert "functional_topology" not in res


def test_functional_topology_routes_motor_module_into_mapped_regions():
    architecture = CognitiveArchitecture(_base_config(enable_topology=True))

    res = _run(
        architecture.process_cognitive_cycle(
            1.0,
            {},
            {"module_activity": {"motor": 1.0}},
        )
    )
    assert res.get("event_driven", {}).get("regions_updated") == 3

    topo = res.get("functional_topology")
    assert isinstance(topo, dict)
    assert topo.get("enabled") is True
    assert topo.get("module_activity_in", {}).get("motor") == 1.0

    mapped = topo.get("module_to_regions", {}).get("motor", [])
    assert set(mapped) == {BrainRegion.MOTOR_CORTEX.value, BrainRegion.BASAL_GANGLIA.value, BrainRegion.CEREBELLUM.value}

    module_out = topo.get("module_activity_out", {}).get("motor")
    assert isinstance(module_out, (int, float))

