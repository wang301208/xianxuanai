# -*- coding: utf-8 -*-

import asyncio

from BrainSimulationSystem.models.architecture import CognitiveArchitecture
from BrainSimulationSystem.models.enums import BrainRegion


def _run(coro):
    return asyncio.run(coro)


def test_parameter_mapping_maps_module_params_into_microcircuits():
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
            "parameter_mapping": {
                "enabled": True,
                "working_memory": {"neurons_per_item": 2, "min_neurons": 1, "max_neurons": 50},
                "basal_ganglia": {"enable_stdp": True, "dopamine_gain_scale": 5.0},
                "visual": {"use_retina_lgn_v1": True},
            },
        },
        "memory": {"working_memory": {"capacity": 3}},
        "decision": {"learning_rate": 0.2},
        "brain_regions": {region.value: {"microcircuit": False} for region in major_regions},
    }

    cfg["brain_regions"][BrainRegion.PREFRONTAL_CORTEX.value] = {"microcircuit": True}
    cfg["brain_regions"][BrainRegion.BASAL_GANGLIA.value] = {
        "microcircuit": {"enabled": True, "model": "biophysical", "params": {"neurons_per_region": 10}}
    }
    cfg["brain_regions"][BrainRegion.VISUAL_CORTEX.value] = {
        "microcircuit": {"enabled": True, "model": "biophysical", "params": {"neurons_per_region": 5}}
    }

    arch = CognitiveArchitecture(cfg)

    pfc = arch.brain_regions[BrainRegion.PREFRONTAL_CORTEX].microcircuit
    assert pfc is not None
    assert int(getattr(pfc, "params", {}).get("neurons_per_region")) == 6  # capacity(3) * neurons_per_item(2)

    bg = arch.brain_regions[BrainRegion.BASAL_GANGLIA].microcircuit
    assert bg is not None
    assert bool(getattr(bg, "params", {}).get("stdp_enabled")) is True
    assert float(getattr(bg, "params", {}).get("dopamine_stdp_gain")) == 1.0  # 0.2 * 5.0

    vis = arch.brain_regions[BrainRegion.VISUAL_CORTEX].microcircuit
    assert vis is not None
    assert list(getattr(vis, "params", {}).get("regions") or []) == ["RETINA", "LGN", "V1"]
    assert int(getattr(vis, "params", {}).get("neurons_per_region")) == 5

    res = _run(arch.process_cognitive_cycle(1.0, {}, {}))
    assert "parameter_mapping" in arch.get_cognitive_state()
    assert res.get("event_driven", {}).get("regions_updated") == 0

