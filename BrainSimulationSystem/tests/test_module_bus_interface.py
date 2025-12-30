# -*- coding: utf-8 -*-

import asyncio

from BrainSimulationSystem.core.module_interface import ModuleSignal, ModuleTopic
from BrainSimulationSystem.models.architecture import CognitiveArchitecture
from BrainSimulationSystem.models.enums import BrainRegion


def _run(coro):
    return asyncio.run(coro)


def _major_regions():
    return [
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


def test_module_bus_top_down_neuromodulation_override_applies() -> None:
    cfg = {
        "runtime": {
            "region_update": {"mode": "event_driven", "parallel": {"enabled": False}},
            "module_bus": {"enabled": True, "manage_cycle": True, "export_in_results": True},
        },
        "brain_regions": {region.value: {"microcircuit": False} for region in _major_regions()},
    }

    arch = CognitiveArchitecture(cfg)
    arch.queue_module_signal(
        ModuleSignal(
            topic=ModuleTopic.CONTROL_TOP_DOWN,
            source="test",
            payload={"neuromodulation": {"dopamine": 1.5}},
        )
    )

    res = _run(arch.process_cognitive_cycle(1.0, {}, {}))
    assert float(res.get("neuromodulation", {}).get("dopamine")) == 1.5
    assert "module_bus" in res


def test_module_bus_microcircuit_command_adjusts_input_gain() -> None:
    cfg = {
        "runtime": {
            "region_update": {"mode": "event_driven", "parallel": {"enabled": False}},
            "module_bus": {"enabled": True, "manage_cycle": True, "export_in_results": False},
        },
        "brain_regions": {region.value: {"microcircuit": False} for region in _major_regions()},
    }
    cfg["brain_regions"][BrainRegion.VISUAL_CORTEX.value] = {
        "microcircuit": {
            "enabled": True,
            "model": "biophysical",
            "preset": "single_region_spiking",
            "params": {"neurons_per_region": 5, "neuron_model": "izhikevich"},
            "cfg": {"input_gain": 10.0},
        }
    }

    arch = CognitiveArchitecture(cfg)
    micro = arch.brain_regions[BrainRegion.VISUAL_CORTEX].microcircuit
    assert micro is not None
    before = float(getattr(micro, "_input_gain", 0.0) or 0.0)
    assert before == 10.0

    arch.queue_module_signal(
        ModuleSignal(
            topic=ModuleTopic.MICROCIRCUIT_COMMAND,
            source="test",
            payload={"target_region": "visual_cortex", "control": {"input_gain_scale": 0.5}},
        )
    )

    _run(arch.process_cognitive_cycle(1.0, {"visual": 0.1}, {}))
    after = float(getattr(micro, "_input_gain", 0.0) or 0.0)
    assert after == 5.0

