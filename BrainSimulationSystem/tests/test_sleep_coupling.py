# -*- coding: utf-8 -*-

import asyncio

from BrainSimulationSystem.models.architecture import CognitiveArchitecture
from BrainSimulationSystem.models.enums import BrainRegion


def _run(coro):
    return asyncio.run(coro)


def test_sleep_synaptic_homeostasis_scales_microcircuit_weights():
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

    config = {
        "runtime": {
            "region_update": {"parallel": {"enabled": False}},
            "sleep": {
                "enabled": True,
                "auto": False,
                "synaptic_homeostasis": {"downscale_rate_per_s": 1000.0, "exc_only": True},
                "replay": {"enabled": False},
            },
        },
        "brain_regions": {region.value: {"microcircuit": False} for region in major_regions},
    }

    config["brain_regions"]["hippocampus"]["microcircuit"] = {
        "enabled": True,
        "model": "biophysical",
        "preset": "single_region_spiking",
        "params": {
            "seed": 1,
            "neurons_per_region": 12,
            "neuron_model": "hybrid",
            "synapse_model": "receptor",
            "stp_enabled": False,
            "stdp_enabled": False,
            "baseline_current_mean": 0.0,
            "baseline_current_std": 0.0,
            "noise_std": 0.0,
            "intra_connection_prob": 0.5,
            "inter_connection_prob": 0.0,
            "max_delay_ms": 5.0,
        },
        "cfg": {"input_gain": 0.0},
    }

    architecture = CognitiveArchitecture(config)

    architecture.set_sleep_stage("wake")
    res1 = _run(architecture.process_cognitive_cycle(1.0, {}, {}))
    w1 = (
        res1["region_activities"]["hippocampus"]
        .get("microcircuit", {})
        .get("state", {})
        .get("weights", {})
    )
    assert int(w1.get("synapse_count", 0) or 0) > 0
    exc_mean_1 = float(w1.get("exc_mean", 0.0) or 0.0)

    architecture.set_sleep_stage("n3")
    _run(architecture.process_cognitive_cycle(1.0, {}, {}))  # apply scaling at end of the cycle
    res3 = _run(architecture.process_cognitive_cycle(1.0, {}, {}))
    w3 = (
        res3["region_activities"]["hippocampus"]
        .get("microcircuit", {})
        .get("state", {})
        .get("weights", {})
    )
    exc_mean_3 = float(w3.get("exc_mean", 0.0) or 0.0)

    assert exc_mean_3 < exc_mean_1


def test_sleep_replay_injects_inputs_in_event_driven_mode():
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

    config = {
        "runtime": {
            "region_update": {
                "mode": "event_driven",
                "parallel": {"enabled": False},
                "event_driven": {
                    "activation_threshold": 0.05,
                    "input_epsilon": 1e-6,
                    "max_pending_events": 10_000,
                    "max_events_per_cycle": 1_000,
                },
            },
            "sleep": {
                "enabled": True,
                "auto": False,
                "force_stage": "rem",
                "replay": {"enabled": True, "mode": "interval", "interval_ms": 0.0, "input_strength": 1.0},
            },
        },
        "brain_regions": {region.value: {"microcircuit": False} for region in major_regions},
    }

    architecture = CognitiveArchitecture(config)

    res = _run(architecture.process_cognitive_cycle(1.0, {}, {}))

    assert res.get("sleep", {}).get("stage") == "rem"
    assert res.get("sleep", {}).get("replay", {}).get("triggered", False)
    assert res.get("event_driven", {}).get("regions_updated") == 2

    assert not res["region_activities"]["hippocampus"].get("skipped", False)
    assert not res["region_activities"]["prefrontal_cortex"].get("skipped", False)

