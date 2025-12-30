# -*- coding: utf-8 -*-

import asyncio

from BrainSimulationSystem.models.architecture import CognitiveArchitecture
from BrainSimulationSystem.models.enums import BrainRegion


def _run(coro):
    return asyncio.run(coro)


def test_event_driven_scheduler_delivers_delayed_connection_events():
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
            }
        },
        "brain_regions": {region.value: {"microcircuit": False} for region in major_regions},
    }

    architecture = CognitiveArchitecture(config)

    # Force a short delay so we can observe delivery at the next cycle.
    visual = architecture.brain_regions[BrainRegion.VISUAL_CORTEX]
    visual.connection_strengths[BrainRegion.PARIETAL_CORTEX]["delay"] = 1.0

    res1 = _run(architecture.process_cognitive_cycle(1.0, {"visual": 10.0}, {}))
    assert res1.get("event_driven", {}).get("regions_updated") == 1
    assert res1.get("event_driven", {}).get("queue_depth", 0) > 0

    res2 = _run(architecture.process_cognitive_cycle(1.0, {}, {}))
    parietal = res2["region_activities"]["parietal_cortex"]
    assert not parietal.get("skipped", False)

