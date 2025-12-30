# -*- coding: utf-8 -*-

import json

from BrainSimulationSystem.models.architecture import CognitiveArchitecture
from BrainSimulationSystem.models.enums import BrainRegion


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


def test_parameter_mapping_calibration_file_sets_neurons_per_region(tmp_path) -> None:
    calibration_file = tmp_path / "calibration.json"
    calibration_file.write_text(
        json.dumps({"neurons_per_region": {BrainRegion.THALAMUS.value: 33}}), encoding="utf-8"
    )

    cfg = {
        "runtime": {
            "region_update": {"mode": "event_driven", "parallel": {"enabled": False}},
            "parameter_mapping": {
                "enabled": True,
                "calibration": {"enabled": True, "source": "file", "path": str(calibration_file)},
            },
        },
        "brain_regions": {region.value: {"microcircuit": False} for region in _major_regions()},
    }
    cfg["brain_regions"][BrainRegion.THALAMUS.value] = {"microcircuit": True}

    arch = CognitiveArchitecture(cfg)
    micro = arch.brain_regions[BrainRegion.THALAMUS].microcircuit
    assert micro is not None
    assert int(getattr(micro, "params", {}).get("neurons_per_region")) == 33


def test_parameter_mapping_calibration_atlas_respects_explicit_neurons_per_region() -> None:
    cfg = {
        "runtime": {
            "region_update": {"mode": "event_driven", "parallel": {"enabled": False}},
            "parameter_mapping": {
                "enabled": True,
                "calibration": {
                    "enabled": True,
                    "source": "brain_atlas_default",
                    "target_mean_neurons_per_region": 120,
                    "min_neurons": 1,
                    "max_neurons": 200,
                    "override_existing": False,
                },
            },
        },
        "brain_regions": {region.value: {"microcircuit": False} for region in _major_regions()},
    }
    cfg["brain_regions"][BrainRegion.THALAMUS.value] = {"microcircuit": True}
    cfg["brain_regions"][BrainRegion.HIPPOCAMPUS.value] = {
        "microcircuit": {"enabled": True, "model": "biophysical", "params": {"neurons_per_region": 5}}
    }

    arch = CognitiveArchitecture(cfg)

    th = arch.brain_regions[BrainRegion.THALAMUS].microcircuit
    assert th is not None
    th_n = int(getattr(th, "params", {}).get("neurons_per_region") or 0)
    assert th_n > 0

    hpc = arch.brain_regions[BrainRegion.HIPPOCAMPUS].microcircuit
    assert hpc is not None
    assert int(getattr(hpc, "params", {}).get("neurons_per_region")) == 5

    state = arch.get_cognitive_state()
    applied = state.get("parameter_mapping", {}).get("applied", {})
    assert BrainRegion.THALAMUS.value in applied
    assert applied[BrainRegion.THALAMUS.value].get("calibration", {}).get("neurons_per_region") == th_n

