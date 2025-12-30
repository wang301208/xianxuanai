# -*- coding: utf-8 -*-

from BrainSimulationSystem.models.enums import BrainRegion
from BrainSimulationSystem.models.regions import PhysiologicalBrainRegion


def test_region_microcircuit_drives_activation_and_exports_state():
    region = PhysiologicalBrainRegion(
        BrainRegion.VISUAL_CORTEX,
        {
            "microcircuit": {
                "enabled": True,
                "model": "biophysical",
                "preset": "single_region_spiking",
                "params": {
                    "seed": 21,
                    "neurons_per_region": 20,
                    "baseline_current_mean": 0.0,
                    "baseline_current_std": 0.0,
                    "noise_std": 0.0,
                    "synapse_model": "receptor",
                },
                "cfg": {"input_gain": 25.0, "target_rate_hz": 20.0, "smooth_tau_ms": 10.0},
            }
        },
    )

    out = region.update(1.0, {"visual": 1.0}, {"dopamine": 1.0})
    assert 0.0 <= float(out.get("activation", -1.0)) <= 1.0

    micro = out.get("microcircuit")
    assert isinstance(micro, dict) and micro
    assert "state" in micro and isinstance(micro["state"], dict)
    assert "spike_count" in micro["state"]


def test_hippocampus_microcircuit_preset_exposes_subfield_rates():
    region = PhysiologicalBrainRegion(
        BrainRegion.HIPPOCAMPUS,
        {
            "microcircuit": {
                "enabled": True,
                "model": "biophysical",
                "preset": "hippocampus_dg_ca3_ca1",
                "params": {
                    "seed": 22,
                    "neurons_per_region": 16,
                    "baseline_current_mean": 0.0,
                    "baseline_current_std": 0.0,
                    "noise_std": 0.0,
                },
                "cfg": {"input_gain": 25.0, "target_rate_hz": 20.0, "smooth_tau_ms": 10.0},
            }
        },
    )

    out = region.update(1.0, {"drive": 1.0}, {})
    micro = out.get("microcircuit") or {}
    region_rates = micro.get("region_rates_hz") or {}
    assert "DG" in region_rates
    assert "CA3" in region_rates
    assert "CA1" in region_rates
