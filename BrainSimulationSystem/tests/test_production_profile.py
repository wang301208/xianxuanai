"""Integration tests for the production configuration profile."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.brain_simulation import BrainSimulation  # noqa: E402


def _compute_readiness_from_capabilities(capabilities):
    readiness = 0.0

    full_brain = capabilities.get("full_brain_ready", {}) if isinstance(capabilities, dict) else {}
    if bool(full_brain.get("is_full_brain_scale", False)):
        readiness += 0.3
    else:
        columns = float(full_brain.get("cortical_columns_count", 0))
        long_range = float(full_brain.get("long_range_connections_count", 0))
        readiness += min(0.2, (columns / 1000.0) + (long_range / 5000.0))

    physiology = capabilities.get("physiology", {})
    if bool(physiology.get("thalamocortical_enabled", False)):
        readiness += 0.05
    if bool(physiology.get("hippocampus_pfc_enabled", False)):
        readiness += 0.05
    glia_vascular = physiology.get("glia_vascular", {}) if isinstance(physiology, dict) else {}
    if isinstance(glia_vascular, dict) and bool(glia_vascular.get("enabled", False)):
        readiness += min(0.05, float(glia_vascular.get("modulation_strength", 0.0)))

    neuromorphic = capabilities.get("neuromorphic", {})
    if bool(neuromorphic.get("bridge_enabled", False)):
        readiness += 0.05
    if bool(neuromorphic.get("aer_export_enabled", False)):
        readiness += 0.05
    if bool(neuromorphic.get("mapping_export_enabled", False)):
        readiness += 0.05
    if bool(neuromorphic.get("backend_manager_enabled", False)):
        readiness += 0.1

    runtime = capabilities.get("runtime", {})
    monitoring = capabilities.get("monitoring_visualization", {})
    if bool(runtime.get("distributed_enabled", False)):
        readiness += 0.05
    if bool(runtime.get("checkpoint_enabled", False)):
        readiness += 0.05
    if bool(monitoring.get("monitoring_enabled", False)):
        readiness += 0.05
    if bool(monitoring.get("performance_enabled", False)):
        readiness += 0.05

    return readiness


def test_production_profile_readiness_score_high() -> None:
    """Production profile should enable all readiness contributions."""

    simulation = BrainSimulation(profile="production")
    capabilities = simulation.get_capabilities_report()
    readiness = _compute_readiness_from_capabilities(capabilities)

    assert readiness >= 0.8, f"Expected readiness >= 0.8 for production profile, got {readiness:.2f}"
