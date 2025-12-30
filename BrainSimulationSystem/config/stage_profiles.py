"""Curriculum-aware brain configuration profiles.

This module defines scalable brain architecture profiles that start with a
miniature “baby brain” and incrementally expand toward the full-scale
configuration described in :mod:`BrainSimulationSystem.core.full_brain_architecture`.
Each stage piggybacks on the existing ``default_config`` profiles while
injecting region-specific volume and neuron-density overrides that keep the
network trainable at small scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

from BrainSimulationSystem.models.enums import BrainRegion
from BrainSimulationSystem.config.default_config import get_config


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegionScaling:
    """Scaling factors applied to a canonical brain region template."""

    volume: float = 1.0
    surface_area: float = 1.0
    thickness: float = 1.0
    neuron_density: float = 1.0

    def clamp(self, *, min_thickness: float = 0.5) -> "RegionScaling":
        """Return a scaling where thickness never drops below ``min_thickness``."""

        thickness = max(self.thickness, min_thickness)
        return RegionScaling(
            volume=self.volume,
            surface_area=self.surface_area,
            thickness=thickness,
            neuron_density=self.neuron_density,
        )


@dataclass(frozen=True)
class StageRuntimePolicy:
    """Constraints that determine when a stage promotion is allowed."""

    max_latency_s: float
    promotion_window: int
    min_success_rate: float
    bottleneck_latency_s: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "max_latency_s": self.max_latency_s,
            "promotion_window": self.promotion_window,
            "min_success_rate": self.min_success_rate,
            "bottleneck_latency_s": self.bottleneck_latency_s,
        }


@dataclass(frozen=True)
class StageSpec:
    """Declarative definition of a curriculum stage."""

    key: str
    label: str
    description: str
    base_profile: str
    default_scaling: RegionScaling
    runtime_policy: StageRuntimePolicy
    region_overrides: Mapping[BrainRegion, RegionScaling] = field(default_factory=dict)
    config_overrides: Mapping[str, object] = field(default_factory=dict)
    runtime_overrides: Mapping[str, object] = field(default_factory=dict)
    module_budget: Mapping[str, int] = field(default_factory=dict)
    expansion_policy: Mapping[str, float] = field(default_factory=dict)

    def scaling_for(self, region: BrainRegion) -> RegionScaling:
        """Return the scaling factors for ``region``."""

        scaling = self.region_overrides.get(region, self.default_scaling)
        return scaling.clamp()


# --------------------------------------------------------------------------- #
# Canonical anatomical templates (volumes in mm^3, density neurons/mm^3)
# --------------------------------------------------------------------------- #


_GENERIC_REGION = {
    "volume": 5_000.0,
    "surface_area": 1_000.0,
    "thickness": 2.5,
    "neuron_density": 100_000,
}

_CANONICAL_REGIONS: Dict[BrainRegion, Dict[str, float]] = {
    BrainRegion.PREFRONTAL_CORTEX: {
        "volume": 15_000.0,
        "surface_area": 2_000.0,
        "thickness": 3.0,
        "neuron_density": 80_000,
    },
    BrainRegion.HIPPOCAMPUS: {
        "volume": 4_000.0,
        "surface_area": 500.0,
        "thickness": 1.0,
        "neuron_density": 150_000,
    },
    BrainRegion.VISUAL_CORTEX: {
        "volume": 20_000.0,
        "surface_area": 3_000.0,
        "thickness": 2.0,
        "neuron_density": 120_000,
    },
    BrainRegion.MOTOR_CORTEX: {
        "volume": 8_000.0,
        "surface_area": 1_200.0,
        "thickness": 4.0,
        "neuron_density": 100_000,
    },
    BrainRegion.THALAMUS: {
        "volume": 6_000.0,
        "surface_area": 800.0,
        "thickness": 0.8,
        "neuron_density": 200_000,
    },
    BrainRegion.CEREBELLUM: {
        "volume": 150_000.0,
        "surface_area": 10_000.0,
        "thickness": 1.5,
        "neuron_density": 300_000,
    },
    BrainRegion.PARIETAL_CORTEX: {
        "volume": 11_000.0,
        "surface_area": 1_700.0,
        "thickness": 2.5,
        "neuron_density": 90_000,
    },
    BrainRegion.TEMPORAL_CORTEX: {
        "volume": 12_000.0,
        "surface_area": 1_900.0,
        "thickness": 2.6,
        "neuron_density": 95_000,
    },
    BrainRegion.AUDITORY_CORTEX: {
        "volume": 7_000.0,
        "surface_area": 900.0,
        "thickness": 2.3,
        "neuron_density": 105_000,
    },
    BrainRegion.SOMATOSENSORY_CORTEX: {
        "volume": 9_000.0,
        "surface_area": 1_200.0,
        "thickness": 2.8,
        "neuron_density": 110_000,
    },
    BrainRegion.AMYGDALA: {
        "volume": 2_500.0,
        "surface_area": 400.0,
        "thickness": 0.8,
        "neuron_density": 180_000,
    },
    BrainRegion.BASAL_GANGLIA: {
        "volume": 5_500.0,
        "surface_area": 700.0,
        "thickness": 1.2,
        "neuron_density": 160_000,
    },
}


# --------------------------------------------------------------------------- #
# Stage definitions
# --------------------------------------------------------------------------- #


_STAGE_SPECS: Dict[str, StageSpec] = {
    "infant": StageSpec(
        key="infant",
        label="Stage 0 · Infant",
        description="Minimal baby brain configuration for initial curriculum steps.",
        base_profile="prototype",
        default_scaling=RegionScaling(
            volume=0.02,
            surface_area=0.03,
            thickness=0.7,
            neuron_density=0.08,
        ),
        region_overrides={
            BrainRegion.HIPPOCAMPUS: RegionScaling(
                volume=0.05, surface_area=0.05, thickness=0.8, neuron_density=0.12
            ),
            BrainRegion.PREFRONTAL_CORTEX: RegionScaling(
                volume=0.03, surface_area=0.04, thickness=0.75, neuron_density=0.1
            ),
            BrainRegion.CEREBELLUM: RegionScaling(
                volume=0.01, surface_area=0.015, thickness=0.7, neuron_density=0.05
            ),
        },
        runtime_policy=StageRuntimePolicy(
            max_latency_s=0.8,
            promotion_window=60,
            min_success_rate=0.65,
            bottleneck_latency_s=0.9,
        ),
        config_overrides={
            "scope": {
                "columns_per_region": 1,
                "column_total_neurons": 64,
            },
            "perception": {
                "noise_level": 0.03,
                "vision": {
                    "enabled": True,
                    "return_feature_maps": False,
                    "model": {
                        "backend": "numpy",
                        "input_size": (64, 64),
                        "feature_dim": 32,
                    },
                },
                "auditory": {
                    "enabled": True,
                    "model": {
                        "backend": "numpy",
                        "sample_rate": 8000,
                        "n_mels": 20,
                        "feature_dim": 64,
                    },
                },
                "somatosensory": {"enabled": False},
                "structured": {"enabled": False},
                "multimodal_fusion": {"enabled": False},
            },
            "memory": {
                "system": {
                    "working_memory": {
                        "strategy": "priority",
                        "capacity": 7,
                        "decay_rate": 0.15,
                    },
                    "hippocampal": {
                        "ca1_capacity": 256,
                        "ca3_capacity": 128,
                        "dg_capacity": 1_024,
                        "reconsolidation_window": 600,
                    },
                    "neocortical": {
                        "max_concepts": 512,
                        "embedding_size": 64,
                    },
                    "semantic_bridge": {
                        "enabled": False,
                        "max_associations": 8,
                        "relation_strength": 0.25,
                    },
                    "consolidation_max_items": 4,
                    "sleep_consolidation_max_items": 8,
                },
                "experience": {
                    "enabled": True,
                    "interval_steps": 2,
                    "memory_type": "EPISODIC",
                    "knowledge_graph": {"enabled": False},
                },
            },
            "learning": {
                "interactive_language_loop": {
                    "mentor_interval": 1,
                    "max_steps": 24,
                },
                "mentor": {
                    "enabled": True,
                    "reward_bonus": 0.15,
                    "utterance": "Try this action next.",
                },
                "reward_shaping": {
                    "enabled": True,
                    "mentor_bonus": 0.08,
                    "critique_penalty": 0.04,
                    "low_confidence_penalty": 0.02,
                    "reward_channel_weight": 0.05,
                    "success_bonus": 0.0,
                },
                "replay_buffer": {"capacity": 512},
                "offline_training": {
                    "enabled": True,
                    "algorithm": "dqn",
                    "train_every_episodes": 2,
                    "batch_size": 8,
                    "updates": 2,
                    "mentor_fraction": 0.7,
                },
            },
            "environment": {
                "kind": "toy_room",
                "episode_length": 24,
                "object_count": 3,
                "include_audio": True,
            },
        },
        runtime_overrides={
            "performance": {"max_region_time": 0.12, "evaluation_interval": 5},
            "resource_management": {
                "partitioning": {"max_neurons_per_partition": 2_000},
                "memory_budget_gb": 1,
            },
        },
        module_budget={"max_active_modules": 4, "max_dynamic_modules": 1},
        expansion_policy={"threshold": 0.85, "cooldown_steps": 20},
    ),
    "juvenile": StageSpec(
        key="juvenile",
        label="Stage 1 · Juvenile",
        description="Adds richer perception-memory loops once stability improves.",
        base_profile="prototype",
        default_scaling=RegionScaling(
            volume=0.1,
            surface_area=0.12,
            thickness=0.85,
            neuron_density=0.25,
        ),
        region_overrides={
            BrainRegion.VISUAL_CORTEX: RegionScaling(
                volume=0.14, surface_area=0.16, thickness=0.9, neuron_density=0.3
            ),
            BrainRegion.AUDITORY_CORTEX: RegionScaling(
                volume=0.12, surface_area=0.14, thickness=0.9, neuron_density=0.28
            ),
            BrainRegion.HIPPOCAMPUS: RegionScaling(
                volume=0.18, surface_area=0.18, thickness=0.9, neuron_density=0.32
            ),
            BrainRegion.CEREBELLUM: RegionScaling(
                volume=0.08, surface_area=0.09, thickness=0.85, neuron_density=0.2
            ),
        },
        runtime_policy=StageRuntimePolicy(
            max_latency_s=0.45,
            promotion_window=80,
            min_success_rate=0.75,
            bottleneck_latency_s=0.5,
        ),
        config_overrides={
            "scope": {
                "columns_per_region": 2,
                "column_total_neurons": 96,
            },
            "perception": {
                "noise_level": 0.02,
                "vision": {
                    "enabled": True,
                    "return_feature_maps": True,
                    "model": {
                        "backend": "auto",
                        "input_size": (128, 128),
                        "feature_dim": 96,
                    },
                },
                "auditory": {
                    "enabled": True,
                    "model": {
                        "backend": "auto",
                        "sample_rate": 16000,
                        "n_mels": 40,
                        "feature_dim": 128,
                    },
                },
                "somatosensory": {"enabled": False},
                "structured": {"enabled": False},
                "multimodal_fusion": {"enabled": True},
            },
            "memory": {
                "system": {
                    "working_memory": {
                        "strategy": "indexed",
                        "capacity": 16,
                        "decay_rate": 0.12,
                        "min_weight": 0.1,
                    },
                    "hippocampal": {
                        "ca1_capacity": 1_024,
                        "ca3_capacity": 512,
                        "dg_capacity": 4_096,
                        "reconsolidation_window": 900,
                    },
                    "neocortical": {
                        "max_concepts": 2_048,
                        "embedding_size": 128,
                    },
                    "semantic_bridge": {
                        "enabled": True,
                        "max_associations": 12,
                        "relation_strength": 0.35,
                    },
                    "consolidation_max_items": 12,
                    "sleep_consolidation_max_items": 32,
                },
                "experience": {
                    "enabled": True,
                    "interval_steps": 1,
                    "memory_type": "EPISODIC",
                    "knowledge_graph": {"enabled": True, "max_triples": 32},
                },
            },
            "learning": {
                "interactive_language_loop": {
                    "mentor_interval": 4,
                    "max_steps": 32,
                },
                "mentor": {
                    "enabled": True,
                    "reward_bonus": 0.08,
                    "utterance": "Here's a hint.",
                },
                "reward_shaping": {
                    "enabled": True,
                    "mentor_bonus": 0.05,
                    "critique_penalty": 0.05,
                    "low_confidence_penalty": 0.02,
                    "reward_channel_weight": 0.08,
                    "success_bonus": 0.15,
                },
                "replay_buffer": {"capacity": 2_048},
                "offline_training": {
                    "enabled": True,
                    "algorithm": "dqn",
                    "train_every_episodes": 1,
                    "batch_size": 16,
                    "updates": 4,
                    "mentor_fraction": 0.35,
                },
            },
            "environment": {
                "kind": "toy_teacher",
                "episode_length": 32,
                "object_count": 4,
                "include_audio": True,
            },
            "modules": {
                "components": {
                    "language_hub": {
                        "semantic_fallback": {"enabled": False},
                        "llm_service": {"enabled": False},
                    }
                }
            },
        },
        runtime_overrides={
            "performance": {"max_region_time": 0.08},
            "resource_management": {
                "partitioning": {"max_neurons_per_partition": 5_000},
                "memory_budget_gb": 4,
            },
        },
        module_budget={"max_active_modules": 8, "max_dynamic_modules": 2},
        expansion_policy={"threshold": 0.75, "cooldown_steps": 15},
    ),
    "adolescent": StageSpec(
        key="adolescent",
        label="Stage 2 · Adolescent",
        description="Mid-scale network aligned with the research profile.",
        base_profile="research",
        default_scaling=RegionScaling(
            volume=0.45,
            surface_area=0.5,
            thickness=0.95,
            neuron_density=0.55,
        ),
        region_overrides={
            BrainRegion.PREFRONTAL_CORTEX: RegionScaling(
                volume=0.55, surface_area=0.6, thickness=1.0, neuron_density=0.65
            ),
            BrainRegion.TEMPORAL_CORTEX: RegionScaling(
                volume=0.5, surface_area=0.55, thickness=1.0, neuron_density=0.6
            ),
            BrainRegion.CEREBELLUM: RegionScaling(
                volume=0.4, surface_area=0.45, thickness=0.95, neuron_density=0.5
            ),
        },
        runtime_policy=StageRuntimePolicy(
            max_latency_s=0.22,
            promotion_window=120,
            min_success_rate=0.82,
            bottleneck_latency_s=0.28,
        ),
        config_overrides={
            "scope": {
                "columns_per_region": 3,
                "column_total_neurons": 128,
            },
            "perception": {
                "noise_level": 0.015,
                "vision": {
                    "enabled": True,
                    "return_feature_maps": True,
                    "model": {
                        "backend": "auto",
                        "input_size": (192, 192),
                        "feature_dim": 128,
                    },
                },
                "auditory": {
                    "enabled": True,
                    "model": {
                        "backend": "auto",
                        "sample_rate": 16000,
                        "n_mels": 64,
                        "feature_dim": 192,
                    },
                },
                "somatosensory": {"enabled": True},
                "structured": {"enabled": True},
                "multimodal_fusion": {"enabled": True},
            },
            "memory": {
                "system": {
                    "working_memory": {
                        "strategy": "indexed",
                        "capacity": 32,
                        "decay_rate": 0.1,
                        "min_weight": 0.08,
                    },
                    "hippocampal": {
                        "ca1_capacity": 4_096,
                        "ca3_capacity": 2_048,
                        "dg_capacity": 16_384,
                        "reconsolidation_window": 1_800,
                    },
                    "neocortical": {
                        "max_concepts": 10_000,
                        "embedding_size": 256,
                    },
                    "semantic_bridge": {
                        "enabled": True,
                        "max_associations": 24,
                        "relation_strength": 0.4,
                    },
                    "consolidation_max_items": 24,
                    "sleep_consolidation_max_items": 64,
                },
                "experience": {
                    "enabled": True,
                    "interval_steps": 1,
                    "memory_type": "EPISODIC",
                    "knowledge_graph": {"enabled": True, "max_triples": 48},
                },
            },
            "learning": {
                "interactive_language_loop": {
                    "mentor_interval": 10,
                    "max_steps": 48,
                },
                "mentor": {
                    "enabled": True,
                    "reward_bonus": 0.03,
                    "utterance": None,
                },
                "reward_shaping": {
                    "enabled": True,
                    "mentor_bonus": 0.02,
                    "critique_penalty": 0.05,
                    "low_confidence_penalty": 0.015,
                    "reward_channel_weight": 0.1,
                    "success_bonus": 0.12,
                },
                "replay_buffer": {"capacity": 8_192},
                "offline_training": {
                    "enabled": True,
                    "algorithm": "dqn",
                    "train_every_episodes": 1,
                    "batch_size": 32,
                    "updates": 8,
                    "mentor_fraction": 0.15,
                },
            },
            "environment": {
                "kind": "open_world",
                "episode_length": 48,
                "grid_world": {"size": 9},
                "unity": {"file_name": "", "no_graphics": True, "time_scale": 1.0},
            },
            "modules": {
                "components": {
                    "language_hub": {
                        "semantic_fallback": {"enabled": True},
                        "llm_service": {"provider": "internal_pipeline"},
                    }
                }
            },
            "self_model": {"enabled": True},
            "metacognition": {"enabled": True},
            "meta_reasoning": {"enabled": True},
        },
        runtime_overrides={
            "performance": {"max_region_time": 0.05, "evaluation_interval": 3},
            "resource_management": {
                "partitioning": {"max_neurons_per_partition": 50_000},
                "memory_budget_gb": 32,
            },
        },
        module_budget={"max_active_modules": 16, "max_dynamic_modules": 4},
        expansion_policy={"threshold": 0.7, "cooldown_steps": 10},
    ),
    "full": StageSpec(
        key="full",
        label="Stage 3 · Full Brain",
        description="Targets the production/full_brain blueprint with anatomy-aligned scaling.",
        base_profile="production",
        default_scaling=RegionScaling(
            volume=1.0,
            surface_area=1.0,
            thickness=1.0,
            neuron_density=1.0,
        ),
        runtime_policy=StageRuntimePolicy(
            max_latency_s=0.12,
            promotion_window=160,
            min_success_rate=0.9,
            bottleneck_latency_s=0.15,
        ),
        runtime_overrides={
            "performance": {"max_region_time": 0.02, "evaluation_interval": 2},
            "resource_management": {
                "partitioning": {"max_neurons_per_partition": 2_000_000},
                "memory_budget_gb": 512,
            },
        },
        config_overrides={
            "scope": {
                "columns_per_region": 4,
                "column_total_neurons": 160,
            },
            "perception": {
                "noise_level": 0.01,
                "vision": {
                    "enabled": True,
                    "return_feature_maps": True,
                    "model": {
                        "backend": "auto",
                        "input_size": (224, 224),
                        "feature_dim": 192,
                    },
                },
                "auditory": {
                    "enabled": True,
                    "model": {
                        "backend": "auto",
                        "sample_rate": 16000,
                        "n_mels": 80,
                        "feature_dim": 256,
                    },
                },
                "somatosensory": {"enabled": True},
                "structured": {"enabled": True},
                "multimodal_fusion": {"enabled": True},
            },
            "memory": {
                "system": {
                    "working_memory": {
                        "strategy": "indexed",
                        "capacity": 64,
                        "decay_rate": 0.08,
                        "min_weight": 0.05,
                    },
                    "hippocampal": {
                        "ca1_capacity": 20_000,
                        "ca3_capacity": 10_000,
                        "dg_capacity": 60_000,
                        "reconsolidation_window": 3_600,
                    },
                    "neocortical": {
                        "max_concepts": 50_000,
                        "embedding_size": 300,
                    },
                    "semantic_bridge": {
                        "enabled": True,
                        "max_associations": 32,
                        "relation_strength": 0.45,
                    },
                    "consolidation_max_items": 48,
                    "sleep_consolidation_max_items": 128,
                },
                "experience": {
                    "enabled": True,
                    "interval_steps": 1,
                    "memory_type": "EPISODIC",
                    "knowledge_graph": {"enabled": True, "max_triples": 64},
                },
            },
            "learning": {
                "interactive_language_loop": {
                    "mentor_interval": 0,
                    "max_steps": 64,
                },
                "mentor": {
                    "enabled": False,
                    "reward_bonus": 0.0,
                    "utterance": None,
                },
                "reward_shaping": {
                    "enabled": True,
                    "mentor_bonus": 0.0,
                    "critique_penalty": 0.05,
                    "low_confidence_penalty": 0.01,
                    "reward_channel_weight": 0.12,
                    "success_bonus": 0.08,
                },
                "replay_buffer": {"capacity": 16_384},
                "offline_training": {
                    "enabled": True,
                    "algorithm": "dqn",
                    "train_every_episodes": 1,
                    "batch_size": 64,
                    "updates": 12,
                    "mentor_fraction": 0.0,
                },
            },
            "environment": {
                "kind": "open_world",
                "episode_length": 64,
                "grid_world": {"size": 15},
                "unity": {"file_name": "", "no_graphics": True, "time_scale": 1.0},
            },
            "modules": {
                "components": {
                    "language_hub": {
                        "semantic_fallback": {"enabled": True},
                        "llm_service": {"provider": "internal_pipeline"},
                    }
                }
            },
            "self_model": {"enabled": True},
            "metacognition": {"enabled": True},
            "meta_reasoning": {"enabled": True},
            "planner": {"enabled": True},
        },
        module_budget={"max_active_modules": 32, "max_dynamic_modules": 8},
        expansion_policy={"threshold": 0.65, "cooldown_steps": 5},
    ),
}


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def list_stages() -> List[str]:
    """Return the available stage identifiers in curriculum order."""

    return list(_STAGE_SPECS.keys())


def get_stage_spec(stage: str) -> StageSpec:
    """Look up a stage definition by identifier."""

    key = stage.lower()
    if key not in _STAGE_SPECS:
        available = ", ".join(sorted(_STAGE_SPECS))
        raise ValueError(f"Unknown stage '{stage}'. Available stages: {available}")
    return _STAGE_SPECS[key]


def build_stage_config(
    stage: str,
    *,
    overrides: MutableMapping[str, object] | None = None,
    base_profile: str | None = None,
) -> Dict[str, object]:
    """Return a merged config for ``stage`` (optionally layering ``overrides``)."""

    spec = get_stage_spec(stage)
    profile = base_profile or spec.base_profile
    base_config = deepcopy(get_config(profile=profile))
    region_block, neuron_estimate = _build_brain_region_block(spec)

    curriculum_payload: Dict[str, object] = {
        "stage": spec.key,
        "label": spec.label,
        "description": spec.description,
        "runtime_policy": spec.runtime_policy.to_dict(),
        "module_budget": dict(spec.module_budget),
        "expansion_policy": dict(spec.expansion_policy),
        "base_profile": profile,
    }

    stage_overrides: Dict[str, object] = {
        "brain_regions": region_block,
        "scope": {
            **base_config.get("scope", {}),
            "total_neurons": max(neuron_estimate, 1_000),
        },
        "curriculum": curriculum_payload,
    }
    if spec.runtime_overrides:
        stage_overrides["runtime"] = deepcopy(spec.runtime_overrides)

    merged = _deep_merge(base_config, stage_overrides)
    if spec.config_overrides:
        merged = _deep_merge(merged, spec.config_overrides)
    metadata = merged.setdefault("metadata", {})
    metadata["stage"] = spec.key
    metadata["stage_label"] = spec.label
    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged


def _build_brain_region_block(
    spec: StageSpec,
) -> Tuple[Dict[str, Dict[str, float]], int]:
    """Create region entries and return (region_config, total_neurons)."""

    region_config: Dict[str, Dict[str, float]] = {}
    total_neurons = 0

    for region, template in _CANONICAL_REGIONS.items():
        scaling = spec.scaling_for(region)
        scaled = _scale_region(template, scaling)
        region_config[region.value] = scaled
        total_neurons += int(scaled["volume"] * scaled["neuron_density"])

    return region_config, total_neurons


def _scale_region(
    template: Mapping[str, float],
    scaling: RegionScaling,
) -> Dict[str, float]:
    """Apply ``scaling`` to a canonical template."""

    base = dict(_GENERIC_REGION)
    base.update(template)
    scaled = {
        "volume": base["volume"] * scaling.volume,
        "surface_area": base["surface_area"] * scaling.surface_area,
        "thickness": base["thickness"] * scaling.thickness,
        "neuron_density": base["neuron_density"] * scaling.neuron_density,
    }
    return scaled


def _deep_merge(
    base: MutableMapping[str, object],
    overrides: Mapping[str, object],
) -> MutableMapping[str, object]:
    """Recursively merge ``overrides`` into ``base`` (modifies and returns base)."""

    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_merge(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = deepcopy(value)
    return base


__all__ = [
    "RegionScaling",
    "StageRuntimePolicy",
    "StageSpec",
    "build_stage_config",
    "get_stage_spec",
    "list_stages",
]
