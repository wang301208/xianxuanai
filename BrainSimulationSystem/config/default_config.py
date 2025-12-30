# -*- coding: utf-8 -*-
"""
默认配置（小规模可运行），包含 ``network.py`` 期望的关键字段，并新增可扩展的规模档案：

- ``prototype``：默认测试档案，维持轻量设置。
- ``ci_scalable``：CI 使用的缩放档案，提供较大的列/神经元数但保持资源可控。
- ``research``：研究用中等规模示例。
- ``full_brain``：与 ``BrainSimulationConfig`` 默认值对齐的整脑规模参数。
- ``production``：启用全套硬件/运行时/生理特性用于生产级就绪度验证。

配置同时暴露资源管理选项（分区、后端偏好），方便未来 ``SimulationBackend`` 集成。
"""

from copy import deepcopy
from typing import Dict, Any


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` returning a new dictionary."""

    merged = deepcopy(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _base_config() -> Dict[str, Any]:
    """Return the baseline configuration shared by all profiles."""

    scope = {
        "brain_regions": [
            "PRIMARY_VISUAL_CORTEX",
            "PRIMARY_AUDITORY_CORTEX",
            "PRIMARY_MOTOR_CORTEX",
            "PRIMARY_SOMATOSENSORY_CORTEX",
            "PREFRONTAL_CORTEX",
            "HIPPOCAMPUS_CA1",
            "HIPPOCAMPUS_CA3",
            "DENTATE_GYRUS",
            "THALAMUS_LGN",
            "THALAMUS_MD",
            "THALAMUS_VPL",
            "THALAMUS_VPM",
            "AMYGDALA",
            "NUCLEUS_ACCUMBENS",
            "STRIATUM",
        ],
        "total_neurons": 20_000,
        "columns_per_region": 4,
    }

    cortical_structure = {
        "layers": [
            {"name": "L1", "thickness": 150.0, "cell_density": 30_000},
            {"name": "L2", "thickness": 300.0, "cell_density": 90_000},
            {"name": "L3", "thickness": 400.0, "cell_density": 110_000},
            {"name": "L4", "thickness": 400.0, "cell_density": 120_000},
            {"name": "L5", "thickness": 700.0, "cell_density": 100_000},
            {"name": "L6", "thickness": 800.0, "cell_density": 90_000},
        ]
    }

    connectivity_patterns = {
        "local_connectivity": {
            "intra_columnar": 0.02,
            "layer_specific": {
                "L2_to_L5": 0.01,
                "L3_to_L5": 0.01,
                "L4_to_L2": 0.015,
                "L5_to_L2": 0.008,
                "L6_to_L4": 0.006,
            },
        },
        "long_range_connectivity": {
            "cortico_cortical": 0.2,
        },
    }

    neuromorphic = {
        "hardware_platforms": {
            "intel_loihi": {"enabled": False},
            "spinnaker": {"enabled": False},
        },
        "backend_manager": {
            "enabled": False,
            "requirements": {
                "real_time": False,
                "power_limit": 1000.0,
            },
        },
        "device_discovery": {"enabled": False},
        "bridge": {"enabled": False},
        "export": {
            "aer": {
                "enabled": False,
                "path": "BrainSimulationSystem/monitoring/aer_events.csv",
            },
            "mapping": {
                "enabled": False,
                "json_path": "BrainSimulationSystem/monitoring/backend_mapping.json",
                "csv_path": "BrainSimulationSystem/monitoring/backend_mapping.csv",
                "validate_ranges": True,
            },
        },
    }

    physiology = {
        "thalamocortical": {
            "enabled": True,
            "nuclei": {
                "VPL": {"size": 150, "enabled": True},
                "VPM": {"size": 120, "enabled": True},
                "LGN": {"size": 200, "enabled": True},
                "MGN": {"size": 100, "enabled": True},
                "MD": {"size": 180, "enabled": True},
                "PULVINAR": {"size": 160, "enabled": True},
                "RETICULAR": {"size": 300, "enabled": True},
            },
            "oscillation_params": {
                "alpha_frequency": 10.0,
                "spindle_frequency": 12.0,
                "delta_frequency": 2.0,
                "gamma_frequency": 40.0,
            },
            "plasticity_enabled": True,
            "attention_modulation": True,
        },
        "enhanced_cortical_column": {
            "enabled": True,
            "layer_specific_oscillations": True,
            "inter_layer_plasticity": True,
            "thalamic_integration": True,
        },
        "hippocampus_pfc": {
            "enabled": False,
            "pathway": ["DG", "CA3", "CA1", "PFC"],
            "projection_size": 100,
        },
        "glia_vascular": {
            "enabled": False,
            "modulation_strength": 0.05,
        },
    }

    runtime = {
        "timestep_ms": 0.1,
        "parallel": {
            "enabled": True,
            "mode": "auto",
            "max_workers": None,
            "distributed": {"enabled": False},
        },
        "gpu": {
            "enabled": False,
            "use_cuda": True,
            "device": None,
            "precision": "fp32",
        },
        "performance": {
            "window": 60,
            "max_region_time": 0.03,
            "min_region_time": 0.004,
            "max_global_time": 0.12,
            "evaluation_interval": 10,
        },
        "parallelism": {"processes": 2, "threads": 2},
        "partition": {
            "strategy": "round_robin",
            "num_partitions": 2,
        },
        "resource_management": {
            "backend_selection": {
                "preferred": "native",
                "allowed": ["native"],
                "fallback": "native",
            },
            "partitioning": {
                "max_neurons_per_partition": 10_000,
                "max_columns_per_partition": 2,
                "autoscale": False,
            },
            "memory_budget_gb": 2,
        },
        "checkpoint": {
            "enabled": False,
            "path": "BrainSimulationSystem/checkpoints/default.ckpt",
            "export_plan_path": "BrainSimulationSystem/checkpoints/plan.json",
        },
        "distributed": {
            "enabled": False,
            "export_plan_path": "BrainSimulationSystem/monitoring/distributed_plan.json",
        },
        # CognitiveArchitecture runtime toggles (hierarchical hybrid modeling).
        "region_update": {
            "mode": "event_driven",
            "parallel": {"enabled": True, "strategy": "auto", "max_workers": None},
            "event_driven": {
                "input_epsilon": 1e-3,
                "activation_threshold": 0.05,
                "max_pending_events": 200_000,
                "max_events_per_cycle": 50_000,
                "always_update": [],
            },
        },
        "functional_topology": {
            "enabled": True,
            "gain": 1.0,
            "key_prefix": "module_",
            "activity_key": "module_activity",
            "input_key_prefix": "functional_",
            "include_sensory": False,
        },
        "parameter_mapping": {
            "enabled": True,
            "working_memory": {"neurons_per_item": 12, "min_neurons": 40, "max_neurons": 400},
            "basal_ganglia": {
                "enable_stdp": True,
                "dopamine_gain_scale": 5.0,
                "loihi_dopamine_input_gain": 0.5,
            },
            "visual": {"use_retina_lgn_v1": False},
            "calibration": {
                "enabled": False,
                "source": "brain_atlas_default",
                "target_mean_neurons_per_region": 120,
                "min_neurons": 20,
                "max_neurons": 800,
                "override_existing": False,
            },
        },
        "module_bus": {
            "enabled": True,
            "manage_cycle": True,
            "export_in_results": True,
            "publish_cognitive_state": True,
            "publish_microcircuit_events": True,
            "thresholds": {
                "v1_salience_rate_hz": 20.0,
                "ca1_retrieval_rate_hz": 12.0,
                "high_activity_rate_hz": 25.0,
            },
        },
    }

    simulation = {
        "backend": "native",
        "dt": 1.0,
        "backend_options": {
            "preferred": "native",
            "allowed": ["native", "biophysical", "distributed", "neuromorphic"],
            "fallback": "native",
        },
        # Configuration for the optional ``biophysical`` backend.
        # Enable by setting ``simulation.backend = 'biophysical'``.
        "biophysical": {
            "seed": 42,
            "neurons_per_region": 120,
            "excitatory_ratio": 0.8,
            "neuron_model": "izhikevich",
        },
        "save_interval": 0,
    }

    monitoring = {
        "enabled": False,
        "metrics": ["update_time", "spike_rate", "power"],
        "export_path": "BrainSimulationSystem/monitoring/metrics.json",
        "performance": {"enabled": False},
    }

    visualization = {
        "enabled": False,
        "export_path": "BrainSimulationSystem/visualization/metrics.csv",
    }

    optional_modules = {
        "anatomy": {"enabled": False},
        "cognition_binding": {"enabled": False},
    }

    cognition = {
        "enabled": False,
        "tasks": ["working_memory", "pattern_completion", "sequence_learning"],
    }

    perception = {
        "input_mapping": "direct",
        "normalization": "minmax",
        "noise_level": 0.0,
        "vision": {
            "enabled": False,
            "return_feature_maps": False,
            "model": {
                "backend": "auto",
                "feature_dim": 128,
                "input_size": [224, 224],
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
        },
        "auditory": {
            "enabled": False,
            "model": {
                "backend": "auto",
                "sample_rate": 16000,
                "frame_length": 0.025,
                "frame_step": 0.010,
                "n_mels": 40,
                "feature_dim": 128,
            },
        },
        "somatosensory": {
            "enabled": False,
            "model": {
                "normalize": True,
                "feature_dim": 64,
                "history_decay": 0.8,
            },
        },
    }

    memory = {
        "enabled": True,
        "persistence": {
            "path": "BrainSimulationSystem/data/persistent_memory.json",
            "embedding_dim": 128,
            "max_entries": 10000,
            "working_memory_size": 32
        },
        "vector_store": {
            "enabled": False,
            "backend": "chroma",
            "path": "BrainSimulationSystem/data/vector_memory",
            "collection": "cognitive_memory",
            "top_k": 8
        }
    }

    attention_manager = {
        "enabled": False,
        "controller": {
            "focus_count": 5,
            "focus_capacity": 5,
            "salience_weight": 0.5,
            "confidence_weight": 0.3,
            "novelty_weight": 0.2,
            "motivation_weight": 0.3,
            "scoring": "heuristic",
            "transformer_hidden_dim": 32,
            "transformer_weights_path": None,
            "history_size": 32,
            "freeze_sources": [],
            "suppressed_decay": 0.85,
            "rl_model_path": None,
        },
    }

    # 决策模块配置，rl 节点暴露训练脚本的核心调节参数
    decision = {
        "decision_type": "softmax",
        "temperature": 1.0,
        "threshold": 0.5,
        "learning_rate": 0.1,
        "rl": {
            "enabled": False,
            "algorithm": "ppo",
            "policy": "MlpPolicy",
            "model_path": "BrainSimulationSystem/models/rl/decision_policy.zip",
            "train_timesteps": 5000,
            "training_episodes": 64,
            "update_interval": 2048,
            "max_options": 5,
            "device": "auto",
            "verbose": 0,
        },
    }

    self_model = {
        "enabled": False,
        "module": {
            "window_size": 64,
            "report_history": 8,
            "introspection_interval": 1000.0,
            "min_confidence": 0.35,
            "anomaly_threshold": 0.6,
            "enable_text_report": True,
            "track_subsystems": ["decision", "memory", "attention", "emotion"],
        },
    }

    hebbian = {
        "enabled": True,
        "potentiation": 0.005,
        "decay": 0.001,
        "max_weight": 5.0,
        "min_weight": -5.0,
    }

    knowledge = {
        "triples": [
            ["agent", "supports", "exploration"],
        ],
        "rules": [
            {
                "name": "explore_requires_goal",
                "antecedents": [["goal", "type", "exploration"]],
                "consequent": ["agent", "supports", "exploration"],
            }
        ],
        "constraints": [],
        "sources": [],
    }

    planner = {
        "enabled": False,
        "controller": {
            "max_candidates": 5,
            "heuristic_weight": 0.6,
            "rl_model_path": "BrainSimulationSystem/models/rl/planner_policy.zip",
        },
    }

    self_supervised = {
        "enabled": True,
        "predictor": {
            "max_observation_dim": 192,
            "latent_dim": 48,
            "action_embedding_dim": 24,
            "learning_rate": 0.005,
            "prediction_learning_rate": 0.003,
            "contrastive_margin": 0.2,
            "contrastive_weight": 0.15,
            "history_size": 256,
            "auto_store_enabled": False,
            "auto_store_error_threshold": 0.05,
        },
    }

    meta_learning = {
        "enabled": False,
        "trainer": {
            "meta_iterations": 30,
            "meta_batch_size": 4,
            "inner_steps": 5,
            "inner_learning_rate": 1e-2,
            "meta_learning_rate": 5e-3,
            "algorithm": "maml",
            "task_adaptation_steps": 5,
            "report_interval": 1,
        },
        "ga": {
            "enabled": False,
            "generations": 8,
            "search_space": {
                "inner_learning_rate": {"min": 1e-3, "max": 0.2},
                "meta_learning_rate": {"min": 1e-4, "max": 0.05},
                "inner_steps": {"min": 1, "max": 10, "type": "int"},
            },
        },
    }

    meta_reasoning = {
        "enabled": False,
        "controller": {
            "enable_counterfactual": True,
            "enable_consistency": True,
            "enable_self_reflection": True,
        },
    }

    biology = {
        "enabled": False,
        "cell_distribution": {
            "pyramidal_ratio": {"L2": 0.8, "L3": 0.8, "L5": 0.7, "L6": 0.6},
            "interneuron_types": [
                "pv_interneuron",
                "sst_interneuron",
                "vip_interneuron",
            ],
            "interneuron_weights": [0.5, 0.3, 0.2],
        },
    }

    return {
        "metadata": {
            "profile": "prototype",
            "description": "Default lightweight configuration for unit tests.",
        },
        "scope": scope,
        "cortical_structure": cortical_structure,
        "connectivity_patterns": connectivity_patterns,
        "neuromorphic": neuromorphic,
        "physiology": physiology,
        "runtime": runtime,
        "simulation": simulation,
        "monitoring": monitoring,
        "visualization": visualization,
        "optional_modules": optional_modules,
        "cognition": cognition,
        "perception": perception,
        "attention_manager": attention_manager,
        "decision": decision,
        "self_model": self_model,
        "hebbian": hebbian,
        "memory": memory,
        "biology": biology,
        "knowledge": knowledge,
        "planner": planner,
        "self_supervised": self_supervised,
        "meta_learning": meta_learning,
        "meta_reasoning": meta_reasoning,
    }


_PROFILE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "prototype": {
        "metadata": {
            "description": "Default lightweight configuration for smoke tests",
        },
    },
    "ci_scalable": {
        "metadata": {
            "description": "CI-friendly scalable profile with modest neuron count.",
        },
        "scope": {
            "total_neurons": 150_000,
            "columns_per_region": 24,
        },
        "runtime": {
            "parallelism": {"processes": 4, "threads": 4},
            "partition": {
                "strategy": "balanced",
                "num_partitions": 8,
            },
            "resource_management": {
                "backend_selection": {
                    "preferred": "native",
                    "allowed": ["native", "distributed"],
                    "fallback": "native",
                },
                "partitioning": {
                    "max_neurons_per_partition": 25_000,
                    "max_columns_per_partition": 4,
                    "autoscale": True,
                },
                "memory_budget_gb": 4,
            },
        },
        "simulation": {
            "backend": "native",
        },
    },
    "research": {
        "metadata": {
            "description": "Intermediate research scale prototype.",
        },
        "scope": {
            "total_neurons": 5_000_000,
            "columns_per_region": 1_200,
        },
        "runtime": {
            "parallelism": {"processes": 32, "threads": 8},
            "partition": {
                "strategy": "hierarchical",
                "num_partitions": 128,
            },
            "resource_management": {
                "backend_selection": {
                    "preferred": "distributed",
                    "allowed": ["native", "distributed", "neuromorphic"],
                    "fallback": "native",
                },
                "partitioning": {
                    "max_neurons_per_partition": 200_000,
                    "max_columns_per_partition": 64,
                    "autoscale": True,
                },
                "memory_budget_gb": 128,
            },
        },
        "simulation": {
            "backend": "distributed",
        },
    },
    "full_brain": {
        "metadata": {
            "description": "Full brain scale aligned with BrainSimulationConfig defaults.",
        },
        "scope": {
            "total_neurons": 86_000_000_000,
            "columns_per_region": 120_000,
        },
        "runtime": {
            "parallelism": {"processes": 512, "threads": 16},
            "partition": {
                "strategy": "hierarchical",
                "num_partitions": 4_096,
            },
            "resource_management": {
                "backend_selection": {
                    "preferred": "neuromorphic",
                    "allowed": ["native", "distributed", "neuromorphic"],
                    "fallback": "distributed",
                },
                "partitioning": {
                    "max_neurons_per_partition": 50_000_000,
                    "max_columns_per_partition": 2_000,
                    "autoscale": True,
                },
                "memory_budget_gb": 65_536,
            },
        },
        "simulation": {
            "backend": "neuromorphic",
        },
    },
    "production": {
        "metadata": {
            "description": "Production readiness profile enabling all runtime contributions.",
        },
        "scope": {
            "total_neurons": 10_000_000_000,
        },
        "perception": {
            "vision": {
                "enabled": True,
                "return_feature_maps": True,
                "model": {
                    "backend": "auto",
                    "feature_dim": 256,
                    "input_size": [256, 256],
                },
            },
            "auditory": {
                "enabled": True,
                "model": {
                    "backend": "auto",
                    "sample_rate": 22050,
                    "feature_dim": 192,
                    "n_mels": 64,
                },
            },
            "somatosensory": {
                "enabled": True,
                "model": {
                    "normalize": True,
                    "feature_dim": 96,
                    "history_decay": 0.9,
                },
            },
        },
        "neuromorphic": {
            "bridge": {"enabled": True},
            "export": {
                "aer": {"enabled": True},
                "mapping": {"enabled": True},
            },
            "backend_manager": {
                "enabled": True,
                "requirements": {
                    "real_time": True,
                    "power_limit": 500.0,
                },
            },
        },
        "physiology": {
            "hippocampus_pfc": {
                "enabled": True,
                "projection_size": 250,
            },
            "glia_vascular": {
                "enabled": True,
                "modulation_strength": 0.05,
            },
        },
        "runtime": {
            "distributed": {
                "enabled": True,
            },
            "checkpoint": {
                "enabled": True,
                "path": "BrainSimulationSystem/checkpoints/production.ckpt",
            },
        },
        "simulation": {
            "backend": "distributed",
        },
        "monitoring": {
            "enabled": True,
            "performance": {"enabled": True},
        },
    },
}


def get_config(profile: str = "prototype") -> Dict[str, Any]:
    """Return a configuration dictionary for the requested scale profile."""

    key = profile.lower()
    base = _base_config()
    overrides = _PROFILE_OVERRIDES.get(key)
    if overrides is None:
        available = ", ".join(sorted(_PROFILE_OVERRIDES))
        raise ValueError(f"Unknown profile '{profile}'. Available profiles: {available}")

    cfg = _deep_merge(base, overrides)
    cfg.setdefault("metadata", {})["profile"] = key
    return cfg
