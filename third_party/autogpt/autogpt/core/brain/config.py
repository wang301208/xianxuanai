from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict

from pydantic import Field

from autogpt.core.configuration import SystemConfiguration, UserConfigurable

from modules.brain.state import BrainRuntimeConfig


class BrainBackend(str, Enum):
    """Selects which cognitive backend drives ``BaseAgent.propose_action``."""

    LLM = "llm"
    TRANSFORMER = "transformer"
    WHOLE_BRAIN = "whole_brain"
    BRAIN_SIMULATION = "brain_simulation"


class TransformerBrainConfig(SystemConfiguration):
    """Configuration options for the internal transformer brain."""

    layers: int = UserConfigurable(default=2)
    """Number of transformer encoder layers."""

    heads: int = UserConfigurable(default=4)
    """Number of attention heads in each transformer layer."""

    dim: int = UserConfigurable(default=256)
    """Dimensionality of the model."""

    dropout: float = UserConfigurable(default=0.1)
    """Dropout probability used in transformer layers."""

    weights_path: str | None = UserConfigurable(default=None)
    """Optional path to load pretrained weights at initialization."""

    dataset_logging_path: str | None = UserConfigurable(default=None)
    """If set, write per-cycle brain samples to this JSONL file."""

    # Training configuration
    learning_rate: float = UserConfigurable(default=1e-3)
    """Learning rate used during training."""

    epochs: int = UserConfigurable(default=10)
    """Number of training epochs."""

    batch_size: int = UserConfigurable(default=4)
    """Batch size for the training dataloader."""

    # Online / continual learning
    online_learning_enabled: bool = UserConfigurable(default=False)
    """Enable continual updates from in-run experiences."""

    online_buffer_size: int = UserConfigurable(default=2048)
    """Maximum number of recent experiences retained for replay."""

    online_min_batch: int = UserConfigurable(default=32)
    """Minimum samples required before triggering an online update."""

    online_update_interval: int = UserConfigurable(default=5)
    """Perform an online update every N completed interactions."""

    online_batch_size: int = UserConfigurable(default=32)
    """Number of experiences sampled per online update."""

    online_learning_rate: float = UserConfigurable(default=5e-4)
    """Learning rate used for the online optimiser (defaults to ``learning_rate`` when unset)."""

    online_ewc_lambda: float = UserConfigurable(default=0.05)
    """Elastic Weight Consolidation regularisation strength (lambda)."""

    online_ewc_decay: float = UserConfigurable(default=0.9)
    """Decay applied to the running Fisher/importance estimate after each update."""

    online_gradient_clip: float = UserConfigurable(default=5.0)
    """Max norm for gradients during online updates (<=0 disables clipping)."""

    enable_chain_of_thought: bool = UserConfigurable(default=True)
    """Whether to record intermediate chain-of-thought states."""

    chain_of_thought_steps: int = UserConfigurable(default=4)
    """Maximum number of chain-of-thought steps to capture."""

    enable_react: bool = UserConfigurable(default=True)
    """Enable ReAct-style reasoning/tool loops during action proposal."""

    react_iterations: int = UserConfigurable(default=2)
    """Number of reasoning/action alternations when ReAct is enabled."""

    enable_task_decomposition: bool = UserConfigurable(default=True)
    """Whether to derive a hierarchical task plan from the current goal."""

    task_decomposition_depth: int = UserConfigurable(default=3)
    """Maximum depth for recursive task decomposition."""

    task_decomposition_branch_factor: int = UserConfigurable(default=4)
    """Maximum number of subtasks per decomposition level."""

    enable_hierarchical_memory: bool = UserConfigurable(default=True)
    """Enable the multi-tier memory manager."""

    working_memory_capacity: int = UserConfigurable(default=7)
    """Capacity of the working/short-term memory buffer."""

    episodic_memory_limit: int = UserConfigurable(default=256)
    """Maximum number of episodic traces retained before consolidation."""

    consolidation_importance_threshold: float = UserConfigurable(default=0.7)
    """Importance score required to immediately consolidate an episode."""

    consolidation_time_window: float = UserConfigurable(default=600.0)
    """Seconds after which episodic traces become eligible for consolidation."""

    consolidation_batch_size: int = UserConfigurable(default=5)
    """Number of hippocampal traces processed per consolidation pass."""

    long_term_memory_path: str | None = UserConfigurable(default=":memory:")
    """Persistent storage path for long-term memories (``:memory:`` keeps it in RAM)."""

    long_term_memory_max_entries: int | None = UserConfigurable(default=5000)
    """Optional cap on stored long-term memories."""

    memory_decay_half_life: float = UserConfigurable(default=86400.0)
    """Half-life in seconds used for decay/forgetting of episodic traces."""

    memory_interference_penalty: float = UserConfigurable(default=0.1)
    """Interference factor applied when buffers grow beyond their limits."""

    semantic_memory_limit: int = UserConfigurable(default=2048)
    """Maximum number of semantic facts retained in fast access storage."""


def _parse_json_mapping(value: str | None) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return dict(parsed)
    return {}


class BrainServingConfig(SystemConfiguration):
    """Configuration for delegating brain inference to an external serving stack."""

    enabled: bool = UserConfigurable(default=False, from_env="BRAIN_SERVING_ENABLED")
    protocol: str = UserConfigurable(default="http", from_env="BRAIN_SERVING_PROTOCOL")
    endpoint: str | None = UserConfigurable(default=None, from_env="BRAIN_SERVING_ENDPOINT")
    path: str = UserConfigurable(default="/infer", from_env="BRAIN_SERVING_PATH")
    method: str = UserConfigurable(default="POST", from_env="BRAIN_SERVING_METHOD")
    model_name: str = UserConfigurable(default="transformer-brain", from_env="BRAIN_SERVING_MODEL")
    timeout: float = UserConfigurable(default=15.0, from_env="BRAIN_SERVING_TIMEOUT")
    prefer_batch: bool = UserConfigurable(default=False, from_env="BRAIN_SERVING_PREFER_BATCH")
    max_batch_size: int = UserConfigurable(default=8, from_env="BRAIN_SERVING_MAX_BATCH")
    max_batch_latency_ms: int = UserConfigurable(default=20, from_env="BRAIN_SERVING_BATCH_TIMEOUT_MS")
    async_mode: bool = UserConfigurable(default=False, from_env="BRAIN_SERVING_ASYNC")
    fallback_to_local: bool = UserConfigurable(default=True, from_env="BRAIN_SERVING_FALLBACK_LOCAL")
    metrics_topic: str | None = UserConfigurable(
        default="brain_metrics", from_env="BRAIN_SERVING_METRICS_TOPIC"
    )
    result_topic: str | None = UserConfigurable(
        default=None, from_env="BRAIN_SERVING_RESULT_TOPIC"
    )
    headers_json: str | None = UserConfigurable(
        default=None, from_env="BRAIN_SERVING_HEADERS_JSON"
    )
    query_json: str | None = UserConfigurable(
        default=None, from_env="BRAIN_SERVING_QUERY_JSON"
    )
    options_json: str | None = UserConfigurable(
        default=None, from_env="BRAIN_SERVING_OPTIONS_JSON"
    )

    def parsed_headers(self) -> Dict[str, str]:
        return {str(k): str(v) for k, v in _parse_json_mapping(self.headers_json).items()}

    def parsed_query(self) -> Dict[str, Any]:
        return _parse_json_mapping(self.query_json)

    def parsed_options(self) -> Dict[str, Any]:
        return _parse_json_mapping(self.options_json)

    def rpc_defaults(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {
            "protocol": self.protocol,
            "timeout": self.timeout,
            "method": self.method,
        }
        if self.endpoint:
            defaults["endpoint"] = self.endpoint
        if self.path:
            defaults["path"] = self.path
        headers = self.parsed_headers()
        if headers:
            defaults["headers"] = headers
        query = self.parsed_query()
        if query:
            defaults["query"] = query
        options = self.parsed_options()
        if options:
            defaults["options"] = options
        return defaults

    def rpc_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "protocol": self.protocol,
            "method": self.method,
        }
        if self.endpoint:
            config["endpoint"] = self.endpoint
        if self.path:
            config["path"] = self.path
        headers = self.parsed_headers()
        if headers:
            config["headers"] = headers
        query = self.parsed_query()
        if query:
            config["query"] = query
        options = self.parsed_options()
        if options:
            config["options"] = options
        return config

    def to_request_metadata(self, *, batch_size: int | None = None) -> Dict[str, Any]:
        serving_meta: Dict[str, Any] = {
            "prefer_batch": bool(self.prefer_batch),
            "max_batch_size": int(self.max_batch_size),
            "max_batch_latency_ms": int(self.max_batch_latency_ms),
            "async_mode": bool(self.async_mode),
        }
        if batch_size is not None:
            serving_meta["batch_size"] = int(batch_size)
        return {"serving": serving_meta}


class WholeBrainRuntimeSettings(SystemConfiguration):
    """Toggles mirrored onto :class:`modules.brain.state.BrainRuntimeConfig`."""

    use_neuromorphic: bool = UserConfigurable(default=True, from_env="WHOLE_BRAIN_USE_NEUROMORPHIC")
    enable_multi_dim_emotion: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_ENABLE_MULTI_DIM_EMOTION"
    )
    enable_emotion_decay: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_ENABLE_EMOTION_DECAY"
    )
    enable_curiosity_feedback: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_ENABLE_CURIOSITY_FEEDBACK"
    )
    enable_self_learning: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_ENABLE_SELF_LEARNING"
    )
    enable_continual_learning: bool = UserConfigurable(
        default=False, from_env="WHOLE_BRAIN_ENABLE_CONTINUAL_LEARNING"
    )
    enable_self_evolution: bool = UserConfigurable(
        default=False, from_env="WHOLE_BRAIN_ENABLE_SELF_EVOLUTION"
    )
    enable_meta_cognition: bool = UserConfigurable(
        default=False, from_env="WHOLE_BRAIN_ENABLE_META_COGNITION"
    )
    enable_personality_modulation: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_ENABLE_PERSONALITY_MODULATION"
    )
    enable_plan_logging: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_ENABLE_PLAN_LOGGING"
    )
    metrics_enabled: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_METRICS_ENABLED"
    )
    continual_learning_experience_root: str = UserConfigurable(
        default="data/experience", from_env="WHOLE_BRAIN_CONTINUAL_EXPERIENCE_ROOT"
    )
    continual_learning_background_interval: float = UserConfigurable(
        default=60.0, from_env="WHOLE_BRAIN_CONTINUAL_BACKGROUND_INTERVAL"
    )
    continual_learning_policy_path: str = UserConfigurable(
        default="data/experience/policies/intention_bandit.json",
        from_env="WHOLE_BRAIN_CONTINUAL_POLICY_PATH",
    )
    evolution_ga_population: int = UserConfigurable(
        default=20, from_env="WHOLE_BRAIN_EVOLUTION_POPULATION"
    )
    evolution_ga_generations: int = UserConfigurable(
        default=5, from_env="WHOLE_BRAIN_EVOLUTION_GENERATIONS"
    )
    evolution_ga_mutation_rate: float = UserConfigurable(
        default=0.3, from_env="WHOLE_BRAIN_EVOLUTION_MUTATION_RATE"
    )
    evolution_ga_mutation_sigma: float = UserConfigurable(
        default=0.1, from_env="WHOLE_BRAIN_EVOLUTION_MUTATION_SIGMA"
    )
    meta_failure_threshold: int = UserConfigurable(
        default=3, from_env="WHOLE_BRAIN_META_FAILURE_THRESHOLD"
    )
    meta_low_confidence_threshold: float = UserConfigurable(
        default=0.35, from_env="WHOLE_BRAIN_META_LOW_CONFIDENCE_THRESHOLD"
    )
    meta_low_confidence_window: int = UserConfigurable(
        default=5, from_env="WHOLE_BRAIN_META_LOW_CONFIDENCE_WINDOW"
    )
    meta_knowledge_gap_threshold: int = UserConfigurable(
        default=2, from_env="WHOLE_BRAIN_META_KNOWLEDGE_GAP_THRESHOLD"
    )

    def to_runtime(self) -> BrainRuntimeConfig:
        """Create a :class:`BrainRuntimeConfig` dataclass reflecting the settings."""

        return BrainRuntimeConfig(
            use_neuromorphic=self.use_neuromorphic,
            enable_multi_dim_emotion=self.enable_multi_dim_emotion,
            enable_emotion_decay=self.enable_emotion_decay,
            enable_curiosity_feedback=self.enable_curiosity_feedback,
            enable_self_learning=self.enable_self_learning,
            enable_continual_learning=self.enable_continual_learning,
            enable_self_evolution=self.enable_self_evolution,
            enable_meta_cognition=self.enable_meta_cognition,
            enable_personality_modulation=self.enable_personality_modulation,
            enable_plan_logging=self.enable_plan_logging,
            metrics_enabled=self.metrics_enabled,
            continual_learning_experience_root=self.continual_learning_experience_root,
            continual_learning_background_interval=float(self.continual_learning_background_interval),
            continual_learning_policy_path=self.continual_learning_policy_path,
            evolution_ga_population=int(self.evolution_ga_population),
            evolution_ga_generations=int(self.evolution_ga_generations),
            evolution_ga_mutation_rate=float(self.evolution_ga_mutation_rate),
            evolution_ga_mutation_sigma=float(self.evolution_ga_mutation_sigma),
            meta_failure_threshold=int(self.meta_failure_threshold),
            meta_low_confidence_threshold=float(self.meta_low_confidence_threshold),
            meta_low_confidence_window=int(self.meta_low_confidence_window),
            meta_knowledge_gap_threshold=int(self.meta_knowledge_gap_threshold),
        )


class WholeBrainConfig(SystemConfiguration):
    """Configuration wrapper for :class:`modules.brain.whole_brain.WholeBrainSimulation`."""

    neuromorphic_encoding: str = UserConfigurable(
        default="rate", from_env="WHOLE_BRAIN_ENCODING"
    )
    encoding_steps: int = UserConfigurable(
        default=5, from_env="WHOLE_BRAIN_ENCODING_STEPS"
    )
    encoding_time_scale: float = UserConfigurable(
        default=1.0, from_env="WHOLE_BRAIN_ENCODING_TIME_SCALE"
    )
    max_neurons: int = UserConfigurable(default=128, from_env="WHOLE_BRAIN_MAX_NEURONS")
    max_cache_size: int = UserConfigurable(
        default=8, from_env="WHOLE_BRAIN_MAX_CACHE_SIZE"
    )
    runtime: WholeBrainRuntimeSettings = Field(default_factory=WholeBrainRuntimeSettings)
    serving: BrainServingConfig = Field(default_factory=BrainServingConfig)

    def to_simulation_kwargs(self) -> dict:
        """Return keyword arguments used to initialize the simulation."""

        runtime = self.runtime.to_runtime()
        return {
            "config": runtime,
            "neuromorphic": runtime.use_neuromorphic,
            "neuromorphic_encoding": self.neuromorphic_encoding,
            "encoding_steps": self.encoding_steps,
            "encoding_time_scale": self.encoding_time_scale,
            "max_neurons": self.max_neurons,
            "max_cache_size": self.max_cache_size,
        }


class BrainSimulationConfig(SystemConfiguration):
    """Configuration payload for :mod:`BrainSimulationSystem` integration."""

    profile: str = UserConfigurable(
        default="production", from_env="BRAIN_SIM_PROFILE"
    )
    """Base profile name defined inside ``BrainSimulationSystem``."""

    stage: str | None = UserConfigurable(
        default=None, from_env="BRAIN_SIM_STAGE"
    )
    """Optional curriculum stage identifier."""

    config_file: str | None = UserConfigurable(
        default=None, from_env="BRAIN_SIM_CONFIG_FILE"
    )
    """Path to a JSON document merged into the final configuration."""

    overrides_json: str | None = UserConfigurable(
        default=None, from_env="BRAIN_SIM_OVERRIDES_JSON"
    )
    """Inline JSON string merged into the final configuration."""

    overrides: Dict[str, Any] = Field(default_factory=dict)
    """Programmatic overrides injected via API."""

    timestep_ms: float = UserConfigurable(
        default=100.0, from_env="BRAIN_SIM_TIMESTEP_MS"
    )
    """Simulation timestep passed to :meth:`BrainSimulation.step`."""

    auto_background: bool = UserConfigurable(
        default=False, from_env="BRAIN_SIM_AUTO_BACKGROUND"
    )
    """Whether to start the continuous simulation loop."""

    def resolved_overrides(self) -> Dict[str, Any]:
        """Combine overrides coming from dicts, JSON strings, or files."""

        merged: Dict[str, Any] = dict(self.overrides or {})
        inline = _parse_json_mapping(self.overrides_json)
        if inline:
            merged = _merge_dicts(merged, inline)

        if self.config_file:
            path = Path(self.config_file)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                payload = {}
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                merged = _merge_dicts(merged, payload)
        return merged

    def to_backend_payload(self) -> Dict[str, Any]:
        """Return kwargs for :func:`modules.brain.backends.create_brain_backend`."""

        return {
            "profile": self.profile,
            "stage": self.stage,
            "overrides": self.resolved_overrides(),
            "dt": max(float(self.timestep_ms), 1e-3),
            "auto_background": bool(self.auto_background),
        }


def _merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


__all__ = [
    "BrainBackend",
    "TransformerBrainConfig",
    "WholeBrainConfig",
    "WholeBrainRuntimeSettings",
    "BrainServingConfig",
    "BrainSimulationConfig",
]
