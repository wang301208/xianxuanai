"""Services that construct cognitive controller components for the Brain API."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

from BrainSimulationSystem.config.cognitive_defaults import (
    DEFAULT_ATTENTION_PARAMS,
    DEFAULT_WORKING_MEMORY_PARAMS,
    load_cognitive_defaults,
)

try:  # pragma: no cover - exercised in environments with the full model package
    from BrainSimulationSystem.models.cognitive_controller import CognitiveControllerBuilder
except Exception:  # pragma: no cover - fallback implementation for test environments

    def _merge_with_defaults(
        component: str,
        fallback: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        defaults = load_cognitive_defaults().get(component, {})
        merged: Dict[str, Any] = dict(fallback)
        if isinstance(defaults, dict):
            merged.update(defaults)
        if overrides:
            merged.update(overrides)
        return merged

    class _StubAttention:
        def __init__(self, params: Optional[Dict[str, Any]] = None):
            merged = _merge_with_defaults("attention", DEFAULT_ATTENTION_PARAMS, params)
            self.focus = []
            self.salience_map: Dict[str, Any] = {}
            self.ach_sensitivity = float(merged.get("ach_sensitivity", 1.0))
            self.bottom_up_weight = float(merged.get("bottom_up_weight", 0.5))
            self.top_down_weight = float(merged.get("top_down_weight", 0.5))
            self.attention_span = int(merged.get("attention_span", 3))

    class _StubWorkingMemory:
        def __init__(self, params: Optional[Dict[str, Any]] = None):
            merged = _merge_with_defaults("working_memory", DEFAULT_WORKING_MEMORY_PARAMS, params)
            self.memory_items: Dict[str, Any] = {}
            self.item_strengths: Dict[str, float] = {}
            self.capacity = int(merged.get("capacity", 7))
            self.decay_rate = float(merged.get("decay_rate", 0.05))
            self.ach_sensitivity = float(merged.get("ach_sensitivity", 1.0))
            self.attention_boost = float(merged.get("attention_boost", 0.0))
            self.encoding_factor = float(merged.get("encoding_factor", 1.0))
            self.retrieval_threshold = float(merged.get("retrieval_threshold", 0.0))
            self.interference_factor = float(merged.get("interference_factor", 1.0))

        def get_all_items(self):
            return self.memory_items

        def get_item(self, key):
            return self.memory_items.get(key)

        def _store_item(self, key, value, priority):
            self.memory_items[key] = value
            self.item_strengths[key] = priority

        def _delete_item(self, key):
            self.memory_items.pop(key, None)
            self.item_strengths.pop(key, None)

    class _StubController:
        def __init__(self, params: Optional[Dict[str, Dict[str, Any]]] = None):
            params = params or {}
            attention_params = params.get("attention")
            memory_params = params.get("working_memory")
            self.components = {
                "attention": _StubAttention(attention_params),
                "working_memory": _StubWorkingMemory(memory_params),
            }
            self.state = SimpleNamespace(name="IDLE")
            self.state_history = []
            self.neuromodulators = {
                "dopamine": 0.5,
                "serotonin": 0.5,
                "acetylcholine": 0.5,
                "norepinephrine": 0.5,
            }

        def process(self, data):
            self.state = SimpleNamespace(name="PROCESSING")
            self.state_history.append(self.state)
            return {"status": "processed", "input": data}

        def set_neuromodulator(self, name, level):
            self.neuromodulators[name] = level

    class CognitiveControllerBuilder:  # type: ignore
        def __init__(self):
            self._params = load_cognitive_defaults()

        def with_attention_params(self, params):
            if params:
                self._params.setdefault("attention", {}).update(params)
            return self

        def with_working_memory_params(self, params):
            if params:
                self._params.setdefault("working_memory", {}).update(params)
            return self

        def with_self_model_params(self, params):  # pragma: no cover - API compatibility
            return self

        def build(self):
            return _StubController(self._params)


class CognitiveControllerFactory:
    """Factory responsible for constructing cognitive controllers."""

    def __init__(self, builder_cls: Optional[type] = None) -> None:
        self._builder_cls = builder_cls or CognitiveControllerBuilder

    def create(
        self,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Any:
        """Create a cognitive controller with defaults merged with overrides."""
        overrides = overrides or {}
        defaults = load_cognitive_defaults()

        attention_params = dict(DEFAULT_ATTENTION_PARAMS)
        attention_params.update(defaults.get("attention", {}))
        attention_params.update(overrides.get("attention", {}))

        memory_params = dict(DEFAULT_WORKING_MEMORY_PARAMS)
        memory_params.update(defaults.get("working_memory", {}))
        memory_params.update(overrides.get("working_memory", {}))

        builder = self._builder_cls()
        builder.with_attention_params(attention_params)
        builder.with_working_memory_params(memory_params)
        builder.with_self_model_params(overrides.get("self_model"))

        return builder.build()


__all__ = ["CognitiveControllerFactory"]
