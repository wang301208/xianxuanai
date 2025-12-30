"""Core abstractions shared by all brain simulation network implementations."""

from __future__ import annotations

import logging
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional


@dataclass
class Layer:
    """Lightweight container representing a neuron population."""

    name: str
    neuron_type: str
    neurons: List[Any]
    neuron_ids: List[int]
    population_type: str = "excitatory"

    @property
    def size(self) -> int:
        return len(self.neurons)


class NeuralNetwork(ABC):
    """Minimal neural network interface used across the project."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config.copy() if config else {}
        self.layers: Dict[str, Any] = {}
        self.neurons: Dict[int, Any] = {}
        self.synapses: Dict[Any, Any] = {}
        self.pre_synapses: DefaultDict[Any, List[Any]] = defaultdict(list)
        self.post_synapses: DefaultDict[Any, List[Any]] = defaultdict(list)
        self.input_layer_name: Optional[str] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._input_buffer: List[float] = []
        self.network_id: Optional[int] = self.config.get("network_id")
        self.config.setdefault("dt", None)

        # Optional: build a simple neuron population from a layer config.
        layers_cfg = self.config.get("layers")
        if isinstance(layers_cfg, list):
            try:
                from ..neurons import create_neuron  # type: ignore
            except Exception:  # pragma: no cover - defensive import
                create_neuron = None  # type: ignore

            neuron_params = self.config.get("neuron_params", {})
            next_id = 0
            for layer_cfg in layers_cfg:
                if not isinstance(layer_cfg, dict):
                    continue
                name = str(layer_cfg.get("name") or "")
                if not name:
                    continue
                size = int(layer_cfg.get("size") or 0)
                if size <= 0:
                    continue
                neuron_type = layer_cfg.get("neuron_type") or "lif"
                pop_type = str(layer_cfg.get("population_type") or "excitatory")

                layer_key = name.split("_")[0]
                base_params = neuron_params.get(layer_key, {}) if isinstance(neuron_params, dict) else {}
                if not isinstance(base_params, dict):
                    base_params = {}

                neuron_ids: List[int] = []
                neurons: List[Any] = []
                for _ in range(size):
                    nid = next_id
                    next_id += 1
                    if create_neuron is not None:
                        params_for_neuron = dict(base_params)
                        params_for_neuron.setdefault("population_type", pop_type)
                        params_for_neuron.setdefault("is_inhibitory", pop_type == "inhibitory")
                        layer_cell_type = layer_cfg.get("cell_type") if isinstance(layer_cfg, dict) else None
                        if layer_cell_type is not None:
                            params_for_neuron.setdefault("cell_type", layer_cell_type)
                        neuron = create_neuron(neuron_type, neuron_id=nid, params=params_for_neuron)
                    else:
                        neuron = {"neuron_id": nid, "type": str(neuron_type)}
                    self.neurons[nid] = neuron
                    neuron_ids.append(nid)
                    neurons.append(neuron)

                self.layers[name] = Layer(
                    name=name,
                    neuron_type=str(neuron_type),
                    neurons=neurons,
                    neuron_ids=neuron_ids,
                    population_type=pop_type,
                )

    def set_input(self, inputs: List[float]) -> None:
        """Store the most recent external input for the network."""

        self._input_buffer = list(inputs)

    def reset(self) -> None:
        """Reset transient simulation state used by the high level modules."""

        self._input_buffer = []

    def add_synapse(self, pre_neuron_id: int, post_neuron_id: int, synapse_type: str, params: Dict[str, Any]) -> Any:
        """Create a lightweight synapse record (legacy compatibility)."""

        synapse_id = (int(pre_neuron_id), int(post_neuron_id), str(synapse_type))
        record = {"pre": int(pre_neuron_id), "post": int(post_neuron_id), "type": str(synapse_type), **dict(params or {})}
        self.synapses[synapse_id] = record
        self.pre_synapses[int(pre_neuron_id)].append(record)
        self.post_synapses[int(post_neuron_id)].append(record)
        return synapse_id

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Advance the network by ``dt`` milliseconds."""

        if dt is not None:
            self.config["dt"] = dt

        return {"spikes": [], "voltages": {}, "weights": {}}


__all__ = ["Layer", "NeuralNetwork"]
