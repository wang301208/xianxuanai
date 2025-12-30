from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import heapq
import importlib
import inspect
import json
import logging
import math
import random
import os
import numpy as np
from urllib import request as urllib_request
from urllib.error import URLError
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import Executor
from typing import Any, Callable, Dict, Mapping, Sequence, List, Optional, Type

from modules.brain.neuroplasticity import Neuroplasticity
from .temporal_encoding import decode_average_rate, decode_spike_counts, latency_encode, rate_encode
from modules.environment.registry import get_hardware_registry


logger = logging.getLogger(__name__)


@dataclass
class SpikingNetworkConfig:
    """Configuration container for building spiking neural networks."""

    n_neurons: int
    neuron: str = "lif"
    neuron_params: Mapping[str, Any] = field(default_factory=dict)
    weights: Sequence[Sequence[float]] | None = None
    idle_skip: bool = False
    plasticity: str | None = "stdp"
    learning_rate: float = 0.1
    max_duration: int | None = None
    convergence_window: int | None = None
    convergence_threshold: float | None = None
    convergence_patience: int = 3
    backend: str | None = None
    hardware_options: Mapping[str, Any] = field(default_factory=dict)
    fallback_to_simulation: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SpikingNetworkConfig":
        return cls(
            n_neurons=data["n_neurons"],
            neuron=data.get("neuron", "lif"),
            neuron_params=data.get("neuron_params", {}),
            weights=data.get("weights"),
            idle_skip=data.get("idle_skip", False),
            plasticity=data.get("plasticity", "stdp"),
            learning_rate=data.get("learning_rate", 0.1),
            max_duration=data.get("max_duration"),
            convergence_window=data.get("convergence_window"),
            convergence_threshold=data.get("convergence_threshold"),
            convergence_patience=data.get("convergence_patience", 3),
            backend=data.get("backend"),
            hardware_options=data.get("hardware_options", {}),
            fallback_to_simulation=data.get("fallback_to_simulation", True),
        )

    def create(self) -> "SpikingNeuralNetwork":
        neuron_map = {
            "lif": LIFNeuronModel,
            "adex": AdExNeuronModel,
        }
        neuron_key = self.neuron.lower()
        if neuron_key not in neuron_map:
            raise ValueError(f"Unknown neuron model '{self.neuron}'")
        neuron_cls = neuron_map[neuron_key]
        return SpikingNeuralNetwork(
            self.n_neurons,
            weights=self.weights,
            idle_skip=self.idle_skip,
            neuron_model_cls=neuron_cls,
            neuron_model_kwargs=dict(self.neuron_params),
            plasticity_mode=self.plasticity,
            learning_rate=self.learning_rate,
            max_duration=self.max_duration,
            convergence_window=self.convergence_window,
            convergence_threshold=self.convergence_threshold,
            convergence_patience=self.convergence_patience,
        )

    def create_backend(
        self,
        *,
        backend: str | None = None,
        fallback_to_simulation: bool | None = None,
        **backend_kwargs,
    ) -> "NeuromorphicBackend":
        """Build a reusable backend, optionally targeting hardware adapters."""

        selected = backend or self.backend
        if not selected:
            return NeuromorphicBackend(config=self, **backend_kwargs)

        backend_key = selected.lower()
        if backend_key in {"sim", "software", "simulation", "numpy"}:
            return NeuromorphicBackend(config=self, **backend_kwargs)

        options: Dict[str, Any] = {}
        if isinstance(self.hardware_options, Mapping):
            options.update(dict(self.hardware_options))
        options.update(backend_kwargs)
        factory = HardwareBackendRegistry.get(backend_key)
        if factory is None:
            raise ValueError(f"Unknown neuromorphic backend '{selected}'")
        fallback = (
            self.fallback_to_simulation
            if fallback_to_simulation is None
            else bool(fallback_to_simulation)
        )
        return factory(self, fallback_to_simulation=fallback, **options)



class EventQueue:
    """Priority queue managing spike events by timestamp."""

    def __init__(self) -> None:
        self._queue: list[tuple[float, list[float]]] = []

    def push(self, time: float, inputs: list[float]) -> None:
        heapq.heappush(self._queue, (time, inputs))

    def pop(self) -> tuple[float, list[float]]:
        return heapq.heappop(self._queue)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self._queue)


class NeuronModel(ABC):
    """Abstract base class for neuron dynamics."""

    size: int

    @abstractmethod
    def step(self, inputs: Sequence[float]) -> list[int]:
        """Advance the neuron state given input currents."""

    @abstractmethod
    def reset_state(self) -> None:
        """Reset membrane potentials and auxiliary state."""


class LIFNeuronModel(NeuronModel):
    """Leaky integrate-and-fire neuron population using NumPy."""

    def __init__(
        self,
        size: int,
        *,
        decay: float = 0.9,
        threshold: float = 1.0,
        reset: float = 0.0,
        refractory_period: int = 0,
        dynamic_threshold: float = 0.0,
        noise: float | None = None,
    ) -> None:
        self.size = size
        self.decay = decay
        self.threshold = threshold
        self.reset_value = reset
        self.refractory_period = refractory_period
        self.dynamic_threshold = dynamic_threshold
        self.noise = noise
        self.potentials = np.zeros(size, dtype=float)
        self.refractory = np.zeros(size, dtype=int)
        self.adaptation = np.zeros(size, dtype=float)

    def reset_state(self) -> None:
        self.potentials.fill(self.reset_value)
        self.refractory.fill(0)
        self.adaptation.fill(0.0)

    def step(self, inputs: Sequence[float]) -> list[int]:
        current = np.asarray(inputs, dtype=float)
        if current.shape[0] != self.size:
            raise ValueError("input size does not match neuron population")

        active = self.refractory <= 0
        inactive = ~active

        self.refractory[inactive] -= 1
        self.refractory = np.maximum(self.refractory, 0)

        self.potentials[active] = self.potentials[active] * self.decay + current[active]
        self.potentials[inactive] = self.reset_value

        if self.noise is not None:
            noise = np.random.normal(0.0, self.noise, size=self.size)
            self.potentials[active] += noise[active]

        thresholds = self.threshold + self.adaptation
        fire_mask = active & (self.potentials >= thresholds)

        spikes = fire_mask.astype(int)
        self.potentials[fire_mask] = self.reset_value
        self.refractory[fire_mask] = self.refractory_period
        self.adaptation[fire_mask] += self.dynamic_threshold
        self.adaptation[~fire_mask] *= self.decay
        return spikes.tolist()


class AdExNeuronModel(NeuronModel):
    """Adaptive exponential integrate-and-fire neurons (NumPy)."""

    def __init__(
        self,
        size: int,
        *,
        tau_m: float = 20.0,
        tau_w: float = 100.0,
        a: float = 0.0,
        b: float = 0.02,
        v_reset: float = -65.0,
        v_threshold: float = -50.0,
        delta_t: float = 2.0,
        v_peak: float = 20.0,
        timestep: float = 1.0,
    ) -> None:
        self.size = size
        self.tau_m = tau_m
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.delta_t = delta_t
        self.v_peak = v_peak
        self.timestep = timestep
        self.v = np.full(size, v_reset, dtype=float)
        self.w = np.zeros(size, dtype=float)

    def reset_state(self) -> None:
        self.v.fill(self.v_reset)
        self.w.fill(0.0)

    def step(self, inputs: Sequence[float]) -> list[int]:
        current = np.asarray(inputs, dtype=float)
        if current.shape[0] != self.size:
            raise ValueError("input size does not match neuron population")

        dv = (
            -(self.v - self.v_reset)
            + self.delta_t * np.exp((self.v - self.v_threshold) / self.delta_t)
            - self.w
            + current
        ) * (self.timestep / self.tau_m)
        self.v += dv
        dw = (self.a * (self.v - self.v_reset) - self.w) * (self.timestep / self.tau_w)
        self.w += dw

        fire_mask = self.v >= self.v_peak
        spikes = fire_mask.astype(int)
        self.v[fire_mask] = self.v_reset
        self.w[fire_mask] += self.b
        return spikes.tolist()


class SynapseModel(ABC):
    """Abstract base class for synaptic connectivity."""

    @abstractmethod
    def propagate(self, pre_spikes: Sequence[int]) -> list[float]:
        """Propagate spikes to produce postsynaptic currents."""

    @abstractmethod
    def adapt(
        self,
        pre_spike_times: Sequence[float | None],
        post_spike_times: Sequence[float | None],
    ) -> None:
        """Update weights given spike timing."""

    @abstractmethod
    def reset_state(self) -> None:
        """Reset synaptic weights and plasticity state."""


class DenseSynapseModel(SynapseModel):
    """Fully connected synapses with optional plasticity using NumPy."""

    def __init__(
        self,
        weights: Sequence[Sequence[float]],
        *,
        learning_rate: float = 0.1,
        plasticity: Neuroplasticity | None = None,
        plasticity_mode: str | None = "stdp",
    ) -> None:
        self._initial_weights = np.asarray(weights, dtype=float)
        self.weights = self._initial_weights.copy()
        self.base_learning_rate = float(learning_rate)
        self.learning_rate = float(learning_rate)
        self.plasticity_mode = plasticity_mode.lower() if isinstance(plasticity_mode, str) else None
        self._plasticity_cls = plasticity.__class__ if plasticity is not None else None
        self.plasticity = plasticity if plasticity is not None else Neuroplasticity()
        self.weight_decay = 0.0
        self._modulation_cache: Dict[str, float] = {}

    def reset_state(self) -> None:
        self.weights = self._initial_weights.copy()
        if self._plasticity_cls is not None:
            self.plasticity = self._plasticity_cls()
        self.learning_rate = self.base_learning_rate
        self.weight_decay = 0.0
        self._modulation_cache = {}
        if hasattr(self.plasticity, "update_modulation"):
            self.plasticity.update_modulation(None)

    def propagate(self, pre_spikes: Sequence[int]) -> list[float]:
        pre = np.asarray(pre_spikes, dtype=float)
        postsynaptic = pre @ self.weights
        return postsynaptic.tolist()

    def adapt(
        self,
        pre_spike_times: Sequence[float | None],
        post_spike_times: Sequence[float | None],
    ) -> None:
        if self.plasticity_mode is None:
            return
        pre_times = [t for t in enumerate(pre_spike_times) if t[1] is not None]
        post_times = [t for t in enumerate(post_spike_times) if t[1] is not None]
        for pre_idx, pre_time in pre_times:
            for post_idx, post_time in post_times:
                mode = self.plasticity_mode
                if mode == "stdp":
                    delta = self.plasticity.adapt_connections(pre_time, post_time)
                elif mode == "hebbian":
                    delta = 1.0
                elif mode == "anti_hebbian":
                    delta = -1.0
                elif mode == "oja":
                    delta = 1.0 - float(self.weights[pre_idx, post_idx])
                else:
                    delta = 0.0
                self.weights[pre_idx, post_idx] += self.learning_rate * delta
        if self.weight_decay:
            self.weights *= (1.0 - self.weight_decay)

    def update_modulation(self, modulation: Mapping[str, float] | None) -> None:
        if not modulation:
            self.learning_rate = self.base_learning_rate
            self.weight_decay = 0.0
            self._modulation_cache = {}
            if hasattr(self.plasticity, "update_modulation"):
                self.plasticity.update_modulation(None)
            return
        filtered: Dict[str, float] = {
            key: float(value)
            for key, value in modulation.items()
            if isinstance(value, (int, float))
        }
        amplitude = float(np.clip(filtered.get("amplitude_norm", filtered.get("amplitude", 0.0)), 0.0, 1.0))
        synchrony = float(np.clip(filtered.get("synchrony_norm", filtered.get("synchrony_index", 0.0)), 0.0, 1.0))
        rhythmicity = float(np.clip(filtered.get("rhythmicity", 0.0), 0.0, 1.0))
        gate = float(np.clip(filtered.get("plasticity_gate", (amplitude + synchrony) * 0.5), 0.0, 2.0))
        learning_gain = 0.5 + amplitude * 0.75 + synchrony * 0.25
        learning_gain += rhythmicity * 0.25
        self.learning_rate = self.base_learning_rate * max(0.1, learning_gain)
        decay_term = (1.0 - synchrony) * 0.05 + max(0.0, 0.5 - amplitude) * 0.02
        self.weight_decay = float(np.clip(decay_term, 0.0, 0.2))
        self._modulation_cache = {
            "amplitude": amplitude,
            "synchrony": synchrony,
            "rhythmicity": rhythmicity,
            "plasticity_gate": gate,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
        if hasattr(self.plasticity, "update_modulation"):
            self.plasticity.update_modulation({
                **filtered,
                "plasticity_gate": gate,
                "learning_rate": self.learning_rate,
            })

    @property
    def modulation_cache(self) -> Dict[str, float]:
        return dict(self._modulation_cache)


class SpikingNeuralNetwork:
    """Spiking neural network with pluggable neuron and synapse models."""

    LeakyIntegrateFireNeurons = LIFNeuronModel
    AdaptiveExponentialNeurons = AdExNeuronModel
    DynamicSynapses = DenseSynapseModel

    def __init__(
        self,
        n_neurons,
        *,
        decay=0.9,
        threshold=1.0,
        reset=0.0,
        weights=None,
        refractory_period=0,
        dynamic_threshold=0.0,
        noise=None,
        idle_skip=False,
        neuron_model: NeuronModel | None = None,
        neuron_model_cls: type[NeuronModel] | None = None,
        neuron_model_kwargs: dict | None = None,
        synapse_model: SynapseModel | None = None,
        plasticity_mode: str | None = "stdp",
        learning_rate: float = 0.1,
        max_duration: int | None = None,
        convergence_window: int | None = None,
        convergence_threshold: float | None = None,
        convergence_patience: int | None = 3,
    ) -> None:
        if neuron_model is None:
            if neuron_model_cls is None:
                neuron_model = LIFNeuronModel(
                    n_neurons,
                    decay=decay,
                    threshold=threshold,
                    reset=reset,
                    refractory_period=refractory_period,
                    dynamic_threshold=dynamic_threshold,
                    noise=noise,
                )
            else:
                params = neuron_model_kwargs or {}
                neuron_model = neuron_model_cls(n_neurons, **params)
        elif neuron_model.size != n_neurons:
            raise ValueError("neuron_model size must match n_neurons")

        if synapse_model is None:
            if weights is None:
                weights = np.eye(n_neurons, dtype=float)
            plasticity = None
            if plasticity_mode is None:
                plasticity = None
            else:
                plasticity = Neuroplasticity()
            synapse_model = DenseSynapseModel(
                weights,
                learning_rate=learning_rate,
                plasticity=plasticity,
                plasticity_mode=plasticity_mode,
            )

        self.neurons = neuron_model
        self.synapses = synapse_model
        self.spike_times = [None] * n_neurons
        self._modulation_state: Dict[str, float] = {}
        if hasattr(self.synapses, "update_modulation"):
            self.synapses.update_modulation(None)
        self.idle_skip = idle_skip
        self.energy_usage = 0
        self.idle_skipped_cycles = 0
        self.max_duration = (
            int(max_duration) if max_duration is not None else None
        )
        if self.max_duration is not None and self.max_duration <= 0:
            self.max_duration = 1
        self.convergence_threshold = (
            float(convergence_threshold)
            if convergence_threshold is not None
            else None
        )
        if convergence_window is not None:
            window_value = int(convergence_window)
            self.convergence_window = window_value if window_value > 0 else None
        else:
            self.convergence_window = None
        if convergence_patience is None:
            patience_value = 3
        else:
            patience_value = int(convergence_patience)
        self.convergence_patience = patience_value if patience_value > 0 else 1

    def reset_state(self) -> None:
        if hasattr(self.neurons, "reset_state"):
            self.neurons.reset_state()
        if hasattr(self.synapses, "reset_state"):
            self.synapses.reset_state()
            if hasattr(self.synapses, "update_modulation"):
                if self._modulation_state:
                    self.synapses.update_modulation(self._modulation_state)
                else:
                    self.synapses.update_modulation(None)
        self.spike_times = [None] * len(self.spike_times)
        self.energy_usage = 0
        self.idle_skipped_cycles = 0

    def apply_modulation(self, modulation: Mapping[str, float] | None) -> None:
        if modulation:
            state = {
                key: float(value)
                for key, value in modulation.items()
                if isinstance(value, (int, float))
            }
        else:
            state = {}
        self._modulation_state = state
        if hasattr(self.synapses, "update_modulation"):
            self.synapses.update_modulation(state if state else None)
        if hasattr(self.neurons, "update_modulation") and state:
            try:  # pragma: no cover - optional integration hook
                self.neurons.update_modulation(state)
            except Exception:
                pass

    @property
    def modulation_state(self) -> Dict[str, float]:
        return dict(self._modulation_state)

    def _run_internal(
        self,
        input_events,
        encoder=None,
        *,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
    ):
        queue = EventQueue()
        self.energy_usage = 0
        self.idle_skipped_cycles = 0
        encoder_kwargs = dict(encoder_kwargs or {})
        if encoder is not None:
            for t, analog in enumerate(input_events):
                if not self.idle_skip or any(analog):
                    for time, inputs in encoder(
                        analog, t_start=t, **encoder_kwargs
                    ):
                        queue.push(time, inputs)
                else:
                    self.idle_skipped_cycles += 1
        elif input_events and (
            not isinstance(input_events[0], tuple)
            or len(input_events[0]) != 2
            or not isinstance(input_events[0][0], (int, float))
        ):
            for t, inputs in enumerate(input_events):
                if not self.idle_skip or any(inputs):
                    queue.push(t, inputs)
                else:
                    self.idle_skipped_cycles += 1
        else:
            for t, inputs in input_events:
                if not self.idle_skip or any(inputs):
                    queue.push(t, inputs)
                else:
                    self.idle_skipped_cycles += 1

        outputs: list[tuple[float, list[int]]] = []

        processed = 0
        configured_max = max_duration if max_duration is not None else self.max_duration
        if configured_max is not None:
            max_events = max(1, int(configured_max))
        else:
            estimated_length = (
                len(input_events)
                if isinstance(input_events, Sequence)
                else 0
            )
            baseline_limit = 32 if estimated_length == 0 else estimated_length
            max_events = max(baseline_limit, self.neurons.size * 32)

        threshold = (
            float(convergence_threshold)
            if convergence_threshold is not None
            else (
                float(self.convergence_threshold)
                if self.convergence_threshold is not None
                else None
            )
        )
        window = (
            int(convergence_window)
            if convergence_window is not None
            else self.convergence_window
        )
        if window is not None and window <= 0:
            window = None
        patience = (
            int(convergence_patience)
            if convergence_patience is not None
            else self.convergence_patience
        )
        if patience is None or patience <= 0:
            patience = 1

        recent_spike_totals = (
            deque(maxlen=window) if threshold is not None and window else None
        )
        low_activity_streak = 0

        while queue:
            if processed >= max_events:
                break
            time, inputs = queue.pop()
            self.energy_usage += 1
            spikes = self.neurons.step(inputs)
            for idx, spike in enumerate(spikes):
                if spike:
                    self.spike_times[idx] = time

            if any(spikes):
                self.synapses.adapt(self.spike_times, self.spike_times)

            outputs.append((time, spikes))
            processed += 1

            should_stop = False
            if recent_spike_totals is not None:
                recent_spike_totals.append(sum(spikes))
                if (
                    recent_spike_totals.maxlen
                    and len(recent_spike_totals) == recent_spike_totals.maxlen
                ):
                    avg_spikes = sum(recent_spike_totals) / float(
                        recent_spike_totals.maxlen
                    )
                    if avg_spikes <= (threshold or 0.0):
                        low_activity_streak += 1
                    else:
                        low_activity_streak = 0
                    if low_activity_streak >= patience:
                        should_stop = True

            if should_stop:
                break

            currents = self.synapses.propagate(spikes)
            if any(currents):
                queue.push(time + 1, currents)

        outputs.sort(key=lambda x: x[0])
        return outputs

    def run(
        self,
        input_events,
        encoder=None,
        *,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
        neuromodulation: Mapping[str, float] | None = None,
    ):
        """Run the network using an event-driven simulation."""
        if neuromodulation is not None:
            self.apply_modulation(neuromodulation)
        return self._run_internal(
            input_events,
            encoder,
            encoder_kwargs=encoder_kwargs,
            max_duration=max_duration,
            convergence_threshold=convergence_threshold,
            convergence_window=convergence_window,
            convergence_patience=convergence_patience,
        )

    async def run_async(
        self,
        input_events,
        encoder=None,
        *,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
    ):
        """Asynchronously run the network using ``asyncio``."""
        return await asyncio.to_thread(
            self._run_internal,
            input_events,
            encoder,
            encoder_kwargs=encoder_kwargs,
            max_duration=max_duration,
            convergence_threshold=convergence_threshold,
            convergence_window=convergence_window,
            convergence_patience=convergence_patience,
        )

@dataclass
class NeuromorphicRunResult:
    """Container for spike outputs and derived telemetry."""

    spike_events: List[tuple[float, List[int]]]
    energy_used: float
    idle_skipped: int
    spike_counts: List[int] = field(default_factory=list)
    average_rate: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def spikes(self) -> List[List[int]]:
        return [spike for _, spike in self.spike_events]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spike_events": self.spike_events,
            "energy_used": self.energy_used,
            "idle_skipped": self.idle_skipped,
            "spike_counts": list(self.spike_counts),
            "average_rate": list(self.average_rate),
            "metadata": dict(self.metadata),
        }


class NeuromorphicBackend:
    """Reusable wrapper around :class:`SpikingNeuralNetwork`."""

    _ENCODERS: Dict[str, Callable[..., List[tuple[float, List[int]]]]] = {}

    def __init__(
        self,
        *,
        config: SpikingNetworkConfig | None = None,
        network: SpikingNeuralNetwork | None = None,
        auto_reset: bool = True,
    ) -> None:
        if network is None:
            if config is None:
                raise ValueError("Either config or network must be provided")
            network = config.create()
        self.config = config
        self.network = network
        self.auto_reset = auto_reset
        self._last_modulation: Dict[str, float] = {}
        if not NeuromorphicBackend._ENCODERS:
            NeuromorphicBackend._ENCODERS = {
                "latency": self._latency_encoder,
                "rate": self._rate_encoder,
            }
        self._capability_cache: Optional[Dict[str, Any]] = None

    @staticmethod
    def _latency_encoder(signal: Sequence[float], *, t_start: float = 0.0, t_scale: float = 1.0) -> List[tuple[float, List[int]]]:
        return latency_encode(list(signal), t_start=t_start, t_scale=t_scale)

    @staticmethod
    def _rate_encoder(signal: Sequence[float], *, steps: int = 5, t_start: float = 0.0) -> List[tuple[float, List[int]]]:
        trains = rate_encode(signal, steps=steps)
        return [(t_start + idx, spikes) for idx, spikes in enumerate(trains)]

    def clone(self) -> "NeuromorphicBackend":
        if self.config is None:
            raise ValueError("Cannot clone backend without original config")
        return NeuromorphicBackend(config=self.config, auto_reset=self.auto_reset)

    def reset_state(self) -> None:
        self.network.reset_state()
        if self._last_modulation:
            self.network.apply_modulation(self._last_modulation)

    def run_events(
        self,
        events,
        *,
        encoder: Callable[..., List[tuple[float, List[int]]]] | None = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder: str | None = "counts",
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        neuromodulation: Optional[Mapping[str, float]] = None,
        reset: Optional[bool] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
    ) -> NeuromorphicRunResult:
        if reset or (reset is None and self.auto_reset):
            self.reset_state()
        encoder_kwargs = dict(encoder_kwargs or {})
        modulation = None
        if neuromodulation is not None:
            modulation = {
                key: float(value)
                for key, value in neuromodulation.items()
                if isinstance(value, (int, float))
            }
            self._last_modulation = dict(modulation)
        elif self._last_modulation:
            modulation = dict(self._last_modulation)
        outputs = self.network.run(
            events,
            encoder=encoder,
            encoder_kwargs=encoder_kwargs,
            max_duration=max_duration,
            convergence_threshold=convergence_threshold,
            convergence_window=convergence_window,
            convergence_patience=convergence_patience,
            neuromodulation=modulation,
        )
        counts: List[int] = []
        rates: List[float] = []
        if decoder:
            key = decoder.lower()
            decoder_kwargs = decoder_kwargs or {}
            if key in {"counts", "all"}:
                counts = decode_spike_counts(outputs)
            if key in {"rate", "all"}:
                window = decoder_kwargs.get("window")
                if window is None:
                    window = len(outputs) or 1
                rates = decode_average_rate(outputs, window=float(window))
        result_metadata = dict(metadata or {})
        if modulation:
            result_metadata.setdefault("neuromodulation", dict(modulation))
        synapse_state = getattr(self.network.synapses, "modulation_cache", None)
        if synapse_state:
            result_metadata.setdefault("synapse_modulation", dict(synapse_state))
        return NeuromorphicRunResult(
            spike_events=outputs,
            energy_used=self.network.energy_usage,
            idle_skipped=self.network.idle_skipped_cycles,
            spike_counts=counts,
            average_rate=rates,
            metadata=result_metadata,
        )

    def run_sequence(
        self,
        signal,
        *,
        encoding: str | None = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder: str | None = "counts",
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        neuromodulation: Optional[Mapping[str, float]] = None,
        reset: Optional[bool] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
    ) -> NeuromorphicRunResult:
        encoder = None
        encoder_kwargs = dict(encoder_kwargs or {})
        prepared = signal
        if encoding:
            key = encoding.lower()
            if key not in self._ENCODERS:
                raise ValueError(f"Unsupported encoding '{encoding}'")
            encoder = self._ENCODERS[key]
        return self.run_events(
            prepared,
            encoder=encoder,
            encoder_kwargs=encoder_kwargs,
            decoder=decoder,
            decoder_kwargs=decoder_kwargs,
            metadata=metadata,
            neuromodulation=neuromodulation,
            reset=reset,
            max_duration=max_duration,
            convergence_threshold=convergence_threshold,
            convergence_window=convergence_window,
            convergence_patience=convergence_patience,
        )

    def run_batch(
        self,
        sequences,
        *,
        encoding: str | None = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder: str | None = "counts",
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        reset_each: bool | None = None,
        neuromodulation: Optional[Mapping[str, float]] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
    ) -> List[NeuromorphicRunResult]:
        results: List[NeuromorphicRunResult] = []
        for sequence in sequences:
            results.append(
                self.run_sequence(
                    sequence,
                    encoding=encoding,
                    encoder_kwargs=encoder_kwargs,
                    decoder=decoder,
                    decoder_kwargs=decoder_kwargs,
                    metadata=None,
                    neuromodulation=neuromodulation,
                    reset=reset_each if reset_each is not None else True,
                    max_duration=max_duration,
                    convergence_threshold=convergence_threshold,
                    convergence_window=convergence_window,
                    convergence_patience=convergence_patience,
                )
            )
        return results

    def capability_profile(self) -> Dict[str, Any]:
        """Return a lightweight description of backend capabilities."""

        if getattr(self, "_capability_cache", None) is not None:
            return dict(self._capability_cache)
        n_neurons = getattr(self.network, "n_neurons", None)
        if n_neurons is None:
            spike_times = getattr(self.network, "spike_times", None)
            if isinstance(spike_times, Sequence):
                n_neurons = len(spike_times)
        profile = {
            "backend_type": "software",
            "supports_hardware": False,
            "encoders": sorted(self._ENCODERS.keys()),
            "n_neurons": int(n_neurons) if n_neurons is not None else None,
        }
        self._capability_cache = dict(profile)
        return dict(profile)


class HardwareIntegrationError(RuntimeError):
    """Raised when a hardware backend cannot be initialised or executed."""


class BaseHardwareBackend(NeuromorphicBackend):
    """Common scaffolding for hardware-oriented backends."""

    hardware_name: str = "hardware"

    def __init__(
        self,
        config: SpikingNetworkConfig,
        *,
        auto_reset: bool = True,
        fallback_to_simulation: bool = True,
        **kwargs,
    ) -> None:
        worker_id = kwargs.pop("worker_id", None)
        self._worker_id = worker_id or os.getenv("NEUROMORPHIC_WORKER_ID") or os.getenv("WORKER_ID") or self.hardware_name
        self.fallback_to_simulation = bool(fallback_to_simulation)
        self.hardware_available = False
        self.hardware_error: Exception | None = None
        self.hardware_context: Dict[str, Any] = {}
        super().__init__(config=config, auto_reset=auto_reset)
        try:
            context = self._initialise_hardware(config, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive logging path
            self.hardware_error = exc
            if not self.fallback_to_simulation:
                raise HardwareIntegrationError(
                    f"Failed to initialise {self.hardware_name} backend"
                ) from exc
            logger.warning(
                "Falling back to software neuromorphic backend for %s: %s",
                self.hardware_name,
                exc,
            )
            self.hardware_available = False
        else:
            self.hardware_available = True
            if isinstance(context, Mapping):
                self.hardware_context = dict(context)
        self._register_capabilities()

    def _initialise_hardware(
        self, config: SpikingNetworkConfig, **kwargs
    ) -> Mapping[str, Any] | None:
        raise NotImplementedError

    def _hardware_reset(self) -> None:
        """Hook executed when :meth:`reset_state` is called."""

    def _run_on_hardware(
        self,
        events,
        *,
        encoder: Callable[..., List[tuple[float, List[int]]]] | None = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder: str | None = None,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        neuromodulation: Optional[Mapping[str, float]] = None,
        reset: Optional[bool] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
    ) -> NeuromorphicRunResult:
        raise NotImplementedError

    def reset_state(self) -> None:
        super().reset_state()
        if self.hardware_available:
            try:
                self._hardware_reset()
            except Exception as exc:  # pragma: no cover - defensive logging path
                self.hardware_error = exc
                if not self.fallback_to_simulation:
                    raise HardwareIntegrationError(
                        f"Failed to reset {self.hardware_name} backend"
                    ) from exc
                logger.warning(
                    "Hardware reset failed for %s; disabling hardware backend: %s",
                    self.hardware_name,
                    exc,
                )
                self.hardware_available = False
                self._register_capabilities()

    def run_events(
        self,
        events,
        *,
        encoder: Callable[..., List[tuple[float, List[int]]]] | None = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder: str | None = "counts",
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        neuromodulation: Optional[Mapping[str, float]] = None,
        reset: Optional[bool] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
    ) -> NeuromorphicRunResult:
        if not self.hardware_available:
            return super().run_events(
                events,
                encoder=encoder,
                encoder_kwargs=encoder_kwargs,
                decoder=decoder,
                decoder_kwargs=decoder_kwargs,
                metadata=metadata,
                neuromodulation=neuromodulation,
                reset=reset,
                max_duration=max_duration,
                convergence_threshold=convergence_threshold,
                convergence_window=convergence_window,
                convergence_patience=convergence_patience,
            )
        try:
            result = self._run_on_hardware(
                events,
                encoder=encoder,
                encoder_kwargs=encoder_kwargs,
                decoder=decoder,
                decoder_kwargs=decoder_kwargs,
                metadata=metadata,
                neuromodulation=neuromodulation,
                reset=reset,
                max_duration=max_duration,
                convergence_threshold=convergence_threshold,
                convergence_window=convergence_window,
                convergence_patience=convergence_patience,
            )
        except Exception as exc:
            self.hardware_error = exc
            if not self.fallback_to_simulation:
                raise HardwareIntegrationError(
                    f"{self.hardware_name} execution failed"
                ) from exc
            logger.warning(
                "Hardware backend %s failed; falling back to simulation: %s",
                self.hardware_name,
                exc,
            )
            self.hardware_available = False
            self._register_capabilities()
            return super().run_events(
                events,
                encoder=encoder,
                encoder_kwargs=encoder_kwargs,
                decoder=decoder,
                decoder_kwargs=decoder_kwargs,
                metadata=metadata,
                neuromodulation=neuromodulation,
                reset=reset,
                max_duration=max_duration,
                convergence_threshold=convergence_threshold,
                convergence_window=convergence_window,
                convergence_patience=convergence_patience,
            )
        return self._normalise_result(
            result,
            decoder=decoder,
            decoder_kwargs=decoder_kwargs,
            metadata=metadata,
        )

    def capability_profile(self) -> Dict[str, Any]:
        profile = super().capability_profile()
        profile["backend_type"] = "hardware"
        profile["supports_hardware"] = bool(self.hardware_available)
        profile["hardware"] = {
            "name": self.hardware_name,
            "available": bool(self.hardware_available),
            "context": self._serialise_context(self.hardware_context),
        }
        if self.hardware_error is not None:
            profile["hardware"]["last_error"] = repr(self.hardware_error)
        return profile

    @staticmethod
    def _serialise_context(context: Mapping[str, Any]) -> Dict[str, Any]:
        serialised: Dict[str, Any] = {}
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                serialised[str(key)] = value
        return serialised

    def _register_capabilities(self) -> None:
        worker_id = self._worker_id or self.hardware_name
        try:
            registry = get_hardware_registry()
            registry.register(worker_id, self.capability_profile())
        except Exception:  # pragma: no cover - best effort
            logger.debug("Failed to register hardware capabilities for %s", worker_id, exc_info=True)

    def _normalise_events(self, outputs: Any) -> List[tuple[float, List[int]]]:
        if isinstance(outputs, Mapping):
            if "spike_events" in outputs:
                return self._normalise_events(outputs["spike_events"])
            if "events" in outputs:
                return self._normalise_events(outputs["events"])
        events: List[tuple[float, List[int]]] = []
        if outputs is None:
            return events
        if isinstance(outputs, np.ndarray):
            outputs = outputs.tolist()
        if isinstance(outputs, Sequence):
            for idx, item in enumerate(outputs):
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], (int, float))
                ):
                    time, spikes = item
                else:
                    time = idx
                    spikes = item
                if isinstance(spikes, np.ndarray):
                    spikes = spikes.tolist()
                elif isinstance(spikes, Mapping):
                    spikes = list(spikes.values())
                elif not isinstance(spikes, Sequence):
                    spikes = [spikes]
                numeric = []
                for value in spikes:
                    if isinstance(value, (int, float)):
                        numeric.append(int(round(float(value))))
                    else:
                        numeric.append(int(bool(value)))
                events.append((float(time), numeric))
        return events

    def _augment_metadata(
        self,
        metadata: Optional[Dict[str, Any]],
        *,
        base: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        combined: Dict[str, Any] = dict(base or {})
        if metadata:
            combined.update(metadata)
        combined.setdefault("backend", self.hardware_name)
        if self.hardware_context:
            combined.setdefault("hardware_context", dict(self.hardware_context))
        if self.hardware_error and not self.hardware_available:
            combined.setdefault("hardware_warning", str(self.hardware_error))
        return combined

    def _normalise_result(
        self,
        result: Any,
        *,
        decoder: str | None,
        decoder_kwargs: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
    ) -> NeuromorphicRunResult:
        if isinstance(result, NeuromorphicRunResult):
            result.metadata = self._augment_metadata(result.metadata, base=metadata)
            return result

        if isinstance(result, tuple) and len(result) == 2:
            events_part, meta_part = result
            base: Dict[str, Any] = {}
            if isinstance(meta_part, Mapping):
                base = dict(meta_part)
            else:
                base = {"telemetry": meta_part}
            result = {"spike_events": events_part, "metadata": base}

        events = self._normalise_events(result)
        base_metadata: Dict[str, Any] = {}
        counts: Optional[List[int]] = None
        rates: Optional[List[float]] = None
        energy = len(events)
        idle = 0
        if isinstance(result, Mapping):
            base_metadata = dict(result.get("metadata", {}))
            counts_data = result.get("spike_counts")
            if counts_data is not None:
                counts = [int(c) for c in counts_data]
            rates_data = result.get("average_rate")
            if rates_data is not None:
                rates = [float(r) for r in rates_data]
            energy = int(result.get("energy_used", energy))
            idle = int(result.get("idle_skipped", 0))

        key = decoder.lower() if decoder else None
        decoder_kwargs = dict(decoder_kwargs or {})
        if counts is None:
            if key in {"counts", "all"}:
                counts = decode_spike_counts(events)
            else:
                counts = []
        if rates is None:
            if key in {"rate", "all"}:
                window = decoder_kwargs.get("window")
                if window is None:
                    window = len(events) or 1
                rates = decode_average_rate(events, window=float(window))
            else:
                rates = []

        metadata_combined = self._augment_metadata(metadata, base=base_metadata)
        return NeuromorphicRunResult(
            spike_events=events,
            energy_used=float(energy),
            idle_skipped=int(idle),
            spike_counts=list(counts),
            average_rate=list(rates),
            metadata=metadata_combined,
        )


class CallableHardwareBackend(BaseHardwareBackend):
    """Generic backend delegating execution to user-provided callables."""

    hardware_name = "external-callable"

    def __init__(
        self,
        config: SpikingNetworkConfig,
        *,
        auto_reset: bool = True,
        fallback_to_simulation: bool = True,
        **kwargs,
    ) -> None:
        self._run_callable: Callable[..., Any] | None = None
        self._compile_callable: Callable[..., Any] | None = None
        self._reset_callable: Callable[[], Any] | None = None
        self._decode_callable: Callable[..., Any] | None = None
        self._describe_callable: Callable[[], Any] | None = None
        super().__init__(
            config,
            auto_reset=auto_reset,
            fallback_to_simulation=fallback_to_simulation,
            **kwargs,
        )

    @staticmethod
    def _normalise_http_headers(headers: Mapping[str, Any] | None) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        if not headers:
            return normalized
        for key, value in headers.items():
            try:
                normalized[str(key)] = str(value)
            except Exception:
                continue
        return normalized

    def _build_http_runner(
        self,
        endpoint: str,
        *,
        timeout: float | None = None,
        headers: Mapping[str, Any] | None = None,
    ) -> Callable[..., Any]:
        normalized_headers = self._normalise_http_headers(headers)
        normalized_headers.setdefault("Content-Type", "application/json")

        def _run(events, **kwargs):
            payload = {
                "events": events,
                "options": kwargs,
            }
            data = json.dumps(payload).encode("utf-8")
            request = urllib_request.Request(
                endpoint,
                data=data,
                headers=normalized_headers,
                method="POST",
            )
            try:
                with urllib_request.urlopen(request, timeout=timeout) as response:
                    raw = response.read()
            except URLError as exc:
                raise HardwareIntegrationError(
                    f"Failed to reach neuromorphic hardware at {endpoint}: {exc}"
                ) from exc
            except Exception as exc:  # pragma: no cover - defensive network handling
                raise HardwareIntegrationError(
                    f"Unexpected error contacting neuromorphic hardware: {exc}"
                ) from exc
            if not raw:
                return {}
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise HardwareIntegrationError(
                    "Hardware response was not valid JSON"
                ) from exc
            return decoded

        return _run

    def _build_software_emulation(
        self,
        config: SpikingNetworkConfig,
        *,
        describe: Mapping[str, Any] | None = None,
    ) -> Dict[str, Callable[..., Any]]:
        """Create run/reset/describe callables backed by the software simulator."""

        emulation_backend = NeuromorphicBackend(config=config, auto_reset=False)

        def run_fn(events, **kwargs):
            result = emulation_backend.run_events(
                events,
                encoder=kwargs.get("encoder"),
                encoder_kwargs=kwargs.get("encoder_kwargs"),
                decoder=kwargs.get("decoder"),
                decoder_kwargs=kwargs.get("decoder_kwargs"),
                metadata=kwargs.get("metadata"),
                neuromodulation=kwargs.get("neuromodulation"),
                reset=kwargs.get("reset"),
                max_duration=kwargs.get("max_duration"),
                convergence_threshold=kwargs.get("convergence_threshold"),
                convergence_window=kwargs.get("convergence_window"),
                convergence_patience=kwargs.get("convergence_patience"),
            )
            return result.to_dict()

        def reset_fn() -> None:
            emulation_backend.reset_state()

        def describe_fn() -> Dict[str, Any]:
            summary: Dict[str, Any] = {
                "mode": "software-emulation",
                "hardware": self.hardware_name,
                "neurons": int(config.n_neurons),
                "neuron_model": config.neuron,
            }
            if describe:
                summary.update({str(key): value for key, value in describe.items()})
            return summary

        return {
            "run_fn": run_fn,
            "reset_fn": reset_fn,
            "describe_fn": describe_fn,
        }

    def _initialise_hardware(
        self, config: SpikingNetworkConfig, **kwargs
    ) -> Mapping[str, Any] | None:
        run_fn = kwargs.pop("run_fn", None)
        compile_fn = kwargs.pop("compile_fn", None)
        reset_fn = kwargs.pop("reset_fn", None)
        describe_fn = kwargs.pop("describe_fn", None)
        decode_fn = kwargs.pop("decode_fn", None)
        http_endpoint = kwargs.pop("hardware_endpoint", None) or kwargs.pop("http_endpoint", None)
        http_headers = kwargs.pop("hardware_headers", None)
        http_timeout = kwargs.pop("hardware_timeout", None)
        if not callable(run_fn) and http_endpoint:
            run_fn = self._build_http_runner(
                str(http_endpoint),
                timeout=float(http_timeout) if http_timeout is not None else None,
                headers=http_headers,
            )
        if not callable(run_fn):
            raise HardwareIntegrationError(
                "Callable hardware backend requires a callable 'run_fn'."
            )
        if compile_fn is not None and not callable(compile_fn):
            raise HardwareIntegrationError("'compile_fn' must be callable if provided")
        if reset_fn is not None and not callable(reset_fn):
            raise HardwareIntegrationError("'reset_fn' must be callable if provided")
        if describe_fn is not None and not callable(describe_fn):
            raise HardwareIntegrationError("'describe_fn' must be callable if provided")
        if decode_fn is not None and not callable(decode_fn):
            raise HardwareIntegrationError("'decode_fn' must be callable if provided")
        self._run_callable = run_fn
        self._compile_callable = compile_fn
        self._reset_callable = reset_fn
        self._describe_callable = describe_fn
        self._decode_callable = decode_fn
        context: Dict[str, Any] = {}
        if self._compile_callable is not None:
            self._compile_callable(config, **kwargs)
        if self._describe_callable is not None:
            try:
                description = self._describe_callable()
                if description:
                    context["description"] = description
            except Exception:
                pass
        if kwargs or http_endpoint:
            context.setdefault("options", dict(kwargs))
        if http_endpoint:
            context.setdefault("hardware_endpoint", str(http_endpoint))
        return context

    def _hardware_reset(self) -> None:
        if self._reset_callable is not None:
            self._reset_callable()

    def _run_on_hardware(
        self,
        events,
        *,
        encoder: Callable[..., List[tuple[float, List[int]]]] | None = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder: str | None = None,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        neuromodulation: Optional[Mapping[str, float]] = None,
        reset: Optional[bool] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
    ) -> NeuromorphicRunResult:
        if self._run_callable is None:
            raise HardwareIntegrationError("Hardware run function is not initialised")
        potential_kwargs = {
            "encoder": encoder,
            "encoder_kwargs": dict(encoder_kwargs or {}),
            "decoder": decoder,
            "decoder_kwargs": dict(decoder_kwargs or {}),
            "metadata": metadata,
            "neuromodulation": neuromodulation,
            "reset": reset,
            "max_duration": max_duration,
            "convergence_threshold": convergence_threshold,
            "convergence_window": convergence_window,
            "convergence_patience": convergence_patience,
        }
        signature = inspect.signature(self._run_callable)
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        call_kwargs: Dict[str, Any] = {}
        if accepts_kwargs:
            call_kwargs = potential_kwargs
        else:
            for name, value in potential_kwargs.items():
                if name in signature.parameters:
                    call_kwargs[name] = value
        outputs = self._run_callable(events, **call_kwargs)
        if self._decode_callable is not None:
            decode_signature = inspect.signature(self._decode_callable)
            decode_kwargs = {}
            potential_decode_kwargs = {
                "decoder": decoder,
                "decoder_kwargs": dict(decoder_kwargs or {}),
            }
            if any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in decode_signature.parameters.values()
            ):
                decode_kwargs = potential_decode_kwargs
            else:
                for name, value in potential_decode_kwargs.items():
                    if name in decode_signature.parameters:
                        decode_kwargs[name] = value
            outputs = self._decode_callable(outputs, **decode_kwargs)
        return self._normalise_result(
            outputs,
            decoder=decoder,
            decoder_kwargs=decoder_kwargs,
            metadata=metadata,
        )


class LoihiHardwareBackend(CallableHardwareBackend):
    """Adapter for Intel Loihi hardware via user-provided runners."""

    hardware_name = "intel-loihi"

    def __init__(
        self,
        config: SpikingNetworkConfig,
        *,
        auto_reset: bool = True,
        fallback_to_simulation: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            config,
            auto_reset=auto_reset,
            fallback_to_simulation=fallback_to_simulation,
            **kwargs,
        )
        if self.hardware_context.get("emulation"):
            self.hardware_available = False

    def _probe_endpoint(
        self,
        endpoint: str,
        *,
        timeout: float | None = None,
    ) -> Dict[str, Any]:
        base = endpoint.rstrip("/")
        for suffix in ("", "/status", "/health"):
            url = f"{base}{suffix}"
            try:
                with urllib_request.urlopen(url, timeout=timeout) as response:
                    raw = response.read()
            except Exception as exc:  # pragma: no cover - network diagnostics
                logger.debug("Loihi hardware probe failed for %s: %s", url, exc)
                continue
            if not raw:
                continue
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            if isinstance(decoded, dict):
                decoded.setdefault("endpoint", url)
                return decoded
        return {}

    def _initialise_hardware(
        self, config: SpikingNetworkConfig, **kwargs
    ) -> Mapping[str, Any] | None:
        runner = kwargs.pop("runner", None)
        endpoint = kwargs.get("hardware_endpoint") or kwargs.get("http_endpoint")
        probe_context: Dict[str, Any] = {}
        if endpoint and kwargs.pop("probe_endpoint", True):
            timeout_value = kwargs.get("hardware_timeout")
            try:
                timeout = float(timeout_value) if timeout_value is not None else None
            except (TypeError, ValueError):
                timeout = None
            probe_context = self._probe_endpoint(str(endpoint), timeout=timeout)
            headers = probe_context.get("headers")
            if headers and "hardware_headers" not in kwargs:
                kwargs["hardware_headers"] = headers
        if runner is not None and "run_fn" not in kwargs:
            run_candidate = getattr(runner, "run", None) or getattr(runner, "execute", None)
            if callable(run_candidate):
                kwargs["run_fn"] = run_candidate
            compile_candidate = getattr(runner, "compile", None)
            if callable(compile_candidate):
                kwargs.setdefault("compile_fn", compile_candidate)
            reset_candidate = getattr(runner, "reset", None)
            if callable(reset_candidate):
                kwargs.setdefault("reset_fn", reset_candidate)
            describe_candidate = getattr(runner, "describe", None)
            if callable(describe_candidate):
                kwargs.setdefault("describe_fn", describe_candidate)
        used_emulation = False
        if "run_fn" not in kwargs or not callable(kwargs["run_fn"]):
            describe: Dict[str, Any] = {}
            if probe_context:
                describe["probe"] = dict(probe_context)
            emulation = self._build_software_emulation(config, describe=describe or None)
            kwargs.setdefault("run_fn", emulation["run_fn"])
            kwargs.setdefault("reset_fn", emulation["reset_fn"])
            kwargs.setdefault("describe_fn", emulation["describe_fn"])
            kwargs.setdefault("compile_fn", lambda *args, **_kwargs: {})
            used_emulation = True
            self.hardware_error = HardwareIntegrationError(
                "Loihi hardware runner unavailable; using software emulation"
            )
        if "run_fn" not in kwargs or not callable(kwargs["run_fn"]):
            try:  # pragma: no cover - depends on optional SDK
                importlib.import_module("lava")
            except ImportError as exc:
                raise HardwareIntegrationError(
                    "Intel Lava SDK not available; provide a 'runner' or explicit 'run_fn'."
                ) from exc
            raise HardwareIntegrationError(
                "Loihi backend requires a runner exposing a 'run' method or a 'run_fn' argument."
            )
        context = super()._initialise_hardware(config, **kwargs) or {}
        if probe_context:
            context = dict(context)
            context.setdefault("probe", probe_context)
        if used_emulation:
            context = dict(context)
            context.setdefault("emulation", True)
        return context


class BrainScaleSHardwareBackend(CallableHardwareBackend):
    """Adapter for BrainScaleS hardware runners."""

    hardware_name = "brainscales"

    def __init__(
        self,
        config: SpikingNetworkConfig,
        *,
        auto_reset: bool = True,
        fallback_to_simulation: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            config,
            auto_reset=auto_reset,
            fallback_to_simulation=fallback_to_simulation,
            **kwargs,
        )
        if self.hardware_context.get("emulation"):
            self.hardware_available = False

    def _initialise_hardware(
        self, config: SpikingNetworkConfig, **kwargs
    ) -> Mapping[str, Any] | None:
        runner = kwargs.pop("runner", None)
        if runner is not None and "run_fn" not in kwargs:
            run_candidate = getattr(runner, "run", None) or getattr(runner, "execute", None)
            if callable(run_candidate):
                kwargs["run_fn"] = run_candidate
            compile_candidate = getattr(runner, "compile", None)
            if callable(compile_candidate):
                kwargs.setdefault("compile_fn", compile_candidate)
            reset_candidate = getattr(runner, "reset", None)
            if callable(reset_candidate):
                kwargs.setdefault("reset_fn", reset_candidate)
            describe_candidate = getattr(runner, "describe", None)
            if callable(describe_candidate):
                kwargs.setdefault("describe_fn", describe_candidate)
        used_emulation = False
        if "run_fn" not in kwargs or not callable(kwargs["run_fn"]):
            describe: Dict[str, Any] = {}
            wafer = kwargs.get("wafer") or kwargs.get("chip")
            if wafer is not None:
                describe["target"] = wafer
            emulation = self._build_software_emulation(config, describe=describe or None)
            kwargs.setdefault("run_fn", emulation["run_fn"])
            kwargs.setdefault("reset_fn", emulation["reset_fn"])
            kwargs.setdefault("describe_fn", emulation["describe_fn"])
            kwargs.setdefault("compile_fn", lambda *args, **_kwargs: {})
            used_emulation = True
            self.hardware_error = HardwareIntegrationError(
                "BrainScaleS runner unavailable; using software emulation"
            )
        if "run_fn" not in kwargs or not callable(kwargs["run_fn"]):
            raise HardwareIntegrationError(
                "BrainScaleS backend requires a runner exposing a 'run' method or a 'run_fn'."
            )
        context = super()._initialise_hardware(config, **kwargs)
        if used_emulation and isinstance(context, Mapping):
            context = dict(context)
            context.setdefault("emulation", True)
        return context


class SpiNNakerHardwareBackend(CallableHardwareBackend):
    """Adapter for SpiNNaker neuromorphic systems with optional emulation."""

    hardware_name = "spinnaker"

    def __init__(
        self,
        config: SpikingNetworkConfig,
        *,
        auto_reset: bool = True,
        fallback_to_simulation: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            config,
            auto_reset=auto_reset,
            fallback_to_simulation=fallback_to_simulation,
            **kwargs,
        )
        if self.hardware_context.get("emulation"):
            self.hardware_available = False

    def _initialise_hardware(
        self, config: SpikingNetworkConfig, **kwargs
    ) -> Mapping[str, Any] | None:
        runner = kwargs.pop("runner", None)
        if runner is not None and "run_fn" not in kwargs:
            run_candidate = getattr(runner, "run", None) or getattr(runner, "execute", None)
            if callable(run_candidate):
                kwargs["run_fn"] = run_candidate
            reset_candidate = getattr(runner, "reset", None)
            if callable(reset_candidate):
                kwargs.setdefault("reset_fn", reset_candidate)
            describe_candidate = getattr(runner, "describe", None)
            if callable(describe_candidate):
                kwargs.setdefault("describe_fn", describe_candidate)
            compile_candidate = getattr(runner, "compile", None)
            if callable(compile_candidate):
                kwargs.setdefault("compile_fn", compile_candidate)
        used_emulation = False
        if "run_fn" not in kwargs or not callable(kwargs["run_fn"]):
            describe: Dict[str, Any] = {}
            boards = kwargs.get("boards") or kwargs.get("board_count")
            if boards is not None:
                describe["boards"] = boards
            emulation = self._build_software_emulation(config, describe=describe or None)
            kwargs.setdefault("run_fn", emulation["run_fn"])
            kwargs.setdefault("reset_fn", emulation["reset_fn"])
            kwargs.setdefault("describe_fn", emulation["describe_fn"])
            kwargs.setdefault("compile_fn", lambda *args, **_kwargs: {})
            used_emulation = True
            self.hardware_error = HardwareIntegrationError(
                "SpiNNaker runner unavailable; using software emulation"
            )
        if "run_fn" not in kwargs or not callable(kwargs["run_fn"]):
            raise HardwareIntegrationError(
                "SpiNNaker backend requires a runner exposing a 'run' method or a 'run_fn'."
            )
        context = super()._initialise_hardware(config, **kwargs)
        if used_emulation and isinstance(context, Mapping):
            context = dict(context)
            context.setdefault("emulation", True)
        return context


class HardwareBackendRegistry:
    """Registry of available hardware backends."""

    _registry: Dict[str, Type[BaseHardwareBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_cls: Type[BaseHardwareBackend]) -> None:
        cls._registry[name.lower()] = backend_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseHardwareBackend] | None:
        if not name:
            return None
        return cls._registry.get(name.lower())

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._registry)


HardwareBackendRegistry.register("callable", CallableHardwareBackend)
HardwareBackendRegistry.register("external", CallableHardwareBackend)
HardwareBackendRegistry.register("hardware", CallableHardwareBackend)
HardwareBackendRegistry.register("loihi", LoihiHardwareBackend)
HardwareBackendRegistry.register("intel-loihi", LoihiHardwareBackend)
HardwareBackendRegistry.register("brainscales", BrainScaleSHardwareBackend)
HardwareBackendRegistry.register("brainscales2", BrainScaleSHardwareBackend)
HardwareBackendRegistry.register("spinnaker", SpiNNakerHardwareBackend)
HardwareBackendRegistry.register("spynnaker", SpiNNakerHardwareBackend)


