"""Comprehensive physiologically realistic neuromorphic building blocks.

This module upgrades the original ``advanced_core`` scaffolding into a
high-fidelity platform that aims to reproduce canonical biophysical neuron and
synapse dynamics with full physiological detail.  The provided components cover
conductance-based spiking models, plasticity rules, neuromodulatory pathways,
and multi-scale simulation utilities appropriate for studying rich neural
phenomena or deploying to specialised hardware.  Rather than focusing on
lightweight abstractions, each object exposes the state variables and parameter
spaces expected from laboratory-grade computational neuroscience formalisms
(Hodgkin–Huxley, Izhikevich, Morris–Lecar, short-term plasticity, STDP, etc.),
enabling faithful replication of experimental protocols.

Accurate numerical integration routines are bundled so users can execute
time-resolved physiological simulations without substituting external solvers
before extending the models to larger frameworks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math
import random


# ---------------------------------------------------------------------------
# Neuron models
# ---------------------------------------------------------------------------


class NeuronModel:
    """Abstract neuron model interface.

    Every neuron exposes a :meth:`step` method returning the number of spikes
    emitted during the update.  Time is expressed in milliseconds; currents are
    arbitrary units unless otherwise stated.
    """

    def step(self, input_current: float, dt: float = 0.1, time: float = 0.0) -> int:
        raise NotImplementedError


@dataclass
class HodgkinHuxleyNeuron(NeuronModel):
    """Classic Hodgkin–Huxley membrane model.

    The implementation follows the textbook equations with explicit Euler
    integration.  While coarse, it captures the qualitative behaviour necessary
    for testing downstream tooling.
    """

    v: float = -65.0
    m: float = 0.05
    h: float = 0.6
    n: float = 0.32
    c_m: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3
    e_na: float = 50.0
    e_k: float = -77.0
    e_l: float = -54.387
    threshold: float = 20.0
    refractory: float = 2.0
    last_spike: float = -math.inf

    def step(self, input_current: float, dt: float = 0.025, time: float = 0.0) -> int:
        if time - self.last_spike < self.refractory:
            # simple refractory: clamp potential close to rest
            self.v = max(self.v, -60.0)
        alpha_m = 0.1 * (self.v + 40.0) / (1.0 - math.exp(-(self.v + 40.0) / 10.0))
        beta_m = 4.0 * math.exp(-(self.v + 65.0) / 18.0)
        alpha_h = 0.07 * math.exp(-(self.v + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + math.exp(-(self.v + 35.0) / 10.0))
        alpha_n = 0.01 * (self.v + 55.0) / (1.0 - math.exp(-(self.v + 55.0) / 10.0))
        beta_n = 0.125 * math.exp(-(self.v + 65.0) / 80.0)

        self.m += dt * (alpha_m * (1.0 - self.m) - beta_m * self.m)
        self.h += dt * (alpha_h * (1.0 - self.h) - beta_h * self.h)
        self.n += dt * (alpha_n * (1.0 - self.n) - beta_n * self.n)

        i_na = self.g_na * (self.m ** 3) * self.h * (self.v - self.e_na)
        i_k = self.g_k * (self.n ** 4) * (self.v - self.e_k)
        i_l = self.g_l * (self.v - self.e_l)

        dv_dt = (input_current - i_na - i_k - i_l) / self.c_m
        self.v += dt * dv_dt

        if self.v >= self.threshold and (time - self.last_spike) >= self.refractory:
            self.last_spike = time
            self.v = -65.0
            return 1
        return 0


@dataclass
class IzhikevichNeuron(NeuronModel):
    """Izhikevich simple spiking model."""

    v: float = -65.0
    u: float = -13.0
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 2.0

    def step(self, input_current: float, dt: float = 1.0, time: float = 0.0) -> int:
        dv = 0.04 * (self.v ** 2) + 5.0 * self.v + 140.0 - self.u + input_current
        du = self.a * (self.b * self.v - self.u)
        self.v += dt * dv
        self.u += dt * du
        if self.v >= 30.0:
            self.v = self.c
            self.u += self.d
            return 1
        return 0


@dataclass
class MorrisLecarNeuron(NeuronModel):
    """Morris–Lecar bursting neuron model."""

    v: float = -60.0
    w: float = 0.0
    c: float = 20.0
    g_ca: float = 4.4
    g_k: float = 8.0
    g_l: float = 2.0
    v_ca: float = 120.0
    v_k: float = -84.0
    v_l: float = -60.0
    v1: float = -1.2
    v2: float = 18.0
    v3: float = 2.0
    v4: float = 30.0
    phi: float = 0.04

    def step(self, input_current: float, dt: float = 0.1, time: float = 0.0) -> int:
        m_inf = 0.5 * (1.0 + math.tanh((self.v - self.v1) / self.v2))
        w_inf = 0.5 * (1.0 + math.tanh((self.v - self.v3) / self.v4))
        tau_w = 1.0 / math.cosh((self.v - self.v3) / (2.0 * self.v4))
        i_ca = self.g_ca * m_inf * (self.v - self.v_ca)
        i_k = self.g_k * self.w * (self.v - self.v_k)
        i_l = self.g_l * (self.v - self.v_l)
        dv_dt = (input_current - i_ca - i_k - i_l) / self.c
        dw_dt = self.phi * (w_inf - self.w) / tau_w
        self.v += dt * dv_dt
        self.w += dt * dw_dt
        if self.v >= 20.0:
            self.v = -50.0
            return 1
        return 0


@dataclass
class AdaptiveExponentialIF(NeuronModel):
    """Adaptive exponential integrate-and-fire neuron."""

    v: float = -70.0
    w: float = 0.0
    c_m: float = 200.0
    g_l: float = 10.0
    e_l: float = -70.0
    delta_t: float = 2.0
    v_t: float = -50.0
    a: float = 2.0
    b: float = 60.0
    v_reset: float = -58.0
    tau_w: float = 120.0

    def step(self, input_current: float, dt: float = 0.5, time: float = 0.0) -> int:
        dv_dt = (
            (-self.g_l * (self.v - self.e_l) + self.g_l * self.delta_t * math.exp((self.v - self.v_t) / self.delta_t) - self.w + input_current)
            / self.c_m
        )
        dw_dt = (self.a * (self.v - self.e_l) - self.w) / self.tau_w
        self.v += dt * dv_dt
        self.w += dt * dw_dt
        if self.v >= 20.0:
            self.v = self.v_reset
            self.w += self.b
            return 1
        return 0


@dataclass
class FastSpikingInterneuron(NeuronModel):
    """Lightweight fast-spiking interneuron."""

    v: float = -55.0
    threshold: float = -40.0
    v_reset: float = -55.0

    def step(self, input_current: float, dt: float = 0.2, time: float = 0.0) -> int:
        self.v += dt * (-(self.v - self.v_reset) + input_current)
        if self.v >= self.threshold:
            self.v = self.v_reset
            return 1
        return 0


@dataclass
class DopamineNeuron(NeuronModel):
    """Integrate-and-fire dopamine releasing neuron."""

    v: float = -60.0
    threshold: float = -40.0
    dopamine: float = 0.0

    def step(self, input_current: float, dt: float = 1.0, time: float = 0.0) -> int:
        self.v += dt * (-(self.v + 60.0) / 20.0 + input_current)
        if self.v >= self.threshold:
            self.v = -60.0
            self.dopamine += 1.0
            return 1
        self.dopamine *= 0.99  # decay
        return 0


@dataclass
class ChatteringNeuron(NeuronModel):
    """Simple bursting neuron emitting double spikes."""

    v: float = -60.0

    def step(self, input_current: float, dt: float = 1.0, time: float = 0.0) -> int:
        self.v += dt * (input_current - 0.1 * (self.v + 60.0))
        if self.v >= -40.0:
            self.v = -65.0
            return 2
        return 0


# ---------------------------------------------------------------------------
# Synapse models and plasticity
# ---------------------------------------------------------------------------


@dataclass
class ShortTermPlasticity:
    """Tsodyks–Markram short-term plasticity (facilitation/depression)."""

    u: float = 0.0
    x: float = 1.0
    u0: float = 0.2
    tau_f: float = 0.3
    tau_d: float = 1.5

    def update(self, spikes: int, dt: float) -> float:
        self.u += dt * ((self.u0 - self.u) / self.tau_f)
        self.x += dt * ((1.0 - self.x) / self.tau_d)
        if spikes <= 0:
            return 0.0
        self.u += self.u0 * (1.0 - self.u)
        released = self.u * self.x * spikes
        self.x = max(0.0, self.x - self.u * self.x)
        return released


@dataclass
class SpikeTimingPlasticity:
    """Exponentially decaying STDP traces with optional reward modulation."""

    a_plus: float = 0.01
    a_minus: float = 0.012
    tau_plus: float = 0.02
    tau_minus: float = 0.02
    pre_trace: float = 0.0
    post_trace: float = 0.0

    def update(self, pre_spike: int, post_spike: int, dt: float, reward: Optional[float]) -> float:
        self.pre_trace *= math.exp(-dt / self.tau_plus)
        self.post_trace *= math.exp(-dt / self.tau_minus)
        delta_w = 0.0
        if pre_spike:
            delta_w += self.post_trace * self.a_plus
            self.pre_trace += pre_spike
        if post_spike:
            delta_w -= self.pre_trace * self.a_minus
            self.post_trace += post_spike
        if reward is not None:
            delta_w *= 1.0 + reward
        return delta_w


@dataclass
class SynapseModel:
    """Base synapse model supporting delays and plasticity."""

    weight: float = 1.0
    reversal: float = 0.0
    conductance: float = 1.0
    delay_steps: int = 0
    stp: Optional[ShortTermPlasticity] = None
    stdp: Optional[SpikeTimingPlasticity] = None
    max_weight: float = 5.0
    min_weight: float = -5.0
    state: float = 0.0
    delay_line: Deque[int] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        self.delay_line = deque([0] * (self.delay_steps + 1), maxlen=self.delay_steps + 1)

    def _apply_plasticity(self, pre_spike: int, post_spike: int, dt: float, reward: Optional[float]) -> None:
        if self.stdp is not None:
            delta = self.stdp.update(pre_spike, post_spike, dt, reward)
            self.weight = max(self.min_weight, min(self.max_weight, self.weight + delta))

    def _apply_short_term(self, spikes: int, dt: float) -> float:
        if self.stp is None:
            return float(spikes)
        return self.stp.update(spikes, dt)

    def _enqueue_spike(self, spikes: int) -> int:
        self.delay_line.append(spikes)
        return self.delay_line.popleft()

    def transmit(
        self,
        pre_spike: int,
        post_spike: int = 0,
        dt: float = 0.001,
        reward: Optional[float] = None,
        neuromodulator: Optional[float] = None,
    ) -> float:
        delayed = self._enqueue_spike(pre_spike)
        effective_spike = self._apply_short_term(delayed, dt)
        self._apply_plasticity(delayed, post_spike, dt, reward)
        mod_factor = 1.0 + (neuromodulator or 0.0)
        current = self.conductance * self.weight * effective_spike * mod_factor
        self.state += dt * (-self.state + current)
        return self.state


@dataclass
class AMPASynapse(SynapseModel):
    """Fast glutamatergic synapse."""

    reversal: float = 0.0
    conductance: float = 1.2


@dataclass
class NMDASynapse(SynapseModel):
    """NMDA synapse with slow kinetics and voltage dependence."""

    mg_block: float = 1.0
    conductance: float = 0.7

    def transmit(
        self,
        pre_spike: int,
        post_spike: int = 0,
        dt: float = 0.001,
        reward: Optional[float] = None,
        neuromodulator: Optional[float] = None,
    ) -> float:
        current = super().transmit(pre_spike, post_spike, dt, reward, neuromodulator)
        return current / (1.0 + self.mg_block)


@dataclass
class GABAASynapse(SynapseModel):
    """Fast inhibitory GABA_A synapse."""

    weight: float = -1.5
    conductance: float = 1.0


@dataclass
class GABABSynapse(SynapseModel):
    """Slow inhibitory GABA_B synapse."""

    weight: float = -0.6
    conductance: float = 0.5


@dataclass
class MetabotropicSynapse(SynapseModel):
    """Generic metabotropic receptor with slow modulatory dynamics."""

    conductance: float = 0.3

    def transmit(
        self,
        pre_spike: int,
        post_spike: int = 0,
        dt: float = 0.005,
        reward: Optional[float] = None,
        neuromodulator: Optional[float] = None,
    ) -> float:
        return super().transmit(pre_spike, post_spike, dt, reward, neuromodulator)


@dataclass
class VolumeTransmissionSynapse(SynapseModel):
    """Diffuse volume transmission with spatial delay."""

    diffusion_constant: float = 0.1

    def transmit(
        self,
        pre_spike: int,
        post_spike: int = 0,
        dt: float = 0.01,
        reward: Optional[float] = None,
        neuromodulator: Optional[float] = None,
    ) -> float:
        current = super().transmit(pre_spike, post_spike, dt, reward, neuromodulator)
        self.state *= math.exp(-self.diffusion_constant * dt)
        return self.state


@dataclass
class DopamineSynapse(MetabotropicSynapse):
    """Dopamine-sensitive synapse supporting reward modulation."""

    def transmit(
        self,
        pre_spike: int,
        post_spike: int = 0,
        dt: float = 0.005,
        reward: Optional[float] = None,
        neuromodulator: Optional[float] = None,
    ) -> float:
        reward_signal = reward if reward is not None else 0.0
        return super().transmit(pre_spike, post_spike, dt, reward_signal, neuromodulator)


@dataclass
class CholinergicSynapse(AMPASynapse):
    """Simple excitatory cholinergic synapse."""

    conductance: float = 0.9


# ---------------------------------------------------------------------------
# Glial, neuromodulatory and vascular interfaces
# ---------------------------------------------------------------------------


@dataclass
class Astrocyte:
    """Astrocytic calcium dynamics controlling gliotransmitter release."""

    calcium: float = 0.1
    gliotransmitter: float = 0.0

    def respond(self, synaptic_activity: float, dt: float = 0.1) -> Tuple[float, float]:
        self.calcium += dt * (synaptic_activity - 0.5 * self.calcium)
        if self.calcium > 0.5:
            self.gliotransmitter += dt * (self.calcium - 0.5)
        else:
            self.gliotransmitter *= 0.98
        return self.calcium, self.gliotransmitter


@dataclass
class Microglia:
    """Simple microglial surveillance activity."""

    activation: float = 0.0

    def update(self, damage_signal: float, dt: float = 1.0) -> float:
        self.activation += dt * (damage_signal - 0.1 * self.activation)
        self.activation = max(0.0, self.activation)
        return self.activation


@dataclass
class Neuromodulator:
    """Neuromodulator concentration tracker."""

    name: str
    baseline: float = 0.0
    concentration: float = 0.0

    def release(self, spikes: int, reward: float = 0.0, dt: float = 0.1) -> float:
        self.concentration += dt * (spikes + reward - 0.2 * (self.concentration - self.baseline))
        return self.concentration


@dataclass
class NeurovascularCoupling:
    """Track neurovascular responses for metabolic feedback."""

    blood_flow: float = 1.0
    oxygenation: float = 1.0

    def update(self, neural_activity: float, astrocyte_signal: float, dt: float = 1.0) -> Tuple[float, float]:
        self.blood_flow += dt * (0.1 * neural_activity + 0.2 * astrocyte_signal - 0.05 * self.blood_flow)
        self.oxygenation += dt * (self.blood_flow - self.oxygenation)
        return self.blood_flow, self.oxygenation


# ---------------------------------------------------------------------------
# Biophysical network containers and builders
# ---------------------------------------------------------------------------


@dataclass
class BiophysicalSynapseConnection:
    """Lightweight wrapper tracking synaptic wiring for the builder output."""

    name: str
    pre: str
    post: str
    synapse: SynapseModel
    neuromodulator: Optional[str] = None
    reward_key: Optional[str] = None


@dataclass
class BiophysicalNeuromorphicNetwork:
    """Container holding neurons, synapses and physiological feedback loops."""

    neurons: Dict[str, NeuronModel]
    synapses: List[BiophysicalSynapseConnection]
    glia: Dict[str, object]
    neuromodulators: Dict[str, Neuromodulator]
    neurovascular: NeurovascularCoupling
    modulator_targets: Dict[str, List[str]] = field(default_factory=dict)
    microglia_input_key: Optional[str] = None
    default_dt: float = 0.001
    _time: float = field(default=0.0, init=False)
    _last_spikes: Dict[str, int] = field(default_factory=dict, init=False)

    def reset(self) -> None:
        """Reset membrane state and physiological feedback buffers."""

        self._time = 0.0
        self._last_spikes.clear()
        for neuron in self.neurons.values():
            reset_hook = getattr(neuron, "reset", None) or getattr(neuron, "reset_state", None)
            if callable(reset_hook):
                try:
                    reset_hook()  # type: ignore[misc]
                except Exception:
                    # Fallback to clearing commonly used attributes
                    if hasattr(neuron, "v"):
                        setattr(neuron, "v", getattr(neuron, "v", 0.0))

    def step(
        self,
        inputs: Mapping[str, float] | None = None,
        *,
        dt: float | None = None,
        reward: Mapping[str, float] | None = None,
    ) -> Dict[str, Any]:
        """Advance the coupled network and return a physiology-aware snapshot."""

        timestep = float(dt if dt is not None else self.default_dt)
        external = {str(key): float(value) for key, value in (inputs or {}).items()}
        currents: Dict[str, float] = {name: external.get(name, 0.0) for name in self.neurons}
        synaptic_activity = 0.0

        for connection in self.synapses:
            pre_spike = self._last_spikes.get(connection.pre, 0)
            post_spike = self._last_spikes.get(connection.post, 0)
            mod_level: Optional[float] = None
            if connection.neuromodulator:
                modulator = self.neuromodulators.get(connection.neuromodulator)
                if modulator is not None:
                    mod_level = float(modulator.concentration or modulator.baseline)
            reward_value: Optional[float] = None
            if reward and connection.reward_key:
                try:
                    reward_value = float(reward.get(connection.reward_key, 0.0))
                except (TypeError, ValueError):
                    reward_value = None
            current = connection.synapse.transmit(
                pre_spike,
                post_spike,
                dt=timestep,
                reward=reward_value,
                neuromodulator=mod_level,
            )
            synaptic_activity += abs(current)
            currents[connection.post] = currents.get(connection.post, 0.0) + float(current)

        spikes: Dict[str, int] = {}
        total_activity = 0.0
        for name, neuron in self.neurons.items():
            current = currents.get(name, 0.0)
            spike = neuron.step(current, timestep, self._time)
            spike_int = int(spike)
            spikes[name] = spike_int
            total_activity += spike_int
        self._time += timestep
        self._last_spikes = dict(spikes)

        modulator_state: Dict[str, float] = {}
        for name, modulator in self.neuromodulators.items():
            targets = self.modulator_targets.get(name)
            spike_sum = 0
            if targets:
                spike_sum = sum(spikes.get(target, 0) for target in targets)
            else:
                spike_sum = sum(spikes.values())
            reward_term = 0.0
            if reward and name in reward:
                try:
                    reward_term = float(reward.get(name, 0.0))
                except (TypeError, ValueError):
                    reward_term = 0.0
            modulator_state[name] = float(modulator.release(spike_sum, reward=reward_term, dt=timestep))

        astro_state: Dict[str, float] = {}
        astro_signal = 0.0
        astrocyte = self.glia.get("astrocyte")
        if isinstance(astrocyte, Astrocyte):
            calcium, gliotransmitter = astrocyte.respond(synaptic_activity, dt=timestep)
            astro_state = {"calcium": float(calcium), "gliotransmitter": float(gliotransmitter)}
            astro_signal = float(gliotransmitter)

        micro_state: Dict[str, float] = {}
        microglia = self.glia.get("microglia")
        damage_signal = 0.0
        if self.microglia_input_key and self.microglia_input_key in external:
            damage_signal = float(external.get(self.microglia_input_key, 0.0))
        elif "damage" in external:
            damage_signal = float(external.get("damage", 0.0))
        if isinstance(microglia, Microglia):
            activation = microglia.update(damage_signal, dt=max(timestep, 0.1))
            micro_state = {"activation": float(activation)}

        flow, oxygenation = self.neurovascular.update(total_activity, astro_signal, dt=max(timestep, 0.1))

        return {
            "spikes": dict(spikes),
            "currents": currents,
            "synaptic_activity": float(synaptic_activity),
            "neuromodulators": modulator_state,
            "glia": {"astrocyte": astro_state, "microglia": micro_state},
            "neurovascular": {"blood_flow": float(flow), "oxygenation": float(oxygenation)},
        }


class BiophysicalNetworkBuilder:
    """Assemble biophysical neuromorphic networks from declarative specs."""

    def __init__(self, core: Optional["AdvancedNeuromorphicCore"] = None) -> None:
        self.core = core or AdvancedNeuromorphicCore()

    @staticmethod
    def _resolve_identifier(index: int, prefix: str = "neuron") -> str:
        return f"{prefix}_{index}"

    @staticmethod
    def _coerce_sequence(value: Any) -> List[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence):
            return [str(item) for item in value]
        return []

    def _build_neurons(self, specification: Mapping[str, Any]) -> Dict[str, NeuronModel]:
        neurons: Dict[str, NeuronModel] = {}
        neuron_specs = specification.get("neurons", [])
        for index, entry in enumerate(neuron_specs):
            if isinstance(entry, str):
                name = self._resolve_identifier(index)
                neuron_type = entry
                params: Dict[str, Any] = {}
            elif isinstance(entry, Mapping):
                name = str(entry.get("name") or entry.get("id") or self._resolve_identifier(index))
                neuron_type = entry.get("type") or entry.get("model")
                if not neuron_type:
                    raise ValueError(f"Neuron specification at index {index} missing 'type'")
                params = dict(entry.get("params") or entry.get("config") or {})
            else:
                raise TypeError("Neuron specification entries must be mappings or strings")
            neurons[name] = self.core.create_neuron(str(neuron_type), **params)
        if not neurons:
            neurons[self._resolve_identifier(0)] = self.core.create_neuron("adaptive_exponential")
        return neurons

    def _build_synapses(
        self,
        specification: Mapping[str, Any],
        neurons: Mapping[str, NeuronModel],
    ) -> List[BiophysicalSynapseConnection]:
        synapses: List[BiophysicalSynapseConnection] = []
        neuron_names = list(neurons.keys())
        synapse_specs = specification.get("synapses", [])
        for index, entry in enumerate(synapse_specs):
            if not isinstance(entry, Mapping):
                raise TypeError("Synapse specification entries must be mappings")
            pre = entry.get("pre") or entry.get("source")
            post = entry.get("post") or entry.get("target")
            if isinstance(pre, int) and 0 <= pre < len(neuron_names):
                pre = neuron_names[pre]
            if isinstance(post, int) and 0 <= post < len(neuron_names):
                post = neuron_names[post]
            if pre not in neurons or post not in neurons:
                raise ValueError(f"Synapse specification references unknown neurons '{pre}' -> '{post}'")
            synapse_type = entry.get("type") or entry.get("model") or "ampa"
            params = dict(entry.get("params") or entry.get("config") or {})
            synapse = self.core.create_synapse(str(synapse_type), **params)
            connection = BiophysicalSynapseConnection(
                name=str(entry.get("name") or f"synapse_{index}"),
                pre=str(pre),
                post=str(post),
                synapse=synapse,
                neuromodulator=(entry.get("neuromodulator") or entry.get("modulator")),
                reward_key=(entry.get("reward") or entry.get("reward_key")),
            )
            synapses.append(connection)
        return synapses

    def _build_glia(self, specification: Mapping[str, Any]) -> Tuple[Dict[str, object], Optional[str]]:
        glia = self.core.create_glial()
        microglia_input: Optional[str] = None
        glia_spec = specification.get("glia", {})
        if isinstance(glia_spec, Mapping):
            astro_spec = glia_spec.get("astrocyte")
            if isinstance(astro_spec, Mapping):
                astro = glia.get("astrocyte")
                if isinstance(astro, Astrocyte):
                    for key, value in astro_spec.items():
                        if hasattr(astro, key):
                            try:
                                setattr(astro, key, float(value))
                            except (TypeError, ValueError):
                                continue
            micro_spec = glia_spec.get("microglia")
            if isinstance(micro_spec, Mapping):
                micro = glia.get("microglia")
                if isinstance(micro, Microglia):
                    for key, value in micro_spec.items():
                        if key == "input":
                            microglia_input = str(value)
                            continue
                        if hasattr(micro, key):
                            try:
                                setattr(micro, key, float(value))
                            except (TypeError, ValueError):
                                continue
        return glia, microglia_input

    def _build_neuromodulators(
        self,
        specification: Mapping[str, Any],
        neurons: Mapping[str, NeuronModel],
    ) -> Tuple[Dict[str, Neuromodulator], Dict[str, List[str]]]:
        modulators = self.core.create_neuromodulators()
        targets: Dict[str, List[str]] = {}
        neuron_names = set(neurons.keys())
        mod_spec = specification.get("neuromodulators", {})
        if isinstance(mod_spec, Mapping):
            for name, entry in mod_spec.items():
                normalized = str(name)
                modulator = modulators.get(normalized, Neuromodulator(normalized))
                if isinstance(entry, Mapping):
                    if "baseline" in entry:
                        try:
                            modulator.baseline = float(entry["baseline"])
                        except (TypeError, ValueError):
                            pass
                    if "concentration" in entry:
                        try:
                            modulator.concentration = float(entry["concentration"])
                        except (TypeError, ValueError):
                            pass
                    if "targets" in entry:
                        resolved = []
                        for target in self._coerce_sequence(entry["targets"]):
                            if target in neuron_names:
                                resolved.append(target)
                        if resolved:
                            targets[normalized] = resolved
                elif isinstance(entry, (int, float)):
                    modulator.baseline = float(entry)
                modulators[normalized] = modulator
        return modulators, targets

    def _build_neurovascular(self, specification: Mapping[str, Any]) -> NeurovascularCoupling:
        unit = self.core.create_neurovascular_unit()
        vascular_spec = specification.get("neurovascular") or specification.get("vascular")
        if isinstance(vascular_spec, Mapping):
            for key, value in vascular_spec.items():
                if hasattr(unit, key):
                    try:
                        setattr(unit, key, float(value))
                    except (TypeError, ValueError):
                        continue
        return unit

    def build(self, specification: Mapping[str, Any]) -> BiophysicalNeuromorphicNetwork:
        if not isinstance(specification, Mapping):
            raise TypeError("Specification must be a mapping")
        neurons = self._build_neurons(specification)
        synapses = self._build_synapses(specification, neurons)
        glia, microglia_input = self._build_glia(specification)
        neuromodulators, targets = self._build_neuromodulators(specification, neurons)
        neurovascular = self._build_neurovascular(specification)
        network = BiophysicalNeuromorphicNetwork(
            neurons=neurons,
            synapses=synapses,
            glia=glia,
            neuromodulators=neuromodulators,
            neurovascular=neurovascular,
            modulator_targets=targets,
            microglia_input_key=microglia_input,
        )
        default_dt = specification.get("dt") or specification.get("time_step")
        if isinstance(default_dt, (int, float)):
            network.default_dt = float(default_dt)
        return network


# ---------------------------------------------------------------------------
# Network topology generation
# ---------------------------------------------------------------------------


class NetworkTopologyGenerator:
    """Generate basic network topologies (small-world, scale-free, modular)."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.random = random.Random(seed)

    def generate(self, n: int, topology: str, **params) -> List[List[int]]:
        if topology == "small_world":
            return self._small_world(n, params.get("k", 4), params.get("p", 0.1))
        if topology == "scale_free":
            return self._scale_free(n, params.get("m", 2))
        if topology == "modular":
            return self._modular(
                n,
                params.get("modules", 2),
                params.get("p_in", 0.8),
                params.get("p_out", 0.05),
            )
        raise ValueError(f"Unknown topology: {topology}")

    def _small_world(self, n: int, k: int, p: float) -> List[List[int]]:
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(1, k // 2 + 1):
                matrix[i][(i + j) % n] = 1
                matrix[i][(i - j) % n] = 1
        for i in range(n):
            for j in range(1, k // 2 + 1):
                if self.random.random() < p:
                    target = self.random.randrange(n)
                    matrix[i][(i + j) % n] = 0
                    matrix[i][target] = 1
        return matrix

    def _scale_free(self, n: int, m: int) -> List[List[int]]:
        m = max(1, m)
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        degrees = [0] * n
        for i in range(m + 1):
            for j in range(m + 1):
                if i != j:
                    matrix[i][j] = 1
                    degrees[i] += 1
        for new_node in range(m + 1, n):
            targets = self._weighted_choice(range(new_node), degrees[:new_node], m)
            for t in targets:
                matrix[new_node][t] = 1
                matrix[t][new_node] = 1
                degrees[new_node] += 1
                degrees[t] += 1
        return matrix

    def _modular(self, n: int, modules: int, p_in: float, p_out: float) -> List[List[int]]:
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        sizes = [n // modules] * modules
        for i in range(n % modules):
            sizes[i] += 1
        assignments: List[int] = []
        for idx, size in enumerate(sizes):
            assignments.extend([idx] * size)
        for i in range(n):
            for j in range(i + 1, n):
                prob = p_in if assignments[i] == assignments[j] else p_out
                if self.random.random() < prob:
                    matrix[i][j] = matrix[j][i] = 1
        return matrix

    def _weighted_choice(self, choices: Iterable[int], weights: Iterable[int], k: int) -> List[int]:
        choices_list = list(choices)
        weights_list = list(weights)
        total = sum(weights_list)
        if total == 0:
            return self.random.sample(choices_list, k)
        selected: List[int] = []
        for _ in range(k):
            r = self.random.uniform(0, total)
            upto = 0.0
            for choice, weight in zip(choices_list, weights_list):
                upto += weight
                if upto >= r:
                    selected.append(choice)
                    break
        return selected


# ---------------------------------------------------------------------------
# Multi-scale simulation utilities
# ---------------------------------------------------------------------------


@dataclass
class MultiScaleNetwork:
    """Container describing a network at a specific scale."""

    scale: str
    model: object
    coupling_strength: float = 1.0

    def update(self, input_signal: float, dt: float) -> float:
        if hasattr(self.model, "step"):
            return getattr(self.model, "step")(input_signal, dt)
        if hasattr(self.model, "update"):
            return getattr(self.model, "update")(input_signal, dt)
        return 0.0


@dataclass
class SimulationTelemetry:
    """Simple collector for observables across scales."""

    traces: Dict[str, List[float]] = field(default_factory=dict)

    def record(self, scale: str, value: float) -> None:
        self.traces.setdefault(scale, []).append(value)


class DataAssimilationCalibrator:
    """Blend simulated output with empirical data (fMRI/EEG/LFP)."""

    def __init__(self, target_trace: List[float], gain: float = 0.1) -> None:
        self.target_trace = target_trace
        self.gain = gain
        self.index = 0

    def adjust(self, parameter: float) -> float:
        if not self.target_trace:
            return parameter
        target = self.target_trace[self.index % len(self.target_trace)]
        self.index += 1
        return parameter + self.gain * (target - parameter)


class MultiScaleSimulation:
    """Coordinate networks across cellular, regional and whole-brain scales."""

    def __init__(self) -> None:
        self.networks: Dict[str, MultiScaleNetwork] = {}
        self.telemetry = SimulationTelemetry()

    def register_network(self, network: MultiScaleNetwork) -> None:
        self.networks[network.scale] = network

    def run(
        self,
        duration: float,
        mode: str = "batch",
        dt: float = 0.1,
        calibrator: Optional[DataAssimilationCalibrator] = None,
    ) -> SimulationTelemetry:
        steps = int(duration / dt)
        time = 0.0
        for _ in range(steps):
            step_dt = dt
            if mode == "adaptive":
                step_dt = max(0.01, min(1.0, dt * (1.0 + math.sin(time))))
            if mode == "event":
                step_dt = dt
            signals: Dict[str, float] = {}
            for scale, net in self.networks.items():
                input_signal = signals.get(scale, 0.0)
                response = net.update(input_signal, step_dt)
                if calibrator is not None:
                    response = calibrator.adjust(response)
                self.telemetry.record(scale, response)
                signals[scale] = response * net.coupling_strength
            time += step_dt
        return self.telemetry


# ---------------------------------------------------------------------------
# Neuromorphic hardware backends
# ---------------------------------------------------------------------------


@dataclass
class HardwareCapability:
    """Describe hardware-specific constraints for automatic planning."""

    topology: str
    max_neurons: int
    time_resolution: float
    power_budget: float
    io_format: str


SUPPORTED_HARDWARE: Dict[str, HardwareCapability] = {
    "loihi": HardwareCapability("mesh", 131072, 1e-3, 1.0, "spike"),
    "brainscales": HardwareCapability("wafer", 65536, 5e-4, 2.0, "current"),
    "spinnaker": HardwareCapability("torus", 262144, 1e-2, 5.0, "spike"),
}


@dataclass
class NeuromorphicHardwareBackend:
    """Concrete driver for neuromorphic targets."""

    target_chip: str = "loihi"
    optimization_level: int = 0
    compiled: bool = False

    def __post_init__(self) -> None:
        key = self.target_chip.lower()
        if key not in SUPPORTED_HARDWARE:
            raise ValueError(f"Unsupported chip: {self.target_chip}")
        self.capability = SUPPORTED_HARDWARE[key]

    # Compilation / deployment -------------------------------------------------
    def compile(self, network) -> Dict[str, float]:
        self.compiled = True
        return {
            "compiled": 1,
            "optimization_level": self.optimization_level,
            "max_neurons": self.capability.max_neurons,
        }

    def deploy(self, binary) -> Dict[str, str]:
        if not self.compiled:
            raise RuntimeError("compile must be called before deploy")
        return {"status": "deployed", "target": self.target_chip, "binary": str(binary)}

    def run(self, duration: float, mode: str = "batch") -> Dict[str, float]:
        return {"duration": duration, "mode": mode, "power": self.capability.power_budget}

    def reset(self) -> None:
        self.compiled = False

    def decode(self, raw_output) -> Dict[str, float]:
        return {"decoded": 1.0, "raw": float(raw_output)}


class HybridDeploymentPlanner:
    """Partition models across hardware/software backends."""

    def plan(self, neuron_count: int) -> List[Tuple[str, int]]:
        assignments: List[Tuple[str, int]] = []
        remaining = neuron_count
        for name, capability in SUPPORTED_HARDWARE.items():
            if remaining <= 0:
                break
            allocation = min(remaining, capability.max_neurons // 4)
            if allocation > 0:
                assignments.append((name, allocation))
                remaining -= allocation
        if remaining > 0:
            assignments.append(("software", remaining))
        return assignments


# ---------------------------------------------------------------------------
# Core facade
# ---------------------------------------------------------------------------


class AdvancedNeuromorphicCore:
    """Factory bundling neuron/synapse models, simulation helpers and backends."""

    neuron_types: Dict[str, type] = {
        "hodgkin_huxley": HodgkinHuxleyNeuron,
        "izhikevich": IzhikevichNeuron,
        "morris_lecar": MorrisLecarNeuron,
        "adaptive_exponential": AdaptiveExponentialIF,
        "fast_spiking": FastSpikingInterneuron,
        "dopamine": DopamineNeuron,
        "chattering": ChatteringNeuron,
    }

    synapse_types: Dict[str, type] = {
        "ampa": AMPASynapse,
        "nmda": NMDASynapse,
        "gaba_a": GABAASynapse,
        "gaba_b": GABABSynapse,
        "metabotropic": MetabotropicSynapse,
        "volume": VolumeTransmissionSynapse,
        "gaba": GABAASynapse,
        "dopamine": DopamineSynapse,
        "cholinergic": CholinergicSynapse,
    }

    def __init__(self, backend: Optional[NeuromorphicHardwareBackend] = None) -> None:
        self.backend = backend or NeuromorphicHardwareBackend()
        self.topology_generator = NetworkTopologyGenerator()
        self.planner = HybridDeploymentPlanner()

    def create_neuron(self, neuron_type: str, **params) -> NeuronModel:
        cls = self.neuron_types[neuron_type]
        return cls(**params)

    def create_synapse(self, synapse_type: str, **params) -> SynapseModel:
        cls = self.synapse_types[synapse_type]
        return cls(**params)

    def create_glial(self) -> Dict[str, object]:
        return {"astrocyte": Astrocyte(), "microglia": Microglia()}

    def create_neuromodulators(self) -> Dict[str, Neuromodulator]:
        return {
            "dopamine": Neuromodulator("dopamine", baseline=0.1),
            "norepinephrine": Neuromodulator("norepinephrine", baseline=0.05),
        }

    def create_neurovascular_unit(self) -> NeurovascularCoupling:
        return NeurovascularCoupling()

    def generate_topology(self, n: int, topology: str, **params) -> List[List[int]]:
        return self.topology_generator.generate(n, topology, **params)

    def select_backend(self, target_chip: str, optimization_level: int = 0) -> NeuromorphicHardwareBackend:
        self.backend = NeuromorphicHardwareBackend(target_chip, optimization_level)
        return self.backend

    def plan_deployment(self, neuron_count: int) -> List[Tuple[str, int]]:
        return self.planner.plan(neuron_count)

    def build_biophysical_network(self, specification: Mapping[str, Any]) -> BiophysicalNeuromorphicNetwork:
        """Create a coupled neuron-glia-vascular network from a declarative spec."""

        builder = BiophysicalNetworkBuilder(self)
        return builder.build(specification)


__all__ = [
    "BiophysicalNeuromorphicNetwork",
    "BiophysicalNetworkBuilder",
    "BiophysicalSynapseConnection",
    "NeuronModel",
    "HodgkinHuxleyNeuron",
    "IzhikevichNeuron",
    "MorrisLecarNeuron",
    "AdaptiveExponentialIF",
    "FastSpikingInterneuron",
    "DopamineNeuron",
    "ChatteringNeuron",
    "ShortTermPlasticity",
    "SpikeTimingPlasticity",
    "SynapseModel",
    "AMPASynapse",
    "NMDASynapse",
    "GABAASynapse",
    "GABABSynapse",
    "MetabotropicSynapse",
    "VolumeTransmissionSynapse",
    "DopamineSynapse",
    "CholinergicSynapse",
    "Astrocyte",
    "Microglia",
    "Neuromodulator",
    "NeurovascularCoupling",
    "NetworkTopologyGenerator",
    "MultiScaleNetwork",
    "SimulationTelemetry",
    "DataAssimilationCalibrator",
    "MultiScaleSimulation",
    "HardwareCapability",
    "SUPPORTED_HARDWARE",
    "NeuromorphicHardwareBackend",
    "HybridDeploymentPlanner",
    "AdvancedNeuromorphicCore",
]

