"""
Detailed synapse model that combines STP, STDP, metaplasticity, and stability controls.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from .synapse_parameters import SynapseParameters

if TYPE_CHECKING:
    from .detailed_neuron import DetailedNeuron


class DetailedSynapse:
    """Detailed synapse model that supports STP, STDP, normalization, and pruning."""

    def __init__(
        self,
        pre_neuron: "DetailedNeuron",
        post_neuron: "DetailedNeuron",
        synapse_params: SynapseParameters,
    ):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.params = synapse_params

        # Base synapse state
        self.weight = float(synapse_params.weight)
        self.delay = float(synapse_params.delay)

        # Short-term plasticity state
        self.u = float(synapse_params.U)  # utilization
        self.x = 1.0  # available resources
        self.y = 0.0  # active resources

        # Long-term plasticity (STDP traces and bookkeeping)
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self._last_trace_update_time = 0.0
        self._last_plasticity_update = 0.0
        self.homeostatic_trace = 0.0
        self.last_pre_spike_time = -np.inf
        self.last_post_spike_time = -np.inf
        self.last_activation_time = -np.inf
        self._processed_pre_spikes = len(getattr(self.pre_neuron, "spike_times", []))
        self._processed_post_spikes = len(getattr(self.post_neuron, "spike_times", []))

        # STDP parameters (fall back to defaults if not provided)
        self.stdp_tau_plus = float(
            getattr(synapse_params, "tau_plus", getattr(synapse_params, "stdp_tau_plus", 20.0))
        )
        self.stdp_tau_minus = float(
            getattr(synapse_params, "tau_minus", getattr(synapse_params, "stdp_tau_minus", 20.0))
        )
        self.stdp_A_plus = float(
            getattr(synapse_params, "A_plus", getattr(synapse_params, "stdp_A_plus", 0.01))
        )
        self.stdp_A_minus = float(
            getattr(synapse_params, "A_minus", getattr(synapse_params, "stdp_A_minus", 0.012))
        )
        self.weight_min = float(getattr(synapse_params, "weight_min", 0.0))
        self.weight_max = float(getattr(synapse_params, "weight_max", 10.0))
        self._stdp_enabled = bool(self.params.ltp_enabled or self.params.ltd_enabled)

        # Stability & metaplasticity controls
        self.normalization_target = getattr(synapse_params, "weight_normalization_target", None)
        self.normalization_interval = float(
            getattr(synapse_params, "weight_normalization_interval", 50.0)
        )
        self.normalization_rate = float(
            max(0.0, getattr(synapse_params, "weight_normalization_rate", 0.1))
        )
        self.inactivity_threshold = float(getattr(synapse_params, "inactivity_threshold", 2000.0))
        self.pruning_rate = float(max(0.0, getattr(synapse_params, "pruning_rate", 0.0)))
        self.metaplasticity_tau = float(max(0.0, getattr(synapse_params, "metaplasticity_tau", 500.0)))
        self.metaplasticity_target = float(getattr(synapse_params, "metaplasticity_target", 0.2))
        self.metaplasticity_beta = float(max(0.0, getattr(synapse_params, "metaplasticity_beta", 0.05)))

        # Neurotransmitter and receptor setup
        self.neurotransmitter = synapse_params.neurotransmitter
        self.receptors = synapse_params.receptor_types

        # Propagation queue for delayed spikes
        self.spike_queue: List[float] = []

    def process_spike(self, spike_time: float) -> float:
        """Handle a presynaptic spike event."""
        self.spike_queue.append(spike_time + self.delay)

        if self.params.stp_enabled:
            self.u += self.params.U * (1 - self.u)
            self.y = self.u * self.x
            self.x -= self.y

        self.last_pre_spike_time = spike_time
        self.last_activation_time = spike_time

        return self.weight * self.y if self.params.stp_enabled else self.weight

    def update(self, dt: float, current_time: float) -> float:
        """Update synapse state and return the postsynaptic current."""
        synaptic_current = 0.0
        ready_spikes: List[float] = []

        for spike_time in self.spike_queue:
            if current_time >= spike_time:
                synaptic_current += self._generate_postsynaptic_current()
                ready_spikes.append(spike_time)

        if ready_spikes:
            for spike_time in ready_spikes:
                self.spike_queue.remove(spike_time)

        if self.params.stp_enabled:
            self.x += (1 - self.x) / self.params.tau_rec * dt
            self.u *= np.exp(-dt / max(self.params.tau_fac, 1e-6))

        self._update_long_term_plasticity(current_time)

        return synaptic_current

    def _generate_postsynaptic_current(self) -> float:
        """Compute the postsynaptic current based on neurotransmitter types."""
        if self.neurotransmitter == "glutamate":
            ampa_current = self.receptors.get("ampa", 0.0) * self.weight * 0.8
            nmda_current = self.receptors.get("nmda", 0.0) * self.weight * 0.2
            return ampa_current + nmda_current

        if self.neurotransmitter == "gaba":
            gaba_a_current = self.receptors.get("gaba_a", 0.0) * self.weight * 0.9
            gaba_b_current = self.receptors.get("gaba_b", 0.0) * self.weight * 0.1
            return -(gaba_a_current + gaba_b_current)

        return self.weight

    def _clip_weight(self) -> None:
        """Keep weight inside configured bounds."""
        self.weight = float(np.clip(self.weight, self.weight_min, self.weight_max))

    def _decay_stdp_traces(self, target_time: float) -> None:
        """Decay STDP and metaplasticity traces up to the target time."""
        if target_time <= self._last_trace_update_time:
            return

        delta_t = target_time - self._last_trace_update_time
        if delta_t <= 0.0:
            return

        self.pre_trace *= np.exp(-delta_t / max(self.stdp_tau_plus, 1e-6))
        self.post_trace *= np.exp(-delta_t / max(self.stdp_tau_minus, 1e-6))
        if self.metaplasticity_tau > 0.0:
            self.homeostatic_trace *= np.exp(-delta_t / max(self.metaplasticity_tau, 1e-6))
        self._last_trace_update_time = target_time

    def _collect_pending_spikes(self, current_time: float) -> Tuple[List[Tuple[float, str]], int, int]:
        """Collect pre/post spikes that occurred up to current_time."""
        events: List[Tuple[float, str]] = []

        pre_spike_times = getattr(self.pre_neuron, "spike_times", [])
        post_spike_times = getattr(self.post_neuron, "spike_times", [])

        pre_idx = self._processed_pre_spikes
        while pre_idx < len(pre_spike_times) and pre_spike_times[pre_idx] <= current_time:
            events.append((pre_spike_times[pre_idx], "pre"))
            pre_idx += 1

        post_idx = self._processed_post_spikes
        while post_idx < len(post_spike_times) and post_spike_times[post_idx] <= current_time:
            events.append((post_spike_times[post_idx], "post"))
            post_idx += 1

        return events, pre_idx, post_idx

    def _apply_metaplasticity(self, delta_w: float, is_ltp: bool) -> float:
        """Scale weight change based on long-term activity history."""
        if self.metaplasticity_tau <= 0.0 or delta_w == 0.0:
            return delta_w

        activity_error = float(np.clip(self.homeostatic_trace - self.metaplasticity_target, -5.0, 5.0))
        beta = self.metaplasticity_beta
        if is_ltp:
            factor = float(np.exp(-beta * activity_error))
        else:
            factor = float(np.exp(beta * activity_error))
        return delta_w * factor

    def _apply_synaptic_pruning(self, dt_step: float, current_time: float) -> None:
        """Gently decay synapses that remain inactive for prolonged periods."""
        if self.pruning_rate <= 0.0 or self.inactivity_threshold <= 0.0:
            return

        last_activity = max(self.last_activation_time, self.last_pre_spike_time, self.last_post_spike_time)
        if not np.isfinite(last_activity):
            return

        if current_time - last_activity < self.inactivity_threshold:
            return

        decay_factor = float(np.exp(-self.pruning_rate * max(dt_step, 1e-6)))
        self.weight *= decay_factor
        if abs(self.weight) < 1e-6:
            self.weight = 0.0
        self._clip_weight()

    def _apply_homeostatic_normalization(self, current_time: float) -> None:
        """Normalize incoming excitatory weights to keep postsynaptic drive bounded."""
        if self.normalization_target is None or self.normalization_interval <= 0.0:
            return

        input_synapses = getattr(self.post_neuron, "input_synapses", None)
        if not input_synapses:
            return

        last_norm = getattr(self.post_neuron, "_last_weight_normalization", -np.inf)
        if current_time - last_norm < self.normalization_interval:
            return

        excitatory_synapses = [syn for syn in input_synapses if getattr(syn, "weight", 0.0) > 0.0]
        if not excitatory_synapses:
            setattr(self.post_neuron, "_last_weight_normalization", current_time)
            return

        total_exc_weight = sum(float(getattr(syn, "weight", 0.0)) for syn in excitatory_synapses)
        if total_exc_weight <= 0.0:
            setattr(self.post_neuron, "_last_weight_normalization", current_time)
            return

        desired_scale = self.normalization_target / total_exc_weight
        limited_adjustment = float(
            np.clip(desired_scale - 1.0, -self.normalization_rate, self.normalization_rate)
        )
        scale = 1.0 + limited_adjustment

        for syn in excitatory_synapses:
            current_weight = float(getattr(syn, "weight", 0.0))
            min_w = float(getattr(syn, "weight_min", -np.inf))
            max_w = float(getattr(syn, "weight_max", np.inf))
            syn.weight = float(np.clip(current_weight * scale, min_w, max_w))

        setattr(self.post_neuron, "_last_weight_normalization", current_time)

    def _update_long_term_plasticity(self, current_time: float) -> None:
        """Apply STDP updates and invoke stability mechanisms."""
        events, pre_idx, post_idx = self._collect_pending_spikes(current_time)
        self._processed_pre_spikes = pre_idx
        self._processed_post_spikes = post_idx

        if events:
            events.sort(key=lambda item: (item[0], 0 if item[1] == "pre" else 1))
            for spike_time, event_type in events:
                if self._stdp_enabled:
                    self._decay_stdp_traces(spike_time)

                if event_type == "pre":
                    delta_w = 0.0
                    if self._stdp_enabled and self.params.ltd_enabled:
                        delta_w = -self.stdp_A_minus * self.post_trace
                        delta_w = self._apply_metaplasticity(delta_w, is_ltp=False)
                        if delta_w != 0.0:
                            self.weight += self.params.learning_rate * delta_w
                            self._clip_weight()

                    if self._stdp_enabled:
                        self.pre_trace += 1.0
                    self.last_pre_spike_time = spike_time
                    self.last_activation_time = spike_time
                else:
                    delta_w = 0.0
                    if self._stdp_enabled and self.params.ltp_enabled:
                        delta_w = self.stdp_A_plus * self.pre_trace
                        delta_w = self._apply_metaplasticity(delta_w, is_ltp=True)
                        if delta_w != 0.0:
                            self.weight += self.params.learning_rate * delta_w
                            self._clip_weight()

                    if self._stdp_enabled:
                        self.post_trace += 1.0
                    self.homeostatic_trace += 1.0
                    self.last_post_spike_time = spike_time
                    self.last_activation_time = max(self.last_activation_time, spike_time)

        # Ensure traces decay up to the current time
        self._decay_stdp_traces(current_time)

        dt_step = max(0.0, current_time - self._last_plasticity_update)
        self._last_plasticity_update = current_time

        self._apply_synaptic_pruning(dt_step, current_time)
        self._apply_homeostatic_normalization(current_time)
