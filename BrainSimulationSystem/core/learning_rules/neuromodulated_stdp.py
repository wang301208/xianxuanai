"""
三因子学习：神经调质调制的 STDP（Neuromodulated / Reward-modulated STDP）

该规则在 STDP eligibility trace 的基础上，用多巴胺等调质信号对权重更新进行门控，
以支持奖励预测误差（RPE）风格的学习。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .base import LearningRule


class NeuromodulatedSTDPLearning(LearningRule):
    """Reward/neuromodulator gated STDP with an eligibility trace."""

    def __init__(self, network: Any, params: Dict[str, Any]):
        super().__init__(network, params)
        neuron_ids = list(getattr(self.network, "neurons", {}).keys())
        self.pre_traces = {int(neuron_id): 0.0 for neuron_id in neuron_ids}
        self.post_traces = {int(neuron_id): 0.0 for neuron_id in neuron_ids}
        self.eligibility_traces: Dict[Tuple[int, int], float] = {}

    @staticmethod
    def _normalize_spikes(raw_spikes: Any) -> set[int]:
        spikes: set[int] = set()
        if raw_spikes is None:
            return spikes
        if isinstance(raw_spikes, (list, tuple, set)):
            for entry in raw_spikes:
                if isinstance(entry, dict):
                    for key in ("neuron_id", "neuron", "id", "neuron_global"):
                        if key in entry:
                            try:
                                spikes.add(int(entry[key]))
                            except Exception:
                                pass
                            break
                else:
                    try:
                        spikes.add(int(entry))
                    except Exception:
                        continue
        return spikes

    @staticmethod
    def _get_weight_and_setter(synapse: Any) -> tuple[float, Any]:
        """Return (weight, setter) where setter(new_weight) updates the synapse weight."""
        if hasattr(synapse, "set_weight") and callable(getattr(synapse, "set_weight")):
            try:
                return float(getattr(synapse, "weight")), synapse.set_weight
            except Exception:
                return 0.0, synapse.set_weight
        if hasattr(synapse, "weight"):
            def _setter(value: float) -> None:
                setattr(synapse, "weight", float(value))

            try:
                return float(getattr(synapse, "weight")), _setter
            except Exception:
                return 0.0, _setter
        if isinstance(synapse, dict):
            def _setter(value: float) -> None:
                synapse["weight"] = float(value)

            try:
                return float(synapse.get("weight", 0.0)), _setter
            except Exception:
                return 0.0, _setter
        return 0.0, lambda _value: None

    def update(self, state: Dict[str, Any], dt: float) -> None:
        learning_rate = float(self.params.get("learning_rate", 0.01))
        a_plus = float(self.params.get("a_plus", 0.01))
        a_minus = float(self.params.get("a_minus", -0.012))
        tau_plus = float(self.params.get("tau_plus", 20.0))
        tau_minus = float(self.params.get("tau_minus", 20.0))
        tau_eligibility = float(self.params.get("tau_eligibility", 200.0))
        weight_min = float(self.params.get("weight_min", 0.0))
        weight_max = float(self.params.get("weight_max", 1.0))

        neuromodulators = state.get("neuromodulators", {}) if isinstance(state, dict) else {}
        dopamine_baseline = float(self.params.get("dopamine_baseline", 0.5))
        try:
            dopamine = float(neuromodulators.get("dopamine", 0.0)) if isinstance(neuromodulators, dict) else 0.0
        except Exception:
            dopamine = 0.0
        dopamine_signal = dopamine - dopamine_baseline

        spikes = self._normalize_spikes(state.get("spikes", []) if isinstance(state, dict) else [])

        if tau_plus > 1e-9:
            pre_decay = float(np.exp(-float(dt) / tau_plus))
        else:
            pre_decay = 0.0
        if tau_minus > 1e-9:
            post_decay = float(np.exp(-float(dt) / tau_minus))
        else:
            post_decay = 0.0
        if tau_eligibility > 1e-9:
            elig_decay = float(np.exp(-float(dt) / tau_eligibility))
        else:
            elig_decay = 0.0

        # Decay traces for all known neurons.
        for neuron_id in list(self.pre_traces.keys()):
            self.pre_traces[neuron_id] *= pre_decay
            self.post_traces[neuron_id] *= post_decay
            if neuron_id in spikes:
                self.pre_traces[neuron_id] += 1.0
                self.post_traces[neuron_id] += 1.0

        # Decay eligibility traces.
        for key in list(self.eligibility_traces.keys()):
            self.eligibility_traces[key] *= elig_decay

        synapses = getattr(self.network, "synapses", {})
        if not isinstance(synapses, dict):
            return

        for syn_key, synapse in synapses.items():
            # Support both dict keys and synapse objects exposing IDs.
            pre_id = getattr(synapse, "pre_neuron_id", None)
            post_id = getattr(synapse, "post_neuron_id", None)
            if pre_id is None or post_id is None:
                if isinstance(syn_key, tuple) and len(syn_key) >= 2:
                    pre_id, post_id = syn_key[0], syn_key[1]
            try:
                pre_id_int = int(pre_id)
                post_id_int = int(post_id)
            except Exception:
                continue

            elig_key = (pre_id_int, post_id_int)
            elig = float(self.eligibility_traces.get(elig_key, 0.0))

            if pre_id_int in spikes:
                elig += learning_rate * a_minus * float(self.post_traces.get(post_id_int, 0.0))
            if post_id_int in spikes:
                elig += learning_rate * a_plus * float(self.pre_traces.get(pre_id_int, 0.0))
            self.eligibility_traces[elig_key] = elig

            if abs(dopamine_signal) < 1e-12 or abs(elig) < 1e-12:
                continue

            weight, setter = self._get_weight_and_setter(synapse)
            new_weight = float(np.clip(weight + dopamine_signal * elig, weight_min, weight_max))
            setter(new_weight)

