"""
尖峰时间依赖可塑性（STDP）学习规则
Spike-Timing-Dependent Plasticity (STDP) Learning Rule
"""
import numpy as np
from typing import Dict, Any

from .base import LearningRule

class STDPLearning(LearningRule):
    """
    根据前后神经元的脉冲时间差调整突触权重
    Adjusts synaptic weights based on the time difference between pre- and post-synaptic spikes.
    """
    
    def __init__(self, network, params: Dict[str, Any]):
        super().__init__(network, params)
        self.pre_traces = {neuron_id: 0.0 for neuron_id in self.network.neurons}
        self.post_traces = {neuron_id: 0.0 for neuron_id in self.network.neurons}
    
    def update(self, state: Dict[str, Any], dt: float) -> None:
        learning_rate = self.params.get("learning_rate", 0.01)
        a_plus = self.params.get("a_plus", 0.1)
        a_minus = self.params.get("a_minus", -0.1)
        tau_plus = self.params.get("tau_plus", 20.0)
        tau_minus = self.params.get("tau_minus", 20.0)
        weight_min = self.params.get("weight_min", 0.0)
        weight_max = self.params.get("weight_max", 1.0)
        
        spikes = state.get("spikes", [])
        
        for neuron_id in self.pre_traces:
            self.pre_traces[neuron_id] *= np.exp(-dt / tau_plus)
            self.post_traces[neuron_id] *= np.exp(-dt / tau_minus)
            if neuron_id in spikes:
                self.pre_traces[neuron_id] += 1.0
                self.post_traces[neuron_id] += 1.0
        
        for (pre_id, post_id), synapse in self.network.synapses.items():
            dw = 0.0
            if pre_id in spikes:
                dw += learning_rate * a_minus * self.post_traces[post_id]
            if post_id in spikes:
                dw += learning_rate * a_plus * self.pre_traces[pre_id]
            
            if dw != 0:
                new_weight = synapse.weight + dw
                new_weight = max(weight_min, min(new_weight, weight_max))
                synapse.set_weight(new_weight)