"""
BCM (Bienenstock-Cooper-Munro) 学习规则
BCM (Bienenstock-Cooper-Munro) Learning Rule
"""
import numpy as np
from typing import Dict, Any

from .base import LearningRule

class BCMLearning(LearningRule):
    """
    一种基于后突触神经元活动的学习规则，具有稳定性和竞争性
    A learning rule based on postsynaptic activity, featuring stability and competition.
    """
    
    def __init__(self, network, params: Dict[str, Any]):
        super().__init__(network, params)
        self.activity = {neuron_id: 0.0 for neuron_id in self.network.neurons}
        self.thresholds = {neuron_id: self.params.get("target_rate", 0.1) 
                          for neuron_id in self.network.neurons}
    
    def update(self, state: Dict[str, Any], dt: float) -> None:
        learning_rate = self.params.get("learning_rate", 0.01)
        sliding_tau = self.params.get("sliding_threshold_tau", 1000.0)
        weight_min = self.params.get("weight_min", 0.0)
        weight_max = self.params.get("weight_max", 1.0)
        
        spikes = set(state.get("spikes", []))
        
        for neuron_id in self.activity:
            tau_activity = 100.0
            decay = np.exp(-dt / tau_activity)
            if neuron_id in spikes:
                self.activity[neuron_id] = decay * self.activity[neuron_id] + (1.0 - decay)
            else:
                self.activity[neuron_id] *= decay
            
            threshold_decay = np.exp(-dt / sliding_tau)
            self.thresholds[neuron_id] = threshold_decay * self.thresholds[neuron_id] + \
                                        (1.0 - threshold_decay) * (self.activity[neuron_id]**2)
        
        for (pre_id, post_id), synapse in self.network.synapses.items():
            pre_activity = 1.0 if pre_id in spikes else 0.0
            post_activity = self.activity[post_id]
            post_threshold = self.thresholds[post_id]
            
            dw = learning_rate * post_activity * (post_activity - post_threshold) * pre_activity
            
            new_weight = synapse.weight + dw
            new_weight = max(weight_min, min(new_weight, weight_max))
            synapse.set_weight(new_weight)