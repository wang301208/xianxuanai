"""
稳态可塑性
Homeostatic Plasticity
"""
from typing import Dict, Any

from .base import LearningRule

class HomeostaticPlasticity(LearningRule):
    """
    调整神经元的内在特性，使其保持在目标活动水平
    Adjusts intrinsic properties of neurons to maintain a target activity level.
    """
    
    def __init__(self, network, params: Dict[str, Any]):
        super().__init__(network, params)
        self.activity_rates = {neuron_id: 0.0 for neuron_id in self.network.neurons}
        self.spike_counts = {neuron_id: 0 for neuron_id in self.network.neurons}
        self.time_elapsed = 0.0
    
    def update(self, state: Dict[str, Any], dt: float) -> None:
        learning_rate = self.params.get("learning_rate", 0.001)
        target_rate = self.params.get("target_rate", 0.01)
        time_window = self.params.get("time_window", 1000.0)
        
        self.time_elapsed += dt
        spikes = state.get("spikes", [])
        
        for neuron_id in spikes:
            if neuron_id in self.spike_counts:
                self.spike_counts[neuron_id] += 1
        
        if self.time_elapsed >= time_window:
            for neuron_id, count in self.spike_counts.items():
                rate = count / self.time_elapsed
                self.activity_rates[neuron_id] = rate
                
                if neuron_id in self.network.neurons:
                    neuron = self.network.neurons[neuron_id]
                    threshold_change = learning_rate * (rate - target_rate)
                    neuron.adjust_threshold(threshold_change)
            
            self.spike_counts = {neuron_id: 0 for neuron_id in self.network.neurons}
            self.time_elapsed = 0.0