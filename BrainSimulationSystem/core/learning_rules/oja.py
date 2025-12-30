"""
Oja学习规则
Oja's Learning Rule
"""
from typing import Dict, Any

from .base import LearningRule

class OjaLearning(LearningRule):
    """
    Hebbian学习的一种变体，具有自我正则化特性
    A variant of Hebbian learning with a self-regularizing property.
    """
    
    def __init__(self, network, params: Dict[str, Any]):
        super().__init__(network, params)
    
    def update(self, state: Dict[str, Any], dt: float) -> None:
        learning_rate = self.params.get("learning_rate", 0.01)
        weight_min = self.params.get("weight_min", 0.0)
        weight_max = self.params.get("weight_max", 1.0)
        
        spikes = set(state.get("spikes", []))
        voltages = state.get("voltages", {})
        
        for (pre_id, post_id), synapse in self.network.synapses.items():
            pre_activity = 1.0 if pre_id in spikes else 0.0
            post_activity = voltages.get(post_id, 0.0)
            
            dw = learning_rate * post_activity * (pre_activity - post_activity * synapse.weight)
            
            new_weight = synapse.weight + dw
            new_weight = max(weight_min, min(new_weight, weight_max))
            synapse.set_weight(new_weight)