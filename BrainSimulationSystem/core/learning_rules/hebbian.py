"""
Hebbian学习规则
Hebbian Learning Rule
"""
from typing import Dict, Any

from .base import LearningRule

class HebbianLearning(LearningRule):
    """
    根据"同时激活的神经元会增强它们之间的连接"的原则调整权重
    Adjusts weights based on the principle "neurons that fire together, wire together".
    """
    
    def __init__(self, network, params: Dict[str, Any]):
        super().__init__(network, params)
    
    def update(self, state: Dict[str, Any], dt: float) -> None:
        learning_rate = self.params.get("learning_rate", 0.01)
        weight_min = self.params.get("weight_min", 0.0)
        weight_max = self.params.get("weight_max", 1.0)
        decay_rate = self.params.get("decay_rate", 0.0001)
        
        spikes = set(state.get("spikes", []))
        
        for (pre_id, post_id), synapse in self.network.synapses.items():
            if pre_id in spikes and post_id in spikes:
                dw = learning_rate
            else:
                dw = -decay_rate
            
            new_weight = synapse.weight + dw
            new_weight = max(weight_min, min(new_weight, weight_max))
            synapse.set_weight(new_weight)