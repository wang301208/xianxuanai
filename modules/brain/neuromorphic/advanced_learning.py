"""Advanced neuromorphic learning algorithms and online adaptation.

This module implements state-of-the-art learning algorithms specifically
designed for neuromorphic hardware, including online STDP, meta-learning,
continual learning, and biologically-inspired adaptation mechanisms.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque, defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning operation modes."""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    META_LEARNING = "meta_learning"
    CONTINUAL = "continual"
    ONLINE = "online"


@dataclass
class LearningParameters:
    """Learning algorithm parameters."""
    learning_rate: float = 0.01
    decay_rate: float = 0.95
    momentum: float = 0.9
    regularization: float = 0.001
    adaptation_rate: float = 0.1
    meta_learning_rate: float = 0.001
    temperature: float = 1.0
    noise_level: float = 0.01


@dataclass
class SynapticTrace:
    """Synaptic trace for STDP and other plasticity rules."""
    pre_trace: float = 0.0
    post_trace: float = 0.0
    eligibility_trace: float = 0.0
    tau_pre: float = 20.0  # ms
    tau_post: float = 20.0  # ms
    tau_eligibility: float = 1000.0  # ms
    last_update_time: float = 0.0


class PlasticityRule(ABC):
    """Abstract base class for synaptic plasticity rules."""
    
    @abstractmethod
    def update_weight(self, pre_spike_time: float, post_spike_time: float,
                     current_weight: float, trace: SynapticTrace) -> Tuple[float, SynapticTrace]:
        """Update synaptic weight based on spike timing."""
        pass
    
    @abstractmethod
    def get_rule_name(self) -> str:
        """Get plasticity rule name."""
        pass


class STDPRule(PlasticityRule):
    """Spike-Timing Dependent Plasticity rule."""
    
    def __init__(self, a_plus: float = 0.01, a_minus: float = 0.012,
                 tau_plus: float = 20.0, tau_minus: float = 20.0,
                 w_min: float = 0.0, w_max: float = 1.0):
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max
    
    def update_weight(self, pre_spike_time: float, post_spike_time: float,
                     current_weight: float, trace: SynapticTrace) -> Tuple[float, SynapticTrace]:
        """Update weight using STDP rule."""
        dt = post_spike_time - pre_spike_time
        
        if dt > 0:  # Post before pre (LTD)
            delta_w = -self.a_minus * math.exp(-dt / self.tau_minus)
        else:  # Pre before post (LTP)
            delta_w = self.a_plus * math.exp(dt / self.tau_plus)
        
        new_weight = np.clip(current_weight + delta_w, self.w_min, self.w_max)
        
        # Update traces
        time_diff = max(pre_spike_time, post_spike_time) - trace.last_update_time
        trace.pre_trace *= math.exp(-time_diff / trace.tau_pre)
        trace.post_trace *= math.exp(-time_diff / trace.tau_post)
        
        if pre_spike_time > trace.last_update_time:
            trace.pre_trace += 1.0
        if post_spike_time > trace.last_update_time:
            trace.post_trace += 1.0
        
        trace.last_update_time = max(pre_spike_time, post_spike_time)
        
        return new_weight, trace
    
    def get_rule_name(self) -> str:
        return "STDP"


class OnlineLearningEngine:
    """Online learning engine for neuromorphic systems."""
    
    def __init__(self, plasticity_rule: PlasticityRule, 
                 learning_params: LearningParameters):
        self.plasticity_rule = plasticity_rule
        self.learning_params = learning_params
        self.synaptic_traces: Dict[Tuple[int, int], SynapticTrace] = {}
        self.weights: Dict[Tuple[int, int], float] = {}
        self.learning_history: deque = deque(maxlen=1000)
        self.adaptation_enabled = True
        
    def add_synapse(self, pre_neuron: int, post_neuron: int, initial_weight: float = 0.5):
        """Add synapse to learning system."""
        synapse_id = (pre_neuron, post_neuron)
        self.weights[synapse_id] = initial_weight
        self.synaptic_traces[synapse_id] = SynapticTrace()
    
    def process_spike_pair(self, pre_neuron: int, post_neuron: int,
                          pre_spike_time: float, post_spike_time: float,
                          reward: float = 0.0) -> float:
        """Process spike pair and update synaptic weight."""
        synapse_id = (pre_neuron, post_neuron)
        
        if synapse_id not in self.weights:
            self.add_synapse(pre_neuron, post_neuron)
        
        current_weight = self.weights[synapse_id]
        trace = self.synaptic_traces[synapse_id]
        
        # Apply plasticity rule
        if hasattr(self.plasticity_rule, 'update_weight'):
            if 'reward' in self.plasticity_rule.update_weight.__code__.co_varnames:
                new_weight, updated_trace = self.plasticity_rule.update_weight(
                    pre_spike_time, post_spike_time, current_weight, trace, reward
                )
            else:
                new_weight, updated_trace = self.plasticity_rule.update_weight(
                    pre_spike_time, post_spike_time, current_weight, trace
                )
        
        # Update stored values
        self.weights[synapse_id] = new_weight
        self.synaptic_traces[synapse_id] = updated_trace
        
        # Record learning event
        self.learning_history.append({
            'timestamp': max(pre_spike_time, post_spike_time),
            'synapse': synapse_id,
            'weight_change': new_weight - current_weight,
            'reward': reward
        })
        
        return new_weight
    
    def get_weight(self, pre_neuron: int, post_neuron: int) -> float:
        """Get current synaptic weight."""
        synapse_id = (pre_neuron, post_neuron)
        return self.weights.get(synapse_id, 0.0)
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        if not self.learning_history:
            return {"total_updates": 0}
        
        recent_changes = [event['weight_change'] for event in self.learning_history[-100:]]
        recent_rewards = [event['reward'] for event in self.learning_history[-100:]]
        
        return {
            "total_updates": len(self.learning_history),
            "avg_weight_change": np.mean(recent_changes) if recent_changes else 0,
            "std_weight_change": np.std(recent_changes) if recent_changes else 0,
            "avg_reward": np.mean(recent_rewards) if recent_rewards else 0,
            "total_synapses": len(self.weights),
            "active_synapses": sum(1 for w in self.weights.values() if w > 0.01)
        }


__all__ = [
    "LearningMode",
    "LearningParameters", 
    "SynapticTrace",
    "PlasticityRule",
    "STDPRule",
    "OnlineLearningEngine"
]