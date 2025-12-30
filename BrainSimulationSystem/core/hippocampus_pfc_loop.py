"""
Hippocampus-Prefrontal Cortex Memory Loop Implementation

This module implements the CA3→CA1→PFC memory circuit with:
- Pattern separation and completion mechanisms in CA3
- Temporal sequence processing in CA1  
- Working memory and executive control in PFC
- Reinforcement learning reward signal integration
- Nengo and BindsNET backend mappings for validation
"""

from __future__ import annotations

import numpy as np
import random
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

from .memory_system import HippocampalFormation, CorticalMemory
from .network import NeuralNetwork
from .neurons import Neuron, create_neuron

try:
    import nengo
    NENGO_AVAILABLE = True
except ImportError:
    NENGO_AVAILABLE = False
    nengo = None

try:
    # BindsNET is a hypothetical binding for PyTorch-based spiking networks
    import bindsnet
    BINDSNET_AVAILABLE = True
except ImportError:
    BINDSNET_AVAILABLE = False
    bindsnet = None


class MemoryPhase(Enum):
    """Memory processing phases."""
    ENCODING = "encoding"
    CONSOLIDATION = "consolidation"
    RETRIEVAL = "retrieval"
    REPLAY = "replay"


@dataclass
class MemoryTrace:
    """Enhanced memory trace with temporal and contextual information."""
    
    # Core content
    pattern: np.ndarray
    context: np.ndarray
    timestamp: float
    
    # CA3 representations
    ca3_sparse_code: np.ndarray
    ca3_completion_strength: float = 0.0
    
    # CA1 representations  
    ca1_sequence_code: np.ndarray = field(default_factory=lambda: np.array([]))
    ca1_temporal_context: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # PFC representations
    pfc_working_memory: np.ndarray = field(default_factory=lambda: np.array([]))
    pfc_executive_tags: List[str] = field(default_factory=list)
    
    # Learning signals
    reward_signal: float = 0.0
    prediction_error: float = 0.0
    novelty_signal: float = 0.0
    
    # State tracking
    consolidation_level: float = 0.0
    retrieval_count: int = 0
    last_accessed: float = 0.0
    
    # Associations
    associated_traces: List[int] = field(default_factory=list)
    causal_predecessors: List[int] = field(default_factory=list)
    causal_successors: List[int] = field(default_factory=list)


class CA3Network:
    """CA3 recurrent network implementing pattern separation and completion."""
    
    def __init__(self, size: int = 1000, sparsity: float = 0.02):
        self.size = size
        self.sparsity = sparsity
        
        # Recurrent connectivity matrix
        self.recurrent_weights = self._initialize_recurrent_weights()
        
        # Pattern separation parameters
        self.separation_threshold = 0.3
        self.completion_threshold = 0.6
        
        # Plasticity parameters
        self.learning_rate = 0.01
        self.homeostatic_scaling = 0.999
        
        # State variables
        self.activity = np.zeros(size)
        self.stored_patterns = []
        self.pattern_indices = {}
        
    def _initialize_recurrent_weights(self) -> np.ndarray:
        """Initialize sparse recurrent connectivity."""
        weights = np.zeros((self.size, self.size))
        
        # Create sparse random connectivity
        for i in range(self.size):
            # Each neuron connects to ~10% of others
            n_connections = int(self.size * 0.1)
            targets = random.sample(range(self.size), n_connections)
            
            for j in targets:
                if i != j:  # No self-connections
                    weights[i, j] = random.gauss(0.0, 0.1)
        
        return weights
    
    def encode_pattern(self, input_pattern: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Encode new pattern with pattern separation."""
        # Combine input and context
        combined_input = np.concatenate([input_pattern, context])
        if len(combined_input) > self.size:
            combined_input = combined_input[:self.size]
        elif len(combined_input) < self.size:
            combined_input = np.pad(combined_input, (0, self.size - len(combined_input)))
        
        # Check similarity to existing patterns
        max_similarity = 0.0
        most_similar_idx = -1
        
        for idx, stored_pattern in enumerate(self.stored_patterns):
            similarity = np.dot(combined_input, stored_pattern) / (
                np.linalg.norm(combined_input) * np.linalg.norm(stored_pattern) + 1e-8)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_idx = idx
        
        # Pattern separation: create new representation if sufficiently different
        if max_similarity < self.separation_threshold:
            # Create sparse representation
            sparse_code = self._create_sparse_code(combined_input)
            self.stored_patterns.append(sparse_code)
            pattern_idx = len(self.stored_patterns) - 1
            
            # Store pattern with Hebbian learning
            self._hebbian_update(sparse_code)
            
        else:
            # Pattern completion: retrieve similar pattern
            sparse_code = self.stored_patterns[most_similar_idx]
            pattern_idx = most_similar_idx
        
        self.activity = sparse_code
        return sparse_code
    
    def _create_sparse_code(self, input_pattern: np.ndarray) -> np.ndarray:
        """Create sparse distributed representation."""
        # Winner-take-all with sparsity constraint
        n_active = int(self.size * self.sparsity)
        
        # Add noise and compute activation
        noisy_input = input_pattern + np.random.normal(0, 0.1, self.size)
        activation = np.tanh(noisy_input)
        
        # Select top-k neurons
        top_indices = np.argpartition(activation, -n_active)[-n_active:]
        sparse_code = np.zeros(self.size)
        sparse_code[top_indices] = activation[top_indices]
        
        return sparse_code
    
    def _hebbian_update(self, pattern: np.ndarray):
        """Update recurrent weights with Hebbian learning."""
        # Outer product for Hebbian update
        hebbian_update = np.outer(pattern, pattern) * self.learning_rate
        
        # Apply update with homeostatic scaling
        self.recurrent_weights += hebbian_update
        self.recurrent_weights *= self.homeostatic_scaling
        
        # Maintain sparsity
        self.recurrent_weights = np.clip(self.recurrent_weights, -1.0, 1.0)
    
    def retrieve_pattern(self, cue: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        """Retrieve pattern using recurrent dynamics."""
        # Initialize with cue
        if len(cue) > self.size:
            cue = cue[:self.size]
        elif len(cue) < self.size:
            cue = np.pad(cue, (0, self.size - len(cue)))
        
        activity = cue.copy()
        
        # Iterative retrieval
        for _ in range(max_iterations):
            new_activity = np.tanh(self.recurrent_weights @ activity)
            
            # Check convergence
            if np.linalg.norm(new_activity - activity) < 0.01:
                break
            
            activity = new_activity
        
        self.activity = activity
        return activity


class CA1Network:
    """CA1 network for temporal sequence processing and context binding."""
    
    def __init__(self, size: int = 800):
        self.size = size
        
        # Feedforward weights from CA3
        self.ca3_weights = np.random.normal(0, 0.1, (size, 1000))  # Assuming CA3 size = 1000
        
        # Temporal context weights
        self.temporal_weights = np.random.normal(0, 0.05, (size, size))
        
        # State variables
        self.activity = np.zeros(size)
        self.temporal_context = np.zeros(size)
        self.sequence_buffer = deque(maxlen=10)
        
        # Learning parameters
        self.learning_rate = 0.005
        self.temporal_decay = 0.9
        
    def process_ca3_input(self, ca3_activity: np.ndarray) -> np.ndarray:
        """Process input from CA3 with temporal context integration."""
        # Feedforward processing
        ff_input = self.ca3_weights @ ca3_activity
        
        # Temporal context integration
        temporal_input = self.temporal_weights @ self.temporal_context
        
        # Combined activation
        combined_input = ff_input + 0.3 * temporal_input
        new_activity = np.tanh(combined_input)
        
        # Update temporal context
        self.temporal_context = (self.temporal_decay * self.temporal_context + 
                               (1 - self.temporal_decay) * self.activity)
        
        # Store in sequence buffer
        self.sequence_buffer.append(new_activity.copy())
        
        self.activity = new_activity
        return new_activity
    
    def encode_sequence(self, ca3_sequence: List[np.ndarray]) -> np.ndarray:
        """Encode temporal sequence from CA3."""
        sequence_representation = np.zeros(self.size)
        
        # Reset temporal context
        self.temporal_context = np.zeros(self.size)
        
        # Process sequence
        for ca3_pattern in ca3_sequence:
            ca1_activity = self.process_ca3_input(ca3_pattern)
            sequence_representation += ca1_activity * (1.0 / len(ca3_sequence))
        
        return sequence_representation
    
    def predict_next(self, current_ca3: np.ndarray) -> np.ndarray:
        """Predict next pattern in sequence."""
        current_ca1 = self.process_ca3_input(current_ca3)
        
        # Use temporal context to predict
        prediction = self.temporal_weights @ current_ca1
        return np.tanh(prediction)


class PFCNetwork:
    """Prefrontal cortex network for working memory and executive control."""
    
    def __init__(self, size: int = 500):
        self.size = size
        
        # Working memory modules
        self.working_memory = np.zeros(size)
        self.attention_weights = np.ones(size)
        self.executive_control = np.zeros(size)
        
        # Input weights from CA1
        self.ca1_weights = np.random.normal(0, 0.1, (size, 800))  # Assuming CA1 size = 800
        
        # Recurrent weights for maintenance
        self.recurrent_weights = np.random.normal(0, 0.05, (size, size))
        
        # Executive control parameters
        self.control_threshold = 0.5
        self.maintenance_strength = 0.8
        
        # Reward prediction
        self.reward_predictor = np.random.normal(0, 0.1, size)
        self.value_estimate = 0.0
        
    def update_working_memory(self, ca1_input: np.ndarray, 
                            executive_signal: Optional[np.ndarray] = None) -> np.ndarray:
        """Update working memory with CA1 input and executive control."""
        # Process CA1 input
        ca1_processed = self.ca1_weights @ ca1_input
        
        # Apply attention weighting
        attended_input = ca1_processed * self.attention_weights
        
        # Executive control modulation
        if executive_signal is not None:
            control_modulation = executive_signal
        else:
            control_modulation = np.tanh(self.executive_control)
        
        # Update working memory with maintenance
        maintenance = self.recurrent_weights @ self.working_memory
        new_working_memory = (self.maintenance_strength * maintenance + 
                            (1 - self.maintenance_strength) * attended_input)
        
        # Apply executive control
        self.working_memory = new_working_memory * (1 + control_modulation)
        
        return self.working_memory
    
    def predict_reward(self, state: np.ndarray) -> float:
        """Predict reward based on current state."""
        if len(state) != self.size:
            # Resize state to match PFC size
            if len(state) > self.size:
                state = state[:self.size]
            else:
                state = np.pad(state, (0, self.size - len(state)))
        
        self.value_estimate = np.dot(self.reward_predictor, state)
        return self.value_estimate
    
    def update_reward_prediction(self, actual_reward: float, learning_rate: float = 0.01):
        """Update reward prediction with temporal difference learning."""
        prediction_error = actual_reward - self.value_estimate
        
        # Update reward predictor
        self.reward_predictor += learning_rate * prediction_error * self.working_memory
        
        return prediction_error
    
    def executive_control_signal(self, goal_state: np.ndarray) -> np.ndarray:
        """Generate executive control signal based on goal."""
        # Compare current working memory to goal
        goal_error = goal_state - self.working_memory
        
        # Generate control signal
        control_signal = np.tanh(goal_error * 2.0)
        
        # Update executive control state
        self.executive_control = 0.7 * self.executive_control + 0.3 * control_signal
        
        return control_signal


class HippocampusPFCLoop:
    """Integrated hippocampus-PFC memory loop system."""
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize subnetworks
        self.ca3 = CA3Network(
            size=config.get('ca3_size', 1000),
            sparsity=config.get('ca3_sparsity', 0.02)
        )
        
        self.ca1 = CA1Network(
            size=config.get('ca1_size', 800)
        )
        
        self.pfc = PFCNetwork(
            size=config.get('pfc_size', 500)
        )
        
        # Memory storage
        self.memory_traces: List[MemoryTrace] = []
        self.trace_index = 0
        
        # Learning parameters
        self.consolidation_rate = config.get('consolidation_rate', 0.001)
        self.replay_probability = config.get('replay_probability', 0.1)
        
        # Current state
        self.current_phase = MemoryPhase.ENCODING
        self.current_context = np.zeros(100)  # Default context size
        
        # Reinforcement learning
        self.reward_history = deque(maxlen=1000)
        self.prediction_errors = deque(maxlen=1000)
        
        # Backend mappings
        self.nengo_model = None
        self.bindsnet_model = None
        
        if config.get('initialize_nengo', False) and NENGO_AVAILABLE:
            self._initialize_nengo_backend()
        
        if config.get('initialize_bindsnet', False) and BINDSNET_AVAILABLE:
            self._initialize_bindsnet_backend()
    
    def encode_memory(self, pattern: np.ndarray, context: np.ndarray, 
                     reward: float = 0.0) -> int:
        """Encode new memory through the hippocampal-PFC loop."""
        self.current_phase = MemoryPhase.ENCODING
        
        # CA3 pattern separation/completion
        ca3_code = self.ca3.encode_pattern(pattern, context)
        
        # CA1 temporal sequence processing
        ca1_code = self.ca1.process_ca3_input(ca3_code)
        
        # PFC working memory integration
        pfc_representation = self.pfc.update_working_memory(ca1_code)
        
        # Reward prediction and learning
        predicted_reward = self.pfc.predict_reward(pfc_representation)
        prediction_error = self.pfc.update_reward_prediction(reward)
        
        # Create memory trace
        trace = MemoryTrace(
            pattern=pattern,
            context=context,
            timestamp=time.time(),
            ca3_sparse_code=ca3_code,
            ca1_sequence_code=ca1_code,
            pfc_working_memory=pfc_representation,
            reward_signal=reward,
            prediction_error=prediction_error,
            novelty_signal=self._compute_novelty(pattern)
        )
        
        # Store trace
        self.memory_traces.append(trace)
        trace_id = len(self.memory_traces) - 1
        
        # Update learning signals
        self.reward_history.append(reward)
        self.prediction_errors.append(prediction_error)
        
        return trace_id
    
    def retrieve_memory(self, cue: np.ndarray, context: Optional[np.ndarray] = None) -> Optional[MemoryTrace]:
        """Retrieve memory using hippocampal pattern completion."""
        self.current_phase = MemoryPhase.RETRIEVAL
        
        if context is None:
            context = self.current_context
        
        # CA3 pattern completion
        ca3_retrieved = self.ca3.retrieve_pattern(cue)
        
        # Find best matching stored trace
        best_match = None
        best_similarity = 0.0
        
        for trace in self.memory_traces:
            similarity = np.dot(ca3_retrieved, trace.ca3_sparse_code) / (
                np.linalg.norm(ca3_retrieved) * np.linalg.norm(trace.ca3_sparse_code) + 1e-8)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = trace
        
        if best_match and best_similarity > 0.3:  # Retrieval threshold
            # Update retrieval statistics
            best_match.retrieval_count += 1
            best_match.last_accessed = time.time()
            
            # Reactivate in PFC working memory
            self.pfc.working_memory = best_match.pfc_working_memory
            
            return best_match
        
        return None
    
    def consolidate_memories(self, sleep_stage: str = 'SWS'):
        """Consolidate memories during sleep-like states."""
        self.current_phase = MemoryPhase.CONSOLIDATION
        
        if sleep_stage == 'SWS':  # Slow-wave sleep
            # Select memories for consolidation based on reward and recency
            consolidation_candidates = [
                trace for trace in self.memory_traces
                if trace.consolidation_level < 1.0 and 
                (trace.reward_signal > 0.1 or trace.novelty_signal > 0.5)
            ]
            
            # Sort by priority (reward + novelty - time decay)
            consolidation_candidates.sort(
                key=lambda t: t.reward_signal + t.novelty_signal - 
                             0.1 * (time.time() - t.timestamp) / 3600,  # Hour decay
                reverse=True
            )
            
            # Consolidate top candidates
            for trace in consolidation_candidates[:10]:  # Limit consolidation
                self._consolidate_trace(trace)
    
    def _consolidate_trace(self, trace: MemoryTrace):
        """Consolidate individual memory trace."""
        # Replay through the circuit
        ca3_reactivated = self.ca3.retrieve_pattern(trace.ca3_sparse_code)
        ca1_reactivated = self.ca1.process_ca3_input(ca3_reactivated)
        pfc_reactivated = self.pfc.update_working_memory(ca1_reactivated)
        
        # Strengthen connections (simplified)
        trace.consolidation_level = min(1.0, trace.consolidation_level + self.consolidation_rate)
        
        # Update trace representations
        trace.ca3_sparse_code = 0.9 * trace.ca3_sparse_code + 0.1 * ca3_reactivated
        trace.ca1_sequence_code = 0.9 * trace.ca1_sequence_code + 0.1 * ca1_reactivated
        trace.pfc_working_memory = 0.9 * trace.pfc_working_memory + 0.1 * pfc_reactivated
    
    def replay_sequence(self, trace_ids: List[int]):
        """Replay sequence of memories for learning."""
        self.current_phase = MemoryPhase.REPLAY
        
        sequence_traces = [self.memory_traces[tid] for tid in trace_ids if tid < len(self.memory_traces)]
        
        if not sequence_traces:
            return
        
        # Replay through CA1 for sequence learning
        ca3_sequence = [trace.ca3_sparse_code for trace in sequence_traces]
        ca1_sequence_code = self.ca1.encode_sequence(ca3_sequence)
        
        # Update sequence representations
        for i, trace in enumerate(sequence_traces):
            trace.ca1_temporal_context = ca1_sequence_code
            
            # Update causal relationships
            if i > 0:
                trace.causal_predecessors.append(trace_ids[i-1])
            if i < len(sequence_traces) - 1:
                trace.causal_successors.append(trace_ids[i+1])
    
    def _compute_novelty(self, pattern: np.ndarray) -> float:
        """Compute novelty signal for new pattern."""
        if not self.memory_traces:
            return 1.0  # First pattern is maximally novel
        
        # Compare to recent memories
        recent_traces = self.memory_traces[-10:]  # Last 10 memories
        similarities = []
        
        for trace in recent_traces:
            similarity = np.dot(pattern, trace.pattern) / (
                np.linalg.norm(pattern) * np.linalg.norm(trace.pattern) + 1e-8)
            similarities.append(similarity)
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        return 1.0 - max_similarity
    
    def update_context(self, new_context: np.ndarray):
        """Update current context representation."""
        self.current_context = new_context
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        if not self.memory_traces:
            return {'total_memories': 0}
        
        consolidation_levels = [trace.consolidation_level for trace in self.memory_traces]
        reward_signals = [trace.reward_signal for trace in self.memory_traces]
        retrieval_counts = [trace.retrieval_count for trace in self.memory_traces]
        
        return {
            'total_memories': len(self.memory_traces),
            'mean_consolidation': np.mean(consolidation_levels),
            'mean_reward': np.mean(reward_signals),
            'mean_retrievals': np.mean(retrieval_counts),
            'recent_prediction_error': np.mean(list(self.prediction_errors)[-10:]) if self.prediction_errors else 0.0
        }
    
    def _initialize_nengo_backend(self):
        """Initialize Nengo backend for validation."""
        if not NENGO_AVAILABLE:
            print("Warning: Nengo not available")
            return
        
        # Create Nengo model
        self.nengo_model = nengo.Network(label="Hippocampus-PFC Loop")
        
        with self.nengo_model:
            # CA3 ensemble
            self.nengo_ca3 = nengo.Ensemble(
                n_neurons=self.ca3.size,
                dimensions=100,  # Reduced dimensionality for Nengo
                label="CA3"
            )
            
            # CA1 ensemble
            self.nengo_ca1 = nengo.Ensemble(
                n_neurons=self.ca1.size,
                dimensions=100,
                label="CA1"
            )
            
            # PFC ensemble
            self.nengo_pfc = nengo.Ensemble(
                n_neurons=self.pfc.size,
                dimensions=100,
                label="PFC"
            )
            
            # Connections
            nengo.Connection(self.nengo_ca3, self.nengo_ca1, 
                           transform=0.5, label="CA3->CA1")
            nengo.Connection(self.nengo_ca1, self.nengo_pfc,
                           transform=0.3, label="CA1->PFC")
            
            # Recurrent connections for working memory
            nengo.Connection(self.nengo_pfc, self.nengo_pfc,
                           transform=0.8, synapse=0.1, label="PFC recurrent")
        
        print("Nengo backend initialized")
    
    def _initialize_bindsnet_backend(self):
        """Initialize BindsNET backend for spiking validation."""
        if not BINDSNET_AVAILABLE:
            print("Warning: BindsNET not available")
            return
        
        # This would implement BindsNET network creation
        # For now, just placeholder
        print("BindsNET backend would be initialized here")
        self.bindsnet_model = "placeholder"
    
    def validate_with_nengo(self, duration: float = 1.0) -> Dict[str, Any]:
        """Run validation with Nengo backend."""
        if not self.nengo_model:
            return {'error': 'Nengo backend not initialized'}
        
        # Create simulator
        with nengo.Simulator(self.nengo_model) as sim:
            sim.run(duration)
        
        # Return simulation results
        return {
            'ca3_activity': sim.data[self.nengo_ca3][-100:],  # Last 100 timesteps
            'ca1_activity': sim.data[self.nengo_ca1][-100:],
            'pfc_activity': sim.data[self.nengo_pfc][-100:]
        }