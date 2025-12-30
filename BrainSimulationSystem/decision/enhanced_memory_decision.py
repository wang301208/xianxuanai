"""
Enhanced Decision-Memory Integration System

Integrates the hippocampus-PFC memory loop with decision making,
incorporating reinforcement learning reward signals and memory-guided decisions.
"""

from __future__ import annotations

import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass

from .core import DecisionEngine
from ..core.hippocampus_pfc_loop import HippocampusPFCLoop, MemoryTrace, MemoryPhase
from ..learning.experience import ExperienceLearningSystem


@dataclass
class DecisionContext:
    """Enhanced decision context with memory integration."""
    
    # Current situation
    state_vector: np.ndarray
    available_actions: List[Dict[str, Any]]
    environmental_context: Dict[str, Any]
    
    # Memory-related
    retrieved_memories: List[MemoryTrace]
    memory_confidence: float = 0.0
    episodic_context: Optional[np.ndarray] = None
    
    # Learning signals
    expected_reward: float = 0.0
    uncertainty: float = 0.0
    novelty: float = 0.0
    
    # Temporal information
    timestamp: float = 0.0
    sequence_position: int = 0
    
    # Executive control
    goal_state: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None


class MemoryGuidedDecisionEngine(DecisionEngine):
    """Decision engine enhanced with hippocampus-PFC memory loop."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Initialize memory loop
        memory_config = config.get('memory_config', {})
        self.memory_loop = HippocampusPFCLoop(memory_config)
        
        # Decision-memory integration parameters
        self.memory_weight = config.get('memory_weight', 0.4)
        self.novelty_bonus = config.get('novelty_bonus', 0.2)
        self.uncertainty_penalty = config.get('uncertainty_penalty', 0.1)
        
        # Working memory capacity
        self.working_memory_capacity = config.get('working_memory_capacity', 7)
        self.active_memories = deque(maxlen=self.working_memory_capacity)
        
        # Reinforcement learning integration
        self.rl_learning_rate = config.get('rl_learning_rate', 0.01)
        self.discount_factor = config.get('discount_factor', 0.95)
        
        # Decision history for sequence learning
        self.decision_history = deque(maxlen=100)
        self.outcome_history = deque(maxlen=100)
        
        # Executive control parameters
        self.goal_persistence = config.get('goal_persistence', 0.8)
        self.current_goal = None
        self.goal_progress = 0.0
    
    def make_enhanced_decision(self, decision_context: DecisionContext) -> Tuple[int, Dict[str, Any]]:
        """Make decision using integrated memory-decision system."""
        
        # Phase 1: Memory retrieval and context building
        memory_info = self._retrieve_relevant_memories(decision_context)
        
        # Phase 2: Working memory integration
        working_memory_state = self._update_working_memory(decision_context, memory_info)
        
        # Phase 3: Value estimation with memory guidance
        action_values = self._estimate_action_values(decision_context, memory_info)
        
        # Phase 4: Executive control and goal modulation
        modulated_values = self._apply_executive_control(action_values, decision_context)
        
        # Phase 5: Decision selection with exploration
        selected_action, selection_info = self._select_action(modulated_values, decision_context)
        
        # Phase 6: Memory encoding of decision
        self._encode_decision_memory(decision_context, selected_action, selection_info)
        
        # Prepare decision metadata
        decision_metadata = {
            'memory_retrievals': len(memory_info['retrieved_traces']),
            'memory_confidence': memory_info['confidence'],
            'working_memory_load': len(self.active_memories),
            'predicted_value': modulated_values[selected_action],
            'novelty_signal': decision_context.novelty,
            'uncertainty': decision_context.uncertainty,
            'goal_relevance': selection_info.get('goal_relevance', 0.0),
            'exploration_bonus': selection_info.get('exploration_bonus', 0.0)
        }
        
        return selected_action, decision_metadata
    
    def _retrieve_relevant_memories(self, context: DecisionContext) -> Dict[str, Any]:
        """Retrieve memories relevant to current decision context."""
        
        # Create retrieval cue from current state
        retrieval_cue = context.state_vector
        
        # Retrieve from hippocampus
        retrieved_trace = self.memory_loop.retrieve_memory(
            retrieval_cue, 
            context.episodic_context
        )
        
        retrieved_traces = [retrieved_trace] if retrieved_trace else []
        
        # Retrieve similar experiences from decision history
        similar_decisions = self._find_similar_decisions(context.state_vector)
        
        # Compute memory confidence
        confidence = 0.0
        if retrieved_traces:
            # Base confidence on retrieval strength and consolidation
            confidence = retrieved_traces[0].consolidation_level * 0.8
            if retrieved_traces[0].retrieval_count > 0:
                confidence += 0.2  # Bonus for previously accessed memories
        
        return {
            'retrieved_traces': retrieved_traces,
            'similar_decisions': similar_decisions,
            'confidence': confidence,
            'memory_context': retrieved_trace.context if retrieved_trace else None
        }
    
    def _find_similar_decisions(self, current_state: np.ndarray, 
                              similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find similar past decisions from decision history."""
        similar_decisions = []
        
        for decision_record in list(self.decision_history)[-20:]:  # Check recent decisions
            past_state = decision_record.get('state_vector')
            if past_state is not None:
                # Compute similarity
                similarity = np.dot(current_state, past_state) / (
                    np.linalg.norm(current_state) * np.linalg.norm(past_state) + 1e-8)
                
                if similarity >= similarity_threshold:
                    similar_decisions.append({
                        'decision_record': decision_record,
                        'similarity': similarity
                    })
        
        # Sort by similarity
        similar_decisions.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_decisions[:5]  # Return top 5
    
    def _update_working_memory(self, context: DecisionContext, 
                             memory_info: Dict[str, Any]) -> np.ndarray:
        """Update working memory with current context and retrieved memories."""
        
        # Add current context to working memory
        current_item = {
            'type': 'current_context',
            'content': context.state_vector,
            'timestamp': context.timestamp,
            'relevance': 1.0
        }
        self.active_memories.append(current_item)
        
        # Add retrieved memories to working memory
        for trace in memory_info['retrieved_traces']:
            memory_item = {
                'type': 'episodic_memory',
                'content': trace.pattern,
                'context': trace.context,
                'reward': trace.reward_signal,
                'relevance': memory_info['confidence'],
                'timestamp': trace.timestamp
            }
            self.active_memories.append(memory_item)
        
        # Update PFC working memory
        if memory_info['retrieved_traces']:
            ca1_input = memory_info['retrieved_traces'][0].ca1_sequence_code
            working_memory_state = self.memory_loop.pfc.update_working_memory(ca1_input)
        else:
            # Use current state as input
            padded_state = np.pad(context.state_vector, 
                                (0, max(0, self.memory_loop.pfc.size - len(context.state_vector))))
            working_memory_state = self.memory_loop.pfc.update_working_memory(padded_state[:self.memory_loop.pfc.size])
        
        return working_memory_state
    
    def _estimate_action_values(self, context: DecisionContext, 
                              memory_info: Dict[str, Any]) -> np.ndarray:
        """Estimate action values using memory-guided prediction."""
        
        n_actions = len(context.available_actions)
        action_values = np.zeros(n_actions)
        
        # Base value estimation using standard decision engine
        for i, action in enumerate(context.available_actions):
            # Cognitive evaluation
            cognitive_value = self._evaluate_cognitive(action)
            
            # Emotional evaluation
            emotional_value = self._evaluate_emotional(action)
            
            # Social evaluation
            social_value = self._evaluate_social(action)
            
            # Base value
            base_value = (cognitive_value * self.weights['cognitive'] +
                         emotional_value * self.weights['emotional'] +
                         social_value * self.weights['social'])
            
            action_values[i] = base_value
        
        # Memory-guided value adjustment
        if memory_info['retrieved_traces']:
            for i, action in enumerate(context.available_actions):
                memory_value = self._compute_memory_guided_value(action, memory_info)
                action_values[i] += self.memory_weight * memory_value
        
        # Similar decision guidance
        for similar_decision in memory_info['similar_decisions']:
            past_action = similar_decision['decision_record'].get('selected_action')
            past_outcome = similar_decision['decision_record'].get('outcome', 0.0)
            similarity = similar_decision['similarity']
            
            if past_action is not None and past_action < n_actions:
                # Boost value of previously successful actions in similar contexts
                action_values[past_action] += similarity * past_outcome * 0.3
        
        # Novelty bonus
        for i, action in enumerate(context.available_actions):
            novelty_score = self._compute_action_novelty(action, context)
            action_values[i] += self.novelty_bonus * novelty_score
        
        # Uncertainty penalty
        for i in range(n_actions):
            uncertainty_penalty = context.uncertainty * self.uncertainty_penalty
            action_values[i] -= uncertainty_penalty
        
        return action_values
    
    def _compute_memory_guided_value(self, action: Dict[str, Any], 
                                   memory_info: Dict[str, Any]) -> float:
        """Compute value contribution from retrieved memories."""
        
        if not memory_info['retrieved_traces']:
            return 0.0
        
        trace = memory_info['retrieved_traces'][0]
        
        # Use PFC reward prediction
        action_vector = np.array([action.get(key, 0.0) for key in 
                                ['expected_value', 'risk', 'cost', 'social_norms']])
        
        # Pad or truncate to match PFC size
        if len(action_vector) < self.memory_loop.pfc.size:
            action_vector = np.pad(action_vector, (0, self.memory_loop.pfc.size - len(action_vector)))
        else:
            action_vector = action_vector[:self.memory_loop.pfc.size]
        
        predicted_reward = self.memory_loop.pfc.predict_reward(action_vector)
        
        # Weight by memory confidence and consolidation
        memory_weight = memory_info['confidence'] * trace.consolidation_level
        
        return predicted_reward * memory_weight
    
    def _compute_action_novelty(self, action: Dict[str, Any], 
                              context: DecisionContext) -> float:
        """Compute novelty score for an action in current context."""
        
        # Create action-context vector
        action_vector = np.array([action.get(key, 0.0) for key in 
                                ['expected_value', 'risk', 'cost']])
        combined_vector = np.concatenate([action_vector, context.state_vector[:10]])
        
        # Compare to recent decision history
        if not self.decision_history:
            return 1.0  # First decision is maximally novel
        
        similarities = []
        for past_decision in list(self.decision_history)[-10:]:
            past_vector = past_decision.get('action_context_vector')
            if past_vector is not None:
                similarity = np.dot(combined_vector, past_vector) / (
                    np.linalg.norm(combined_vector) * np.linalg.norm(past_vector) + 1e-8)
                similarities.append(similarity)
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        return 1.0 - max_similarity
    
    def _apply_executive_control(self, action_values: np.ndarray, 
                               context: DecisionContext) -> np.ndarray:
        """Apply executive control and goal-directed modulation."""
        
        modulated_values = action_values.copy()
        
        # Goal-directed modulation
        if context.goal_state is not None:
            goal_signal = self.memory_loop.pfc.executive_control_signal(context.goal_state)
            
            # Modulate action values based on goal relevance
            for i, action in enumerate(context.available_actions):
                goal_relevance = self._compute_goal_relevance(action, context.goal_state)
                modulated_values[i] += np.mean(goal_signal) * goal_relevance * 0.5
        
        # Attention weighting
        if context.attention_weights is not None:
            attention_modulation = context.attention_weights[:len(modulated_values)]
            modulated_values *= (1.0 + attention_modulation)
        
        return modulated_values
    
    def _compute_goal_relevance(self, action: Dict[str, Any], 
                              goal_state: np.ndarray) -> float:
        """Compute how relevant an action is to the current goal."""
        
        # Simple heuristic: actions with higher expected value are more goal-relevant
        expected_value = action.get('expected_value', 0.0)
        
        # Normalize to [0, 1] range
        relevance = (expected_value + 1.0) / 2.0  # Assuming values in [-1, 1]
        
        return np.clip(relevance, 0.0, 1.0)
    
    def _select_action(self, action_values: np.ndarray, 
                      context: DecisionContext) -> Tuple[int, Dict[str, Any]]:
        """Select action using exploration-exploitation strategy."""
        
        # Softmax selection with temperature based on uncertainty
        temperature = 1.0 + context.uncertainty
        exp_values = np.exp(action_values / temperature)
        probabilities = exp_values / np.sum(exp_values)
        
        # Add exploration bonus for highly uncertain situations
        if context.uncertainty > 0.7:
            # Uniform exploration component
            uniform_prob = 1.0 / len(action_values)
            exploration_weight = 0.2
            probabilities = (1 - exploration_weight) * probabilities + exploration_weight * uniform_prob
        
        # Select action
        selected_action = np.random.choice(len(action_values), p=probabilities)
        
        selection_info = {
            'action_probabilities': probabilities,
            'temperature': temperature,
            'exploration_bonus': 0.2 if context.uncertainty > 0.7 else 0.0,
            'goal_relevance': self._compute_goal_relevance(
                context.available_actions[selected_action], 
                context.goal_state) if context.goal_state is not None else 0.0
        }
        
        return selected_action, selection_info
    
    def _encode_decision_memory(self, context: DecisionContext, 
                              selected_action: int, selection_info: Dict[str, Any]):
        """Encode the decision episode in memory."""
        
        # Create decision pattern
        action_info = context.available_actions[selected_action]
        decision_pattern = np.concatenate([
            context.state_vector,
            [selected_action, action_info.get('expected_value', 0.0)]
        ])
        
        # Encode in hippocampus-PFC loop
        trace_id = self.memory_loop.encode_memory(
            pattern=decision_pattern,
            context=context.episodic_context if context.episodic_context is not None else context.state_vector,
            reward=context.expected_reward
        )
        
        # Store in decision history
        decision_record = {
            'timestamp': context.timestamp,
            'state_vector': context.state_vector,
            'selected_action': selected_action,
            'action_info': action_info,
            'expected_reward': context.expected_reward,
            'memory_trace_id': trace_id,
            'action_context_vector': np.concatenate([
                np.array([action_info.get(key, 0.0) for key in ['expected_value', 'risk', 'cost']]),
                context.state_vector[:10]
            ])
        }
        
        self.decision_history.append(decision_record)
    
    def update_outcome(self, actual_reward: float, outcome_context: Optional[Dict[str, Any]] = None):
        """Update memory and learning systems with actual outcome."""
        
        if not self.decision_history:
            return
        
        # Get last decision
        last_decision = self.decision_history[-1]
        
        # Update outcome history
        outcome_record = {
            'timestamp': time.time(),
            'actual_reward': actual_reward,
            'predicted_reward': last_decision['expected_reward'],
            'prediction_error': actual_reward - last_decision['expected_reward'],
            'context': outcome_context
        }
        self.outcome_history.append(outcome_record)
        
        # Update PFC reward prediction
        prediction_error = self.memory_loop.pfc.update_reward_prediction(
            actual_reward, self.rl_learning_rate)
        
        # Update memory trace with outcome
        trace_id = last_decision.get('memory_trace_id')
        if trace_id is not None and trace_id < len(self.memory_loop.memory_traces):
            trace = self.memory_loop.memory_traces[trace_id]
            trace.reward_signal = actual_reward
            trace.prediction_error = prediction_error
        
        # Trigger consolidation if reward is significant
        if abs(actual_reward) > 0.5:
            self.memory_loop.consolidate_memories()
    
    def set_goal(self, goal_state: np.ndarray, persistence: Optional[float] = None):
        """Set current goal for executive control."""
        self.current_goal = goal_state
        if persistence is not None:
            self.goal_persistence = persistence
        self.goal_progress = 0.0
    
    def update_goal_progress(self, progress: float):
        """Update progress toward current goal."""
        self.goal_progress = np.clip(progress, 0.0, 1.0)
        
        # Clear goal if completed
        if self.goal_progress >= 1.0:
            self.current_goal = None
            self.goal_progress = 0.0
    
    def get_memory_guided_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory-guided decision making."""
        
        # Memory loop statistics
        memory_stats = self.memory_loop.get_memory_statistics()
        
        # Decision history statistics
        if self.outcome_history:
            recent_errors = [record['prediction_error'] for record in list(self.outcome_history)[-20:]]
            mean_prediction_error = np.mean(np.abs(recent_errors))
            
            recent_rewards = [record['actual_reward'] for record in list(self.outcome_history)[-20:]]
            mean_reward = np.mean(recent_rewards)
        else:
            mean_prediction_error = 0.0
            mean_reward = 0.0
        
        return {
            **memory_stats,
            'decision_history_length': len(self.decision_history),
            'working_memory_load': len(self.active_memories),
            'mean_prediction_error': mean_prediction_error,
            'mean_reward': mean_reward,
            'current_goal_set': self.current_goal is not None,
            'goal_progress': self.goal_progress
        }