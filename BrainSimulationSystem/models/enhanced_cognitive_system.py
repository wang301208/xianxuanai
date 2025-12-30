"""
增强的认知系统

将抽象的认知基类映射到真实脑区功能，包括：
- 海马 CA3/CA1、DG 结构的记忆系统
- 基底节-前额叶的决策系统
- 感知-运动系统与环境交互
- 闭环行为模型
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from ..config.enhanced_brain_config import BrainRegion, CellType

logger = logging.getLogger(__name__)

class CognitiveState(Enum):
    """认知状态枚举"""
    RESTING = "resting"
    ENCODING = "encoding"
    RETRIEVAL = "retrieval"
    CONSOLIDATION = "consolidation"
    DECISION_MAKING = "decision_making"
    ATTENTION = "attention"
    WORKING_MEMORY = "working_memory"
    PLANNING = "planning"
    EXECUTION = "execution"

class MemoryType(Enum):
    """记忆类型枚举"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    SENSORY = "sensory"
    EMOTIONAL = "emotional"

class AttentionType(Enum):
    """注意类型枚举"""
    FOCUSED = "focused"
    DIVIDED = "divided"
    SUSTAINED = "sustained"
    SELECTIVE = "selective"
    EXECUTIVE = "executive"

@dataclass
class CognitiveEvent:
    """认知事件"""
    event_id: str
    event_type: str
    timestamp: float
    brain_region: str
    parameters: Dict[str, Any]
    duration: float = 0.0
    intensity: float = 1.0

@dataclass
class MemoryTrace:
    """记忆痕迹"""
    trace_id: str
    content: Any
    memory_type: MemoryType
    encoding_time: float
    strength: float = 1.0
    consolidation_level: float = 0.0
    retrieval_count: int = 0
    last_retrieval: float = 0.0
    
    # 神经基础
    neural_pattern: Optional[np.ndarray] = None
    brain_regions: List[str] = field(default_factory=list)
    synaptic_weights: Dict[str, float] = field(default_factory=dict)

class HippocampalMemorySystem:
    """海马记忆系统"""
    
    def __init__(self, dg_size: int = 15000, ca3_size: int = 8000, ca1_size: int = 12000):
        # 海马子区域
        self.dg_size = dg_size
        self.ca3_size = ca3_size
        self.ca1_size = ca1_size
        
        # 神经元活动模式
        self.dg_activity = np.zeros(dg_size)
        self.ca3_activity = np.zeros(ca3_size)
        self.ca1_activity = np.zeros(ca1_size)
        
        # 连接权重
        self.dg_to_ca3_weights = np.random.rand(dg_size, ca3_size) * 0.1
        self.ca3_to_ca1_weights = np.random.rand(ca3_size, ca1_size) * 0.2
        self.ca3_recurrent_weights = np.random.rand(ca3_size, ca3_size) * 0.15
        
        # 记忆存储
        self.memory_traces = {}
        self.episodic_buffer = []
        self.semantic_network = {}
        
        # 可塑性参数
        self.learning_rate = 0.01
        self.consolidation_rate = 0.001
        self.forgetting_rate = 0.0001
        
        # 振荡参数
        self.theta_frequency = 8.0  # Hz
        self.gamma_frequency = 40.0  # Hz
        self.theta_phase = 0.0
        self.gamma_phase = 0.0
        
        # 状态变量
        self.current_state = CognitiveState.RESTING
        self.attention_level = 0.5
        self.arousal_level = 0.5
    
    def encode_episodic_memory(self, input_pattern: np.ndarray, 
                              context: Dict[str, Any]) -> str:
        """编码情景记忆"""
        self.current_state = CognitiveState.ENCODING
        
        # 齿状回稀疏编码
        dg_pattern = self._sparse_coding(input_pattern, sparsity=0.02)
        self.dg_activity = dg_pattern
        
        # CA3模式分离
        ca3_input = np.dot(dg_pattern, self.dg_to_ca3_weights)
        ca3_pattern = self._pattern_separation(ca3_input, noise_level=0.1)
        self.ca3_activity = ca3_pattern
        
        # CA1模式完成
        ca1_input = np.dot(ca3_pattern, self.ca3_to_ca1_weights)
        ca1_pattern = self._pattern_completion(ca1_input)
        self.ca1_activity = ca1_pattern
        
        # 创建记忆痕迹
        trace_id = f"episodic_{len(self.memory_traces)}"
        memory_trace = MemoryTrace(
            trace_id=trace_id,
            content={'input': input_pattern, 'context': context},
            memory_type=MemoryType.EPISODIC,
            encoding_time=time.time(),
            neural_pattern=ca1_pattern.copy(),
            brain_regions=['DG', 'CA3', 'CA1']
        )
        
        self.memory_traces[trace_id] = memory_trace
        self.episodic_buffer.append(trace_id)
        
        # 更新连接权重（Hebbian学习）
        self._update_synaptic_weights(dg_pattern, ca3_pattern, ca1_pattern)
        
        return trace_id
    
    def retrieve_episodic_memory(self, cue_pattern: np.ndarray, 
                                threshold: float = 0.7) -> Optional[MemoryTrace]:
        """检索情景记忆"""
        self.current_state = CognitiveState.RETRIEVAL
        
        best_match = None
        best_similarity = 0.0
        
        for trace_id in self.episodic_buffer:
            if trace_id in self.memory_traces:
                trace = self.memory_traces[trace_id]
                
                # 计算相似性
                similarity = self._calculate_similarity(
                    cue_pattern, trace.neural_pattern
                )
                
                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_match = trace
        
        if best_match:
            # 更新检索统计
            best_match.retrieval_count += 1
            best_match.last_retrieval = time.time()
            
            # 重新激活神经模式
            self._reactivate_pattern(best_match.neural_pattern)
            
            # 强化记忆（检索诱导的可塑性）
            best_match.strength *= 1.1
        
        return best_match
    
    def consolidate_memories(self, dt: float):
        """记忆巩固"""
        self.current_state = CognitiveState.CONSOLIDATION
        
        for trace in self.memory_traces.values():
            # 时间依赖的巩固
            age = time.time() - trace.encoding_time
            consolidation_factor = 1.0 - np.exp(-age / 3600.0)  # 1小时时间常数
            
            # 更新巩固水平
            trace.consolidation_level += self.consolidation_rate * consolidation_factor * dt
            trace.consolidation_level = min(1.0, trace.consolidation_level)
            
            # 强度衰减（遗忘）
            if trace.retrieval_count == 0:
                trace.strength *= (1.0 - self.forgetting_rate * dt)
            
            # 移除过弱的记忆
            if trace.strength < 0.1:
                self._remove_memory_trace(trace.trace_id)
    
    def _sparse_coding(self, input_pattern: np.ndarray, sparsity: float) -> np.ndarray:
        """稀疏编码（齿状回功能）"""
        # 竞争性学习
        activation = np.dot(input_pattern, np.random.rand(len(input_pattern), self.dg_size))
        
        # 选择最活跃的神经元
        k = int(self.dg_size * sparsity)
        top_k_indices = np.argpartition(activation, -k)[-k:]
        
        sparse_pattern = np.zeros(self.dg_size)
        sparse_pattern[top_k_indices] = activation[top_k_indices]
        
        return sparse_pattern
    
    def _pattern_separation(self, input_pattern: np.ndarray, 
                           noise_level: float) -> np.ndarray:
        """模式分离（CA3功能）"""
        # 添加噪声增强分离
        noisy_input = input_pattern + np.random.normal(0, noise_level, len(input_pattern))
        
        # 递归网络动力学
        ca3_state = np.tanh(noisy_input)
        
        # 多次迭代达到稳定状态
        for _ in range(5):
            recurrent_input = np.dot(ca3_state, self.ca3_recurrent_weights)
            ca3_state = np.tanh(noisy_input + 0.5 * recurrent_input)
        
        return ca3_state
    
    def _pattern_completion(self, input_pattern: np.ndarray) -> np.ndarray:
        """模式完成（CA1功能）"""
        # 线性变换后非线性激活
        ca1_activation = np.tanh(input_pattern)
        
        # 添加噪声容忍性
        noise_threshold = 0.1
        ca1_activation[np.abs(ca1_activation) < noise_threshold] = 0
        
        return ca1_activation
    
    def _calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """计算模式相似性"""
        if pattern1 is None or pattern2 is None:
            return 0.0
        
        # 归一化相关系数
        norm1 = np.linalg.norm(pattern1)
        norm2 = np.linalg.norm(pattern2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        correlation = np.dot(pattern1, pattern2) / (norm1 * norm2)
        return max(0.0, correlation)
    
    def _reactivate_pattern(self, pattern: np.ndarray):
        """重新激活神经模式"""
        # 部分重新激活CA1
        self.ca1_activity = 0.8 * pattern + 0.2 * self.ca1_activity
        
        # 反向传播到CA3
        ca3_reactivation = np.dot(self.ca1_activity, self.ca3_to_ca1_weights.T)
        self.ca3_activity = 0.6 * np.tanh(ca3_reactivation) + 0.4 * self.ca3_activity
    
    def _update_synaptic_weights(self, dg_pattern: np.ndarray, 
                                ca3_pattern: np.ndarray, ca1_pattern: np.ndarray):
        """更新突触权重"""
        # Hebbian学习：DG -> CA3
        dg_ca3_update = np.outer(dg_pattern, ca3_pattern) * self.learning_rate
        self.dg_to_ca3_weights += dg_ca3_update
        
        # CA3 -> CA1
        ca3_ca1_update = np.outer(ca3_pattern, ca1_pattern) * self.learning_rate
        self.ca3_to_ca1_weights += ca3_ca1_update
        
        # CA3递归连接
        ca3_recurrent_update = np.outer(ca3_pattern, ca3_pattern) * self.learning_rate * 0.5
        self.ca3_recurrent_weights += ca3_recurrent_update
        
        # 权重归一化
        self.dg_to_ca3_weights = np.clip(self.dg_to_ca3_weights, 0, 1)
        self.ca3_to_ca1_weights = np.clip(self.ca3_to_ca1_weights, 0, 1)
        self.ca3_recurrent_weights = np.clip(self.ca3_recurrent_weights, 0, 1)
    
    def _remove_memory_trace(self, trace_id: str):
        """移除记忆痕迹"""
        if trace_id in self.memory_traces:
            del self.memory_traces[trace_id]
        
        if trace_id in self.episodic_buffer:
            self.episodic_buffer.remove(trace_id)
    
    def update_oscillations(self, dt: float):
        """更新振荡活动"""
        # Theta振荡
        self.theta_phase += 2 * np.pi * self.theta_frequency * dt / 1000.0
        theta_amplitude = 0.5 * np.sin(self.theta_phase)
        
        # Gamma振荡
        self.gamma_phase += 2 * np.pi * self.gamma_frequency * dt / 1000.0
        gamma_amplitude = 0.2 * np.sin(self.gamma_phase)
        
        # 调制神经元活动
        oscillation_modulation = theta_amplitude + gamma_amplitude
        
        self.dg_activity *= (1.0 + 0.1 * oscillation_modulation)
        self.ca3_activity *= (1.0 + 0.15 * oscillation_modulation)
        self.ca1_activity *= (1.0 + 0.1 * oscillation_modulation)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        total_memories = len(self.memory_traces)
        
        if total_memories == 0:
            return {'total_memories': 0}
        
        strengths = [trace.strength for trace in self.memory_traces.values()]
        consolidation_levels = [trace.consolidation_level for trace in self.memory_traces.values()]
        retrieval_counts = [trace.retrieval_count for trace in self.memory_traces.values()]
        
        return {
            'total_memories': total_memories,
            'episodic_memories': len(self.episodic_buffer),
            'mean_strength': np.mean(strengths),
            'mean_consolidation': np.mean(consolidation_levels),
            'total_retrievals': sum(retrieval_counts),
            'theta_phase': self.theta_phase,
            'gamma_phase': self.gamma_phase,
            'current_state': self.current_state.value
        }

class BasalGangliaDecisionSystem:
    """基底节决策系统"""
    
    def __init__(self, striatum_size: int = 80000, gpe_size: int = 9000, 
                 gpi_size: int = 6000, stn_size: int = 5000):
        # 基底节子结构
        self.striatum_size = striatum_size
        self.gpe_size = gpe_size
        self.gpi_size = gpi_size
        self.stn_size = stn_size
        
        # 神经元活动
        self.striatum_d1_activity = np.zeros(striatum_size // 2)  # 直接通路
        self.striatum_d2_activity = np.zeros(striatum_size // 2)  # 间接通路
        self.gpe_activity = np.zeros(gpe_size)
        self.gpi_activity = np.zeros(gpi_size)
        self.stn_activity = np.zeros(stn_size)
        
        # 连接权重
        self.cortex_to_striatum_weights = np.random.rand(1000, striatum_size) * 0.3
        self.striatum_d1_to_gpi_weights = np.random.rand(striatum_size // 2, gpi_size) * 0.4
        self.striatum_d2_to_gpe_weights = np.random.rand(striatum_size // 2, gpe_size) * 0.4
        self.gpe_to_gpi_weights = np.random.rand(gpe_size, gpi_size) * 0.5
        self.gpe_to_stn_weights = np.random.rand(gpe_size, stn_size) * 0.3
        self.stn_to_gpi_weights = np.random.rand(stn_size, gpi_size) * 0.6
        
        # 多巴胺系统
        self.dopamine_level = 1.0
        self.reward_prediction_error = 0.0
        self.expected_reward = 0.0
        
        # 动作选择
        self.action_values = {}
        self.action_probabilities = {}
        self.selected_action = None
        
        # 学习参数
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1
        
        # 状态变量
        self.current_state = CognitiveState.RESTING
        self.decision_confidence = 0.0
    
    def process_cortical_input(self, cortical_pattern: np.ndarray, 
                              available_actions: List[str]) -> str:
        """处理皮层输入并选择动作"""
        self.current_state = CognitiveState.DECISION_MAKING
        
        # 皮层到纹状体的输入
        striatum_input = np.dot(cortical_pattern, self.cortex_to_striatum_weights)
        
        # 多巴胺调制
        d1_modulation = self.dopamine_level
        d2_modulation = 1.0 / (1.0 + self.dopamine_level)
        
        # 纹状体活动
        self.striatum_d1_activity = np.tanh(striatum_input[:self.striatum_size//2] * d1_modulation)
        self.striatum_d2_activity = np.tanh(striatum_input[self.striatum_size//2:] * d2_modulation)
        
        # 直接通路：纹状体D1 -> GPi（抑制）
        gpi_inhibition_direct = np.dot(self.striatum_d1_activity, self.striatum_d1_to_gpi_weights)
        
        # 间接通路：纹状体D2 -> GPe -> GPi
        gpe_inhibition = np.dot(self.striatum_d2_activity, self.striatum_d2_to_gpe_weights)
        self.gpe_activity = np.maximum(0, 1.0 - gpe_inhibition)  # GPe基础活动被抑制
        
        gpi_disinhibition_indirect = np.dot(self.gpe_activity, self.gpe_to_gpi_weights)
        
        # 超直接通路：皮层 -> STN -> GPi（兴奋）
        stn_excitation = np.mean(cortical_pattern) * 0.5
        self.stn_activity = np.full(self.stn_size, stn_excitation)
        gpi_excitation_hyperdirect = np.dot(self.stn_activity, self.stn_to_gpi_weights)
        
        # GPi最终活动
        self.gpi_activity = (1.0 - gpi_inhibition_direct + 
                           gpi_disinhibition_indirect + 
                           gpi_excitation_hyperdirect * 0.3)
        
        # 动作选择（GPi抑制丘脑，低GPi活动 = 动作释放）
        action_values = {}
        for i, action in enumerate(available_actions):
            if i < len(self.gpi_activity):
                # GPi活动越低，动作价值越高
                action_values[action] = 1.0 / (1.0 + self.gpi_activity[i])
            else:
                action_values[action] = 0.1
        
        self.action_values = action_values
        
        # 计算动作概率（softmax）
        values = np.array(list(action_values.values()))
        exp_values = np.exp(values / 0.1)  # 温度参数
        probabilities = exp_values / np.sum(exp_values)
        
        self.action_probabilities = dict(zip(available_actions, probabilities))
        
        # 选择动作（ε-贪婪策略）
        if np.random.random() < self.exploration_rate:
            selected_action = np.random.choice(available_actions)
        else:
            selected_action = max(action_values, key=action_values.get)
        
        self.selected_action = selected_action
        self.decision_confidence = max(probabilities)
        
        return selected_action
    
    def update_reward_learning(self, received_reward: float):
        """更新奖励学习"""
        # 计算奖励预测误差
        self.reward_prediction_error = received_reward - self.expected_reward
        
        # 更新多巴胺水平
        if self.reward_prediction_error > 0:
            self.dopamine_level = min(2.0, self.dopamine_level + 0.1 * self.reward_prediction_error)
        else:
            self.dopamine_level = max(0.1, self.dopamine_level + 0.05 * self.reward_prediction_error)
        
        # 更新动作价值（强化学习）
        if self.selected_action and self.selected_action in self.action_values:
            current_value = self.action_values[self.selected_action]
            td_error = received_reward - current_value
            self.action_values[self.selected_action] += self.learning_rate * td_error
        
        # 更新期望奖励
        self.expected_reward += 0.1 * self.reward_prediction_error
    
    def update_synaptic_plasticity(self):
        """更新突触可塑性"""
        # 多巴胺依赖的可塑性
        plasticity_factor = (self.dopamine_level - 1.0) * 0.1
        
        # 更新皮层-纹状体连接
        if self.reward_prediction_error != 0:
            # 强化选择的动作路径
            if self.selected_action:
                # 简化的Hebbian更新
                self.cortex_to_striatum_weights *= (1.0 + plasticity_factor * 0.01)
                self.cortex_to_striatum_weights = np.clip(self.cortex_to_striatum_weights, 0, 1)
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """获取决策统计信息"""
        return {
            'dopamine_level': self.dopamine_level,
            'reward_prediction_error': self.reward_prediction_error,
            'expected_reward': self.expected_reward,
            'selected_action': self.selected_action,
            'decision_confidence': self.decision_confidence,
            'action_values': self.action_values.copy(),
            'action_probabilities': self.action_probabilities.copy(),
            'current_state': self.current_state.value,
            'exploration_rate': self.exploration_rate
        }

class PrefrontalWorkingMemorySystem:
    """前额叶工作记忆系统"""
    
    def __init__(self, pfc_size: int = 200000):
        self.pfc_size = pfc_size
        
        # 工作记忆缓冲区
        self.working_memory_buffer = {}
        self.buffer_capacity = 7  # Miller's magic number
        self.maintenance_strength = {}
        
        # 神经元活动
        self.pfc_activity = np.zeros(pfc_size)
        self.persistent_activity = np.zeros(pfc_size)
        
        # 注意控制
        self.attention_weights = np.ones(pfc_size)
        self.attention_focus = None
        
        # 执行控制
        self.inhibition_strength = 0.5
        self.cognitive_flexibility = 0.8
        self.updating_threshold = 0.6
        
        # 状态变量
        self.current_state = CognitiveState.WORKING_MEMORY
        self.cognitive_load = 0.0
    
    def encode_working_memory(self, item_id: str, content: Any, 
                            priority: float = 1.0) -> bool:
        """编码工作记忆项目"""
        # 检查容量限制
        if len(self.working_memory_buffer) >= self.buffer_capacity:
            # 移除优先级最低的项目
            lowest_priority_item = min(self.working_memory_buffer.keys(),
                                     key=lambda x: self.maintenance_strength.get(x, 0))
            self.remove_working_memory_item(lowest_priority_item)
        
        # 添加新项目
        self.working_memory_buffer[item_id] = content
        self.maintenance_strength[item_id] = priority
        
        # 更新神经活动
        self._update_persistent_activity(item_id, content)
        
        # 更新认知负荷
        self.cognitive_load = len(self.working_memory_buffer) / self.buffer_capacity
        
        return True
    
    def retrieve_working_memory(self, item_id: str) -> Optional[Any]:
        """检索工作记忆项目"""
        if item_id in self.working_memory_buffer:
            # 强化维持强度
            self.maintenance_strength[item_id] *= 1.1
            return self.working_memory_buffer[item_id]
        return None
    
    def update_working_memory(self, item_id: str, new_content: Any) -> bool:
        """更新工作记忆项目"""
        if item_id in self.working_memory_buffer:
            self.working_memory_buffer[item_id] = new_content
            self._update_persistent_activity(item_id, new_content)
            return True
        return False
    
    def remove_working_memory_item(self, item_id: str):
        """移除工作记忆项目"""
        if item_id in self.working_memory_buffer:
            del self.working_memory_buffer[item_id]
            del self.maintenance_strength[item_id]
            self.cognitive_load = len(self.working_memory_buffer) / self.buffer_capacity
    
    def focus_attention(self, target: str, intensity: float = 1.0):
        """聚焦注意力"""
        self.attention_focus = target
        
        # 更新注意权重
        if target in self.working_memory_buffer:
            # 增强目标项目的维持
            self.maintenance_strength[target] *= (1.0 + intensity * 0.5)
            
            # 抑制其他项目
            for item_id in self.working_memory_buffer:
                if item_id != target:
                    self.maintenance_strength[item_id] *= (1.0 - intensity * 0.2)
    
    def cognitive_control(self, task_demands: Dict[str, float]) -> Dict[str, float]:
        """认知控制"""
        control_signals = {}
        
        # 抑制控制
        if 'inhibition' in task_demands:
            inhibition_demand = task_demands['inhibition']
            control_signals['inhibition'] = self.inhibition_strength * inhibition_demand
        
        # 更新控制
        if 'updating' in task_demands:
            updating_demand = task_demands['updating']
            if updating_demand > self.updating_threshold:
                control_signals['updating'] = 1.0
            else:
                control_signals['updating'] = 0.0
        
        # 转换控制
        if 'switching' in task_demands:
            switching_demand = task_demands['switching']
            control_signals['switching'] = self.cognitive_flexibility * switching_demand
        
        return control_signals
    
    def _update_persistent_activity(self, item_id: str, content: Any):
        """更新持续活动"""
        # 简化的持续活动模型
        if isinstance(content, (int, float)):
            activity_pattern = np.random.rand(self.pfc_size) * content
        else:
            # 对于复杂内容，使用哈希生成模式
            hash_value = hash(str(content)) % self.pfc_size
            activity_pattern = np.zeros(self.pfc_size)
            activity_pattern[hash_value:hash_value+100] = 1.0
        
        # 添加到持续活动
        maintenance_strength = self.maintenance_strength.get(item_id, 1.0)
        self.persistent_activity += activity_pattern * maintenance_strength * 0.1
        
        # 归一化
        self.persistent_activity = np.clip(self.persistent_activity, 0, 2)
    
    def decay_working_memory(self, dt: float):
        """工作记忆衰减"""
        decay_rate = 0.01  # 每毫秒衰减率
        
        items_to_remove = []
        for item_id in self.maintenance_strength:
            # 衰减维持强度
            self.maintenance_strength[item_id] *= (1.0 - decay_rate * dt)
            
            # 移除过弱的项目
            if self.maintenance_strength[item_id] < 0.1:
                items_to_remove.append(item_id)
        
        for item_id in items_to_remove:
            self.remove_working_memory_item(item_id)
        
        # 衰减持续活动
        self.persistent_activity *= (1.0 - decay_rate * dt * 0.5)
    
    def get_working_memory_statistics(self) -> Dict[str, Any]:
        """获取工作记忆统计信息"""
        return {
            'buffer_size': len(self.working_memory_buffer),
            'buffer_capacity': self.buffer_capacity,
            'cognitive_load': self.cognitive_load,
            'attention_focus': self.attention_focus,
            'mean_maintenance_strength': np.mean(list(self.maintenance_strength.values())) if self.maintenance_strength else 0.0,
            'persistent_activity_level': np.mean(self.persistent_activity),
            'current_state': self.current_state.value
        }

class SensoryMotorSystem:
    """感知-运动系统"""
    
    def __init__(self):
        # 感觉输入
        self.visual_input = np.zeros(1000)
        self.auditory_input = np.zeros(500)
        self.somatosensory_input = np.zeros(800)
        
        # 运动输出
        self.motor_commands = {}
        self.motor_execution_buffer = []
        
        # 感知-动作映射
        self.sensorimotor_mappings = {}
        
        # 环境交互
        self.environment_state = {}
        self.action_history = []
        self.sensory_history = []
        
        # 预测编码
        self.sensory_predictions = {}
        self.prediction_errors = {}
        
    def process_sensory_input(self, modality: str, input_data: np.ndarray) -> np.ndarray:
        """处理感觉输入"""
        if modality == "visual":
            self.visual_input = input_data
            processed = self._process_visual_input(input_data)
        elif modality == "auditory":
            self.auditory_input = input_data
            processed = self._process_auditory_input(input_data)
        elif modality == "somatosensory":
            self.somatosensory_input = input_data
            processed = self._process_somatosensory_input(input_data)
        else:
            processed = input_data
        
        # 记录感觉历史
        self.sensory_history.append({
            'timestamp': time.time(),
            'modality': modality,
            'data': processed
        })
        
        # 计算预测误差
        if modality in self.sensory_predictions:
            prediction_error = processed - self.sensory_predictions[modality]
            self.prediction_errors[modality] = prediction_error
        
        return processed
    
    def generate_motor_command(self, action_type: str, parameters: Dict[str, Any]) -> str:
        """生成运动命令"""
        command_id = f"{action_type}_{len(self.motor_commands)}"
        
        motor_command = {
            'command_id': command_id,
            'action_type': action_type,
            'parameters': parameters,
            'timestamp': time.time(),
            'status': 'pending'
        }
        
        self.motor_commands[command_id] = motor_command
        self.motor_execution_buffer.append(command_id)
        
        return command_id
    
    def execute_motor_commands(self, environment_interface: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """执行运动命令"""
        executed_commands = []
        
        for command_id in self.motor_execution_buffer[:]:
            if command_id in self.motor_commands:
                command = self.motor_commands[command_id]
                
                # 执行命令
                if environment_interface:
                    result = environment_interface(command)
                else:
                    result = self._simulate_motor_execution(command)
                
                command['status'] = 'executed'
                command['result'] = result
                
                executed_commands.append(command)
                self.action_history.append(command)
                
                # 从缓冲区移除
                self.motor_execution_buffer.remove(command_id)
        
        return executed_commands
    
    def update_sensorimotor_mapping(self, sensory_pattern: np.ndarray, 
                                   motor_pattern: np.ndarray, learning_rate: float = 0.01):
        """更新感知-动作映射"""
        # 简化的关联学习
        mapping_key = hash(sensory_pattern.tobytes()) % 1000
        
        if mapping_key not in self.sensorimotor_mappings:
            self.sensorimotor_mappings[mapping_key] = motor_pattern.copy()
        else:
            # 增量学习
            current_mapping = self.sensorimotor_mappings[mapping_key]
            self.sensorimotor_mappings[mapping_key] = (
                (1 - learning_rate) * current_mapping + learning_rate * motor_pattern
            )
    
    def predict_sensory_consequences(self, motor_command: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """预测感觉后果（前向模型）"""
        predictions = {}
        
        # 基于动作类型预测感觉变化
        action_type = motor_command['action_type']
        
        if action_type == "reach":
            # 预测视觉和本体感觉变化
            predictions['visual'] = self.visual_input + np.random.normal(0, 0.1, len(self.visual_input))
            predictions['somatosensory'] = self.somatosensory_input + np.random.normal(0, 0.05, len(self.somatosensory_input))
        
        elif action_type == "speak":
            # 预测听觉反馈
            predictions['auditory'] = np.random.normal(0.5, 0.2, len(self.auditory_input))
        
        # 存储预测用于后续比较
        self.sensory_predictions.update(predictions)
        
        return predictions
    
    def _process_visual_input(self, input_data: np.ndarray) -> np.ndarray:
        """处理视觉输入"""
        # 简化的视觉处理：边缘检测、特征提取等
        processed = np.convolve(input_data, [-1, 0, 1], mode='same')  # 简单边缘检测
        return np.tanh(processed)
    
    def _process_auditory_input(self, input_data: np.ndarray) -> np.ndarray:
        """处理听觉输入"""
        # 简化的听觉处理：频谱分析等
        processed = np.abs(np.fft.fft(input_data))[:len(input_data)]
        return processed / np.max(processed) if np.max(processed) > 0 else processed
    
    def _process_somatosensory_input(self, input_data: np.ndarray) -> np.ndarray:
        """处理体感输入"""
        # 简化的体感处理
        processed = np.tanh(input_data * 2.0)
        return processed
    
    def _simulate_motor_execution(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """模拟运动执行"""
        action_type = command['action_type']
        parameters = command['parameters']
        
        # 模拟执行结果
        if action_type == "reach":
            success_probability = 0.9
            success = np.random.random() < success_probability
            return {
                'success': success,
                'final_position': parameters.get('target_position', [0, 0, 0]),
                'execution_time': np.random.normal(500, 50)  # ms
            }
        
        elif action_type == "grasp":
            success_probability = 0.85
            success = np.random.random() < success_probability
            return {
                'success': success,
                'grip_force': parameters.get('force', 10.0),
                'execution_time': np.random.normal(300, 30)
            }
        
        else:
            return {'success': True, 'execution_time': 100}

class IntegratedCognitiveSystem:
    """集成认知系统"""
    
    def __init__(self):
        # 子系统
        self.hippocampal_memory = HippocampalMemorySystem()
        self.basal_ganglia_decision = BasalGangliaDecisionSystem()
        self.prefrontal_working_memory = PrefrontalWorkingMemorySystem()
        self.sensorimotor_system = SensoryMotorSystem()
        
        # 系统间通信
        self.inter_system_connections = {}
        self.global_workspace = {}
        
        # 认知状态
        self.global_cognitive_state = CognitiveState.RESTING
        self.arousal_level = 0.5
        self.attention_level = 0.5
        
        # 行为循环
        self.behavior_loop_active = False
        self.current_goal = None
        self.goal_stack = []
        
        # 性能监控
        self.cognitive_events = []
        self.system_performance = {}
    
    def process_cognitive_cycle(self, sensory_input: Dict[str, np.ndarray], 
                               available_actions: List[str],
                               current_goal: Optional[str] = None) -> Dict[str, Any]:
        """处理认知循环"""
        cycle_start_time = time.time()
        
        # 1. 感觉处理
        processed_sensory = {}
        for modality, data in sensory_input.items():
            processed_sensory[modality] = self.sensorimotor_system.process_sensory_input(modality, data)
        
        # 2. 记忆检索
        if 'visual' in processed_sensory:
            retrieved_memory = self.hippocampal_memory.retrieve_episodic_memory(
                processed_sensory['visual'], threshold=0.6
            )
        else:
            retrieved_memory = None
        
        # 3. 工作记忆更新
        if current_goal:
            self.prefrontal_working_memory.encode_working_memory(
                "current_goal", current_goal, priority=2.0
            )
        
        if retrieved_memory:
            self.prefrontal_working_memory.encode_working_memory(
                f"memory_{retrieved_memory.trace_id}", retrieved_memory.content, priority=1.5
            )
        
        # 4. 决策制定
        # 构建皮层输入模式
        cortical_input = np.concatenate([
            processed_sensory.get('visual', np.zeros(100))[:100],
            processed_sensory.get('auditory', np.zeros(50))[:50],
            self.prefrontal_working_memory.persistent_activity[:100]
        ])
        
        selected_action = self.basal_ganglia_decision.process_cortical_input(
            cortical_input, available_actions
        )
        
        # 5. 动作执行
        if selected_action:
            motor_command_id = self.sensorimotor_system.generate_motor_command(
                selected_action, {'timestamp': time.time()}
            )
            
            # 预测感觉后果
            motor_command = self.sensorimotor_system.motor_commands[motor_command_id]
            predicted_consequences = self.sensorimotor_system.predict_sensory_consequences(motor_command)
        else:
            motor_command_id = None
            predicted_consequences = {}
        
        # 6. 记忆编码
        if any(processed_sensory.values()):
            context = {
                'goal': current_goal,
                'action': selected_action,
                'timestamp': time.time()
            }
            
            memory_trace_id = self.hippocampal_memory.encode_episodic_memory(
                list(processed_sensory.values())[0], context
            )
        else:
            memory_trace_id = None
        
        # 7. 更新全局工作空间
        self.global_workspace.update({
            'sensory_input': processed_sensory,
            'retrieved_memory': retrieved_memory,
            'selected_action': selected_action,
            'motor_command': motor_command_id,
            'predicted_consequences': predicted_consequences,
            'memory_trace': memory_trace_id
        })
        
        # 记录认知事件
        cognitive_event = CognitiveEvent(
            event_id=f"cycle_{len(self.cognitive_events)}",
            event_type="cognitive_cycle",
            timestamp=cycle_start_time,
            brain_region="integrated_system",
            parameters=self.global_workspace.copy(),
            duration=time.time() - cycle_start_time
        )
        
        self.cognitive_events.append(cognitive_event)
        
        return {
            'selected_action': selected_action,
            'motor_command_id': motor_command_id,
            'memory_trace_id': memory_trace_id,
            'cognitive_state': self.global_cognitive_state.value,
            'processing_time': time.time() - cycle_start_time
        }
    
    def update_systems(self, dt: float, reward: Optional[float] = None):
        """更新所有子系统"""
        # 更新记忆系统
        self.hippocampal_memory.consolidate_memories(dt)
        self.hippocampal_memory.update_oscillations(dt)
        
        # 更新决策系统
        if reward is not None:
            self.basal_ganglia_decision.update_reward_learning(reward)
        self.basal_ganglia_decision.update_synaptic_plasticity()
        
        # 更新工作记忆
        self.prefrontal_working_memory.decay_working_memory(dt)
        
        # 执行运动命令
        executed_commands = self.sensorimotor_system.execute_motor_commands()
        
        # 更新系统间连接强度
        self._update_inter_system_connections()
    
    def _update_inter_system_connections(self):
        """更新系统间连接"""
        # 海马-前额叶连接
        hippocampal_activity = np.mean(self.hippocampal_memory.ca1_activity)
        pfc_activity = np.mean(self.prefrontal_working_memory.persistent_activity)
        
        hippocampal_pfc_strength = 0.5 * (hippocampal_activity + pfc_activity)
        
        # 基底节-前额叶连接
        basal_ganglia_activity = np.mean(self.basal_ganglia_decision.gpi_activity)
        basal_ganglia_pfc_strength = 0.3 * (1.0 - basal_ganglia_activity)  # GPi抑制
        
        self.inter_system_connections = {
            'hippocampal_pfc': hippocampal_pfc_strength,
            'basal_ganglia_pfc': basal_ganglia_pfc_strength,
            'pfc_sensorimotor': pfc_activity * 0.4
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            'hippocampal_memory': self.hippocampal_memory.get_memory_statistics(),
            'basal_ganglia_decision': self.basal_ganglia_decision.get_decision_statistics(),
            'prefrontal_working_memory': self.prefrontal_working_memory.get_working_memory_statistics(),
            'inter_system_connections': self.inter_system_connections.copy(),
            'global_cognitive_state': self.global_cognitive_state.value,
            'arousal_level': self.arousal_level,
            'attention_level': self.attention_level,
            'total_cognitive_events': len(self.cognitive_events),
            'behavior_loop_active': self.behavior_loop_active
        }
    
    def export_cognitive_state(self, filepath: str):
        """导出认知状态"""
        cognitive_state = {
            'systems': self.get_system_statistics(),
            'global_workspace': self.global_workspace.copy(),
            'cognitive_events': [
                {
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'timestamp': event.timestamp,
                    'brain_region': event.brain_region,
                    'duration': event.duration
                }
                for event in self.cognitive_events[-100:]  # 最近100个事件
            ],
            'timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cognitive_state, f, indent=2, ensure_ascii=False)

def create_integrated_cognitive_system() -> IntegratedCognitiveSystem:
    """创建集成认知系统的便捷函数"""
    return IntegratedCognitiveSystem()

if __name__ == "__main__":
    # 测试集成认知系统
    logging.basicConfig(level=logging.INFO)
    
    # 创建系统
    cognitive_system = create_integrated_cognitive_system()
    
    # 模拟认知循环
    for cycle in range(10):
        # 模拟感觉输入
        sensory_input = {
            'visual': np.random.rand(1000),
            'auditory': np.random.rand(500),
            'somatosensory': np.random.rand(800)
        }
        
        # 可用动作
        available_actions = ['reach', 'grasp', 'look', 'listen', 'think']
        
        # 当前目标
        current_goal = f"goal_{cycle % 3}"
        
        # 处理认知循环
        result = cognitive_system.process_cognitive_cycle(
            sensory_input, available_actions, current_goal
        )
        
        print(f"循环 {cycle}: 选择动作 '{result['selected_action']}', "
              f"处理时间 {result['processing_time']*1000:.2f} ms")
        
        # 模拟奖励
        reward = np.random.normal(0.5, 0.2) if result['selected_action'] else 0.0
        
        # 更新系统
        cognitive_system.update_systems(100.0, reward)  # 100ms时间步
    
    # 获取统计信息
    stats = cognitive_system.get_system_statistics()
    print("\n系统统计信息:")
    print(f"  记忆数量: {stats['hippocampal_memory']['total_memories']}")
    print(f"  工作记忆负荷: {stats['prefrontal_working_memory']['cognitive_load']:.2f}")
    print(f"  多巴胺水平: {stats['basal_ganglia_decision']['dopamine_level']:.2f}")
    print(f"  认知事件总数: {stats['total_cognitive_events']}")
    
    # 导出状态
    cognitive_system.export_cognitive_state("integrated_cognitive_system.json")
    
    print("\n测试完成，认知状态已导出到 integrated_cognitive_system.json")