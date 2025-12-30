# -*- coding: utf-8 -*-
"""
可塑性学习规则系统
Plasticity Learning Rules System

实现多种可塑性机制：
1. 突触可塑性（STDP、LTP、LTD）
2. 结构可塑性（轴突生长、树突重塑）
3. 稳态可塑性（突触缩放、内在兴奋性）
4. 元可塑性（学习规则的可塑性）
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

class PlasticityType(Enum):
    """可塑性类型"""
    SYNAPTIC = "synaptic"           # 突触可塑性
    STRUCTURAL = "structural"       # 结构可塑性
    HOMEOSTATIC = "homeostatic"     # 稳态可塑性
    METAPLASTIC = "metaplastic"     # 元可塑性

class LearningRule(Enum):
    """学习规则类型"""
    STDP = "stdp"                   # 尖峰时序依赖可塑性
    BCM = "bcm"                     # BCM规则
    OJA = "oja"                     # Oja规则
    HEBBIAN = "hebbian"             # Hebb规则
    ANTI_HEBBIAN = "anti_hebbian"   # 反Hebb规则
    HOMEOSTATIC = "homeostatic"     # 稳态规则

@dataclass
class SynapticConnection:
    """突触连接"""
    pre_neuron_id: int
    post_neuron_id: int
    weight: float
    delay: float = 1.0
    plasticity_enabled: bool = True
    last_update_time: float = 0.0
    
    # STDP相关
    pre_spike_trace: float = 0.0
    post_spike_trace: float = 0.0
    
    # BCM相关
    post_activity_history: List[float] = field(default_factory=list)
    bcm_threshold: float = 1.0
    
    # 稳态相关
    target_rate: float = 5.0  # Hz
    scaling_factor: float = 1.0

@dataclass
class NeuronPlasticity:
    """神经元可塑性状态"""
    neuron_id: int
    intrinsic_excitability: float = 1.0
    firing_threshold: float = -55.0  # mV
    
    # 活动历史
    spike_times: List[float] = field(default_factory=list)
    firing_rate: float = 0.0
    
    # 稳态参数
    target_firing_rate: float = 5.0  # Hz
    homeostatic_time_constant: float = 3600.0  # 1小时
    
    # 元可塑性
    learning_rate_modulation: float = 1.0
    metaplastic_threshold: float = 10.0  # Hz

class PlasticityManager:
    """可塑性管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("PlasticityManager")
        
        # 连接和神经元
        self.synaptic_connections: Dict[Tuple[int, int], SynapticConnection] = {}
        self.neurons: Dict[int, NeuronPlasticity] = {}
        
        # 学习规则
        self.learning_rules: Dict[LearningRule, Callable] = {
            LearningRule.STDP: self._stdp_rule,
            LearningRule.BCM: self._bcm_rule,
            LearningRule.OJA: self._oja_rule,
            LearningRule.HEBBIAN: self._hebbian_rule,
            LearningRule.HOMEOSTATIC: self._homeostatic_rule
        }
        
        # 可塑性参数
        self.plasticity_params = {
            'stdp': {
                'tau_pre': 20.0,      # ms，前突触迹衰减时间常数
                'tau_post': 20.0,     # ms，后突触迹衰减时间常数
                'A_plus': 0.01,       # LTP幅度
                'A_minus': 0.012,     # LTD幅度
                'w_max': 10.0,        # 最大权重
                'w_min': 0.0          # 最小权重
            },
            'bcm': {
                'tau_theta': 10000.0, # ms，阈值时间常数
                'learning_rate': 0.001,
                'sliding_threshold': True
            },
            'homeostatic': {
                'tau_scaling': 86400000.0,  # ms，24小时
                'target_rate': 5.0,         # Hz
                'scaling_rate': 0.0001
            },
            'structural': {
                'growth_rate': 0.001,
                'pruning_threshold': 0.1,
                'max_connections_per_neuron': 1000
            }
        }
        
        # 全局调制因子
        self.neuromodulation = {
            'dopamine': 0.0,      # 奖励信号
            'acetylcholine': 0.0, # 注意力信号
            'norepinephrine': 0.0,# 觉醒信号
            'serotonin': 0.0      # 情绪信号
        }
        
        # 时间跟踪
        self.current_time = 0.0
        self.last_update_time = 0.0
        
        self._initialize_plasticity_system()
    
    def _initialize_plasticity_system(self):
        """初始化可塑性系统"""
        
        # 创建初始连接网络
        self._create_initial_network()
        
        # 初始化学习规则参数
        self._initialize_learning_parameters()
    
    def _create_initial_network(self):
        """创建初始网络"""
        
        # 创建神经元
        num_neurons = self.config.get('num_neurons', 1000)
        
        for i in range(num_neurons):
            self.neurons[i] = NeuronPlasticity(
                neuron_id=i,
                target_firing_rate=np.random.uniform(2.0, 8.0)  # 不同神经元有不同目标发放率
            )
        
        # 创建随机连接
        connection_probability = self.config.get('connection_probability', 0.1)
        
        for pre_id in range(num_neurons):
            for post_id in range(num_neurons):
                if pre_id != post_id and np.random.random() < connection_probability:
                    # 创建连接
                    initial_weight = np.random.uniform(0.1, 1.0)
                    delay = np.random.uniform(1.0, 5.0)
                    
                    connection = SynapticConnection(
                        pre_neuron_id=pre_id,
                        post_neuron_id=post_id,
                        weight=initial_weight,
                        delay=delay
                    )
                    
                    self.synaptic_connections[(pre_id, post_id)] = connection
    
    def _initialize_learning_parameters(self):
        """初始化学习参数"""
        
        # 为每个连接设置学习规则
        for connection in self.synaptic_connections.values():
            # 随机分配学习规则（可以根据连接类型定制）
            if np.random.random() < 0.7:
                connection.learning_rule = LearningRule.STDP
            elif np.random.random() < 0.9:
                connection.learning_rule = LearningRule.BCM
            else:
                connection.learning_rule = LearningRule.HOMEOSTATIC
    
    def update_plasticity(self, dt: float, spike_data: Dict[int, List[float]], 
                         neuromodulation: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """更新可塑性"""
        
        self.current_time += dt
        
        if neuromodulation:
            self.neuromodulation.update(neuromodulation)
        
        # 更新神经元状态
        neuron_updates = self._update_neurons(dt, spike_data)
        
        # 更新突触可塑性
        synaptic_updates = self._update_synaptic_plasticity(dt, spike_data)
        
        # 更新结构可塑性
        structural_updates = self._update_structural_plasticity(dt)
        
        # 更新稳态可塑性
        homeostatic_updates = self._update_homeostatic_plasticity(dt)
        
        # 更新元可塑性
        metaplastic_updates = self._update_metaplasticity(dt)
        
        self.last_update_time = self.current_time
        
        return {
            'neuron_updates': neuron_updates,
            'synaptic_updates': synaptic_updates,
            'structural_updates': structural_updates,
            'homeostatic_updates': homeostatic_updates,
            'metaplastic_updates': metaplastic_updates,
            'network_statistics': self._compute_network_statistics()
        }
    
    def _update_neurons(self, dt: float, spike_data: Dict[int, List[float]]) -> Dict[int, Dict[str, Any]]:
        """更新神经元状态"""
        
        neuron_updates = {}
        
        for neuron_id, neuron in self.neurons.items():
            # 更新尖峰历史
            if neuron_id in spike_data:
                new_spikes = spike_data[neuron_id]
                neuron.spike_times.extend(new_spikes)
                
                # 保持尖峰历史在合理范围内
                cutoff_time = self.current_time - 1000.0  # 保留1秒历史
                neuron.spike_times = [t for t in neuron.spike_times if t > cutoff_time]
            
            # 计算发放率
            if len(neuron.spike_times) > 1:
                time_window = 100.0  # ms
                recent_spikes = [t for t in neuron.spike_times 
                               if t > self.current_time - time_window]
                neuron.firing_rate = len(recent_spikes) * 1000.0 / time_window  # Hz
            else:
                neuron.firing_rate = 0.0
            
            neuron_updates[neuron_id] = {
                'firing_rate': neuron.firing_rate,
                'intrinsic_excitability': neuron.intrinsic_excitability,
                'num_recent_spikes': len([t for t in neuron.spike_times 
                                        if t > self.current_time - 100.0])
            }
        
        return neuron_updates
    
    def _update_synaptic_plasticity(self, dt: float, spike_data: Dict[int, List[float]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """更新突触可塑性"""
        
        synaptic_updates = {}
        
        for (pre_id, post_id), connection in self.synaptic_connections.items():
            if not connection.plasticity_enabled:
                continue
            
            # 更新突触迹
            self._update_synaptic_traces(connection, dt, spike_data)
            
            # 应用学习规则
            if hasattr(connection, 'learning_rule'):
                learning_rule = connection.learning_rule
                if learning_rule in self.learning_rules:
                    weight_change = self.learning_rules[learning_rule](
                        connection, dt, spike_data
                    )
                    
                    # 应用神经调质调制
                    modulated_change = self._apply_neuromodulation(weight_change, learning_rule)
                    
                    # 更新权重
                    old_weight = connection.weight
                    connection.weight += modulated_change
                    
                    # 权重限制
                    connection.weight = np.clip(
                        connection.weight,
                        self.plasticity_params['stdp']['w_min'],
                        self.plasticity_params['stdp']['w_max']
                    )
                    
                    synaptic_updates[(pre_id, post_id)] = {
                        'old_weight': old_weight,
                        'new_weight': connection.weight,
                        'weight_change': modulated_change,
                        'learning_rule': learning_rule.value
                    }
        
        return synaptic_updates
    
    def _update_synaptic_traces(self, connection: SynapticConnection, dt: float, 
                              spike_data: Dict[int, List[float]]):
        """更新突触迹"""
        
        tau_pre = self.plasticity_params['stdp']['tau_pre']
        tau_post = self.plasticity_params['stdp']['tau_post']
        
        # 衰减突触迹
        connection.pre_spike_trace *= np.exp(-dt / tau_pre)
        connection.post_spike_trace *= np.exp(-dt / tau_post)
        
        # 添加新的尖峰贡献
        if connection.pre_neuron_id in spike_data:
            for spike_time in spike_data[connection.pre_neuron_id]:
                if spike_time > self.last_update_time:
                    connection.pre_spike_trace += 1.0
        
        if connection.post_neuron_id in spike_data:
            for spike_time in spike_data[connection.post_neuron_id]:
                if spike_time > self.last_update_time:
                    connection.post_spike_trace += 1.0
    
    def _stdp_rule(self, connection: SynapticConnection, dt: float, 
                   spike_data: Dict[int, List[float]]) -> float:
        """STDP学习规则"""
        
        params = self.plasticity_params['stdp']
        weight_change = 0.0
        
        # 检查新的尖峰对
        pre_spikes = spike_data.get(connection.pre_neuron_id, [])
        post_spikes = spike_data.get(connection.post_neuron_id, [])
        
        # 只考虑当前时间步的新尖峰
        new_pre_spikes = [t for t in pre_spikes if t > self.last_update_time]
        new_post_spikes = [t for t in post_spikes if t > self.last_update_time]
        
        # LTP: 前突触先发放
        for pre_time in new_pre_spikes:
            # 使用后突触迹（表示最近的后突触活动）
            if connection.post_spike_trace > 0:
                weight_change += params['A_plus'] * connection.post_spike_trace
        
        # LTD: 后突触先发放
        for post_time in new_post_spikes:
            # 使用前突触迹（表示最近的前突触活动）
            if connection.pre_spike_trace > 0:
                weight_change -= params['A_minus'] * connection.pre_spike_trace
        
        return weight_change
    
    def _bcm_rule(self, connection: SynapticConnection, dt: float, 
                  spike_data: Dict[int, List[float]]) -> float:
        """BCM学习规则"""
        
        params = self.plasticity_params['bcm']
        
        # 获取后突触神经元活动
        post_neuron = self.neurons[connection.post_neuron_id]
        post_activity = post_neuron.firing_rate
        
        # 更新活动历史
        connection.post_activity_history.append(post_activity)
        if len(connection.post_activity_history) > 1000:  # 保持历史长度
            connection.post_activity_history.pop(0)
        
        # 计算滑动阈值
        if params['sliding_threshold'] and len(connection.post_activity_history) > 10:
            connection.bcm_threshold = np.mean(connection.post_activity_history) ** 2
        
        # BCM规则: Δw = η * x * y * (y - θ)
        pre_activity = connection.pre_spike_trace  # 使用前突触迹作为前突触活动
        
        weight_change = (params['learning_rate'] * 
                        pre_activity * 
                        post_activity * 
                        (post_activity - connection.bcm_threshold))
        
        return weight_change
    
    def _oja_rule(self, connection: SynapticConnection, dt: float, 
                  spike_data: Dict[int, List[float]]) -> float:
        """Oja学习规则"""
        
        learning_rate = 0.001
        
        # 获取前后突触活动
        pre_activity = connection.pre_spike_trace
        post_activity = connection.post_spike_trace
        
        # Oja规则: Δw = η * y * (x - w * y)
        weight_change = (learning_rate * 
                        post_activity * 
                        (pre_activity - connection.weight * post_activity))
        
        return weight_change
    
    def _hebbian_rule(self, connection: SynapticConnection, dt: float, 
                      spike_data: Dict[int, List[float]]) -> float:
        """Hebb学习规则"""
        
        learning_rate = 0.001
        
        # 简单Hebb规则: Δw = η * x * y
        pre_activity = connection.pre_spike_trace
        post_activity = connection.post_spike_trace
        
        weight_change = learning_rate * pre_activity * post_activity
        
        return weight_change
    
    def _homeostatic_rule(self, connection: SynapticConnection, dt: float, 
                         spike_data: Dict[int, List[float]]) -> float:
        """稳态学习规则"""
        
        params = self.plasticity_params['homeostatic']
        
        # 获取后突触神经元
        post_neuron = self.neurons[connection.post_neuron_id]
        
        # 计算发放率偏差
        rate_error = post_neuron.firing_rate - params['target_rate']
        
        # 稳态缩放: 如果发放率过高，减小权重；如果过低，增大权重
        weight_change = -params['scaling_rate'] * rate_error * connection.weight * dt / 1000.0
        
        return weight_change
    
    def _apply_neuromodulation(self, weight_change: float, learning_rule: LearningRule) -> float:
        """应用神经调质调制"""
        
        modulation_factor = 1.0
        
        # 多巴胺调制（主要影响奖励相关学习）
        if learning_rule == LearningRule.STDP:
            # 多巴胺增强LTP，抑制LTD
            if weight_change > 0:  # LTP
                modulation_factor *= (1.0 + self.neuromodulation['dopamine'])
            else:  # LTD
                modulation_factor *= (1.0 - 0.5 * self.neuromodulation['dopamine'])
        
        # 乙酰胆碱调制（影响注意力相关学习）
        if learning_rule in [LearningRule.STDP, LearningRule.BCM]:
            modulation_factor *= (1.0 + 0.5 * self.neuromodulation['acetylcholine'])
        
        # 去甲肾上腺素调制（影响觉醒状态）
        modulation_factor *= (1.0 + 0.3 * self.neuromodulation['norepinephrine'])
        
        return weight_change * modulation_factor
    
    def _update_structural_plasticity(self, dt: float) -> Dict[str, Any]:
        """更新结构可塑性"""
        
        params = self.plasticity_params['structural']
        
        structural_changes = {
            'new_connections': [],
            'pruned_connections': [],
            'connection_strength_changes': {}
        }
        
        # 连接生长
        growth_candidates = self._identify_growth_candidates()
        for pre_id, post_id in growth_candidates:
            if np.random.random() < params['growth_rate'] * dt / 1000.0:
                # 创建新连接
                new_connection = SynapticConnection(
                    pre_neuron_id=pre_id,
                    post_neuron_id=post_id,
                    weight=np.random.uniform(0.1, 0.5),
                    delay=np.random.uniform(1.0, 5.0)
                )
                
                self.synaptic_connections[(pre_id, post_id)] = new_connection
                structural_changes['new_connections'].append((pre_id, post_id))
        
        # 连接修剪
        pruning_candidates = self._identify_pruning_candidates(params['pruning_threshold'])
        for pre_id, post_id in pruning_candidates:
            if np.random.random() < params['growth_rate'] * dt / 1000.0:
                del self.synaptic_connections[(pre_id, post_id)]
                structural_changes['pruned_connections'].append((pre_id, post_id))
        
        return structural_changes
    
    def _identify_growth_candidates(self) -> List[Tuple[int, int]]:
        """识别生长候选连接"""
        
        candidates = []
        max_connections = self.plasticity_params['structural']['max_connections_per_neuron']
        
        for pre_id in self.neurons.keys():
            # 计算当前连接数
            current_connections = sum(1 for (p, _) in self.synaptic_connections.keys() if p == pre_id)
            
            if current_connections < max_connections:
                # 寻找高活动的后突触候选
                for post_id in self.neurons.keys():
                    if pre_id != post_id and (pre_id, post_id) not in self.synaptic_connections:
                        # 基于活动相关性决定生长概率
                        pre_activity = self.neurons[pre_id].firing_rate
                        post_activity = self.neurons[post_id].firing_rate
                        
                        if pre_activity > 2.0 and post_activity > 2.0:  # 都有一定活动
                            candidates.append((pre_id, post_id))
        
        return candidates
    
    def _identify_pruning_candidates(self, threshold: float) -> List[Tuple[int, int]]:
        """识别修剪候选连接"""
        
        candidates = []
        
        for (pre_id, post_id), connection in self.synaptic_connections.items():
            # 基于权重强度和使用频率决定修剪
            if connection.weight < threshold:
                # 检查最近的使用情况
                pre_neuron = self.neurons[pre_id]
                if pre_neuron.firing_rate < 0.5:  # 很少使用
                    candidates.append((pre_id, post_id))
        
        return candidates
    
    def _update_homeostatic_plasticity(self, dt: float) -> Dict[int, Dict[str, Any]]:
        """更新稳态可塑性"""
        
        homeostatic_updates = {}
        params = self.plasticity_params['homeostatic']
        
        for neuron_id, neuron in self.neurons.items():
            # 计算发放率偏差
            rate_error = neuron.firing_rate - neuron.target_firing_rate
            
            # 内在兴奋性调节
            excitability_change = -0.0001 * rate_error * dt / 1000.0
            neuron.intrinsic_excitability += excitability_change
            neuron.intrinsic_excitability = np.clip(neuron.intrinsic_excitability, 0.1, 2.0)
            
            # 阈值调节
            threshold_change = 0.01 * rate_error * dt / 1000.0
            neuron.firing_threshold += threshold_change
            neuron.firing_threshold = np.clip(neuron.firing_threshold, -70.0, -40.0)
            
            # 突触缩放
            scaling_change = -params['scaling_rate'] * rate_error * dt / 1000.0
            
            # 应用到所有入射连接
            for (pre_id, post_id), connection in self.synaptic_connections.items():
                if post_id == neuron_id:
                    connection.scaling_factor += scaling_change
                    connection.scaling_factor = np.clip(connection.scaling_factor, 0.1, 2.0)
            
            homeostatic_updates[neuron_id] = {
                'rate_error': rate_error,
                'excitability_change': excitability_change,
                'threshold_change': threshold_change,
                'scaling_change': scaling_change,
                'new_excitability': neuron.intrinsic_excitability,
                'new_threshold': neuron.firing_threshold
            }
        
        return homeostatic_updates
    
    def _update_metaplasticity(self, dt: float) -> Dict[int, Dict[str, Any]]:
        """更新元可塑性"""
        
        metaplastic_updates = {}
        
        for neuron_id, neuron in self.neurons.items():
            # 基于历史活动调节学习率
            if neuron.firing_rate > neuron.metaplastic_threshold:
                # 高活动时降低学习率（防止过度学习）
                neuron.learning_rate_modulation *= 0.999
            else:
                # 低活动时提高学习率（促进学习）
                neuron.learning_rate_modulation *= 1.001
            
            neuron.learning_rate_modulation = np.clip(neuron.learning_rate_modulation, 0.1, 2.0)
            
            # 调节相关连接的学习参数
            for (pre_id, post_id), connection in self.synaptic_connections.items():
                if post_id == neuron_id:
                    # 应用元可塑性调制到学习规则
                    if hasattr(connection, 'learning_rate_modulation'):
                        connection.learning_rate_modulation = neuron.learning_rate_modulation
            
            metaplastic_updates[neuron_id] = {
                'firing_rate': neuron.firing_rate,
                'learning_rate_modulation': neuron.learning_rate_modulation,
                'above_threshold': neuron.firing_rate > neuron.metaplastic_threshold
            }
        
        return metaplastic_updates
    
    def _compute_network_statistics(self) -> Dict[str, Any]:
        """计算网络统计"""
        
        # 权重统计
        weights = [conn.weight for conn in self.synaptic_connections.values()]
        
        # 发放率统计
        firing_rates = [neuron.firing_rate for neuron in self.neurons.values()]
        
        # 连接统计
        num_connections = len(self.synaptic_connections)
        
        # 可塑性活跃度
        plastic_connections = sum(1 for conn in self.synaptic_connections.values() 
                                if conn.plasticity_enabled)
        
        return {
            'weight_statistics': {
                'mean': np.mean(weights) if weights else 0.0,
                'std': np.std(weights) if weights else 0.0,
                'min': np.min(weights) if weights else 0.0,
                'max': np.max(weights) if weights else 0.0
            },
            'firing_rate_statistics': {
                'mean': np.mean(firing_rates) if firing_rates else 0.0,
                'std': np.std(firing_rates) if firing_rates else 0.0,
                'min': np.min(firing_rates) if firing_rates else 0.0,
                'max': np.max(firing_rates) if firing_rates else 0.0
            },
            'connectivity_statistics': {
                'total_connections': num_connections,
                'plastic_connections': plastic_connections,
                'connection_density': num_connections / (len(self.neurons) ** 2) if self.neurons else 0.0
            },
            'neuromodulation_levels': self.neuromodulation.copy()
        }
    
    def apply_learning_protocol(self, protocol_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """应用学习协议"""
        
        if protocol_name == "spike_timing_protocol":
            return self._spike_timing_protocol(parameters)
        elif protocol_name == "frequency_protocol":
            return self._frequency_protocol(parameters)
        elif protocol_name == "paired_pulse_protocol":
            return self._paired_pulse_protocol(parameters)
        elif protocol_name == "homeostatic_challenge":
            return self._homeostatic_challenge_protocol(parameters)
        else:
            self.logger.warning(f"Unknown learning protocol: {protocol_name}")
            return {}
    
    def _spike_timing_protocol(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """尖峰时序协议"""
        
        pre_neuron_id = parameters['pre_neuron_id']
        post_neuron_id = parameters['post_neuron_id']
        time_difference = parameters['time_difference']  # ms
        num_pairs = parameters.get('num_pairs', 60)
        frequency = parameters.get('frequency', 1.0)  # Hz
        
        results = []
        
        for i in range(num_pairs):
            # 生成尖峰对
            base_time = self.current_time + i * (1000.0 / frequency)
            
            if time_difference > 0:  # 前突触先发放
                pre_spike_time = base_time
                post_spike_time = base_time + time_difference
            else:  # 后突触先发放
                post_spike_time = base_time
                pre_spike_time = base_time - time_difference
            
            # 模拟尖峰
            spike_data = {
                pre_neuron_id: [pre_spike_time],
                post_neuron_id: [post_spike_time]
            }
            
            # 更新可塑性
            update_result = self.update_plasticity(1.0, spike_data)
            
            # 记录权重变化
            if (pre_neuron_id, post_neuron_id) in self.synaptic_connections:
                connection = self.synaptic_connections[(pre_neuron_id, post_neuron_id)]
                results.append({
                    'pair_number': i,
                    'weight': connection.weight,
                    'time_difference': time_difference
                })
        
        return {
            'protocol': 'spike_timing_protocol',
            'parameters': parameters,
            'results': results
        }
    
    def _frequency_protocol(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """频率协议"""
        
        neuron_ids = parameters['neuron_ids']
        frequency = parameters['frequency']  # Hz
        duration = parameters['duration']  # ms
        
        # 生成规律尖峰序列
        spike_interval = 1000.0 / frequency
        num_spikes = int(duration / spike_interval)
        
        spike_data = {}
        for neuron_id in neuron_ids:
            spike_times = [self.current_time + i * spike_interval for i in range(num_spikes)]
            spike_data[neuron_id] = spike_times
        
        # 更新可塑性
        update_result = self.update_plasticity(duration, spike_data)
        
        return {
            'protocol': 'frequency_protocol',
            'parameters': parameters,
            'update_result': update_result
        }
    
    def _paired_pulse_protocol(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """配对脉冲协议"""
        
        neuron_id = parameters['neuron_id']
        inter_pulse_interval = parameters['inter_pulse_interval']  # ms
        num_pairs = parameters.get('num_pairs', 10)
        
        results = []
        
        for i in range(num_pairs):
            base_time = self.current_time + i * 1000.0  # 每秒一对
            
            spike_data = {
                neuron_id: [base_time, base_time + inter_pulse_interval]
            }
            
            update_result = self.update_plasticity(inter_pulse_interval + 10.0, spike_data)
            results.append(update_result)
        
        return {
            'protocol': 'paired_pulse_protocol',
            'parameters': parameters,
            'results': results
        }
    
    def _homeostatic_challenge_protocol(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """稳态挑战协议"""
        
        # 暂时改变目标发放率
        original_targets = {}
        new_target = parameters['new_target_rate']
        duration = parameters['duration']
        
        # 保存原始目标并设置新目标
        for neuron_id, neuron in self.neurons.items():
            original_targets[neuron_id] = neuron.target_firing_rate
            neuron.target_firing_rate = new_target
        
        # 运行一段时间
        update_results = []
        time_steps = int(duration / 100.0)  # 每100ms更新一次
        
        for step in range(time_steps):
            # 生成随机活动
            spike_data = {}
            for neuron_id in self.neurons.keys():
                if np.random.random() < 0.1:  # 10%概率发放
                    spike_data[neuron_id] = [self.current_time]
            
            update_result = self.update_plasticity(100.0, spike_data)
            update_results.append(update_result)
        
        # 恢复原始目标
        for neuron_id, original_target in original_targets.items():
            self.neurons[neuron_id].target_firing_rate = original_target
        
        return {
            'protocol': 'homeostatic_challenge',
            'parameters': parameters,
            'update_results': update_results
        }
    
    def get_plasticity_state(self) -> Dict[str, Any]:
        """获取可塑性状态"""
        
        return {
            'current_time': self.current_time,
            'num_neurons': len(self.neurons),
            'num_connections': len(self.synaptic_connections),
            'neuromodulation': self.neuromodulation.copy(),
            'network_statistics': self._compute_network_statistics()
        }
    
    def save_plasticity_state(self, filepath: str):
        """保存可塑性状态"""
        
        import pickle
        
        state = {
            'neurons': self.neurons,
            'synaptic_connections': self.synaptic_connections,
            'current_time': self.current_time,
            'neuromodulation': self.neuromodulation,
            'plasticity_params': self.plasticity_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_plasticity_state(self, filepath: str):
        """加载可塑性状态"""
        
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.neurons = state['neurons']
        self.synaptic_connections = state['synaptic_connections']
        self.current_time = state['current_time']
        self.neuromodulation = state['neuromodulation']
        self.plasticity_params = state['plasticity_params']

# 工厂函数
def create_plasticity_manager(config: Optional[Dict[str, Any]] = None) -> PlasticityManager:
    """创建可塑性管理器"""
    if config is None:
        config = {
            'num_neurons': 1000,
            'connection_probability': 0.1
        }
    
    return PlasticityManager(config)