# -*- coding: utf-8 -*-
"""
认知与行为模块深化 - 具体脑区映射
Cognitive and Behavioral Module Enhancement - Specific Brain Region Mapping

将抽象认知功能映射到具体脑区，实现：
1. 海马-前额叶记忆回路
2. 基底节-皮层动作选择
3. 脑干/丘脑调制系统
4. 多模态感觉输入整合
5. 身体模型与运动控制
6. 可塑性学习规则
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

class CognitiveFunction(Enum):
    """认知功能枚举"""
    WORKING_MEMORY = "working_memory"
    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"
    ATTENTION = "attention"
    EXECUTIVE_CONTROL = "executive_control"
    DECISION_MAKING = "decision_making"
    ACTION_SELECTION = "action_selection"
    MOTOR_PLANNING = "motor_planning"
    SENSORY_INTEGRATION = "sensory_integration"
    EMOTIONAL_PROCESSING = "emotional_processing"
    LANGUAGE_PROCESSING = "language_processing"
    SPATIAL_NAVIGATION = "spatial_navigation"

class BrainCircuit(Enum):
    """脑回路枚举"""
    HIPPOCAMPAL_PFC = "hippocampal_pfc"
    BASAL_GANGLIA_CORTICAL = "basal_ganglia_cortical"
    THALAMO_CORTICAL = "thalamo_cortical"
    BRAINSTEM_MODULATION = "brainstem_modulation"
    CORTICO_CEREBELLAR = "cortico_cerebellar"
    LIMBIC_CORTICAL = "limbic_cortical"
    SENSORIMOTOR = "sensorimotor"

@dataclass
class CognitiveBrainMapping:
    """认知功能到脑区的映射"""
    cognitive_function: CognitiveFunction
    primary_regions: List[str]
    supporting_regions: List[str]
    circuit_type: BrainCircuit
    neurotransmitters: List[str]
    oscillation_bands: List[str]
    plasticity_mechanisms: List[str]

class HippocampalPFCCircuit:
    """海马-前额叶记忆回路"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("HippocampalPFCCircuit")
        
        # 回路组件
        self.hippocampus = {
            'DG': {'neurons': {}, 'activity': 0.0, 'theta_phase': 0.0},
            'CA3': {'neurons': {}, 'activity': 0.0, 'gamma_power': 0.0},
            'CA1': {'neurons': {}, 'activity': 0.0, 'sharp_waves': []},
        }
        
        self.prefrontal_cortex = {
            'dlPFC': {'neurons': {}, 'working_memory': [], 'maintenance_activity': 0.0},
            'vmPFC': {'neurons': {}, 'value_signals': 0.0, 'decision_confidence': 0.0},
            'ACC': {'neurons': {}, 'conflict_monitoring': 0.0, 'error_signals': []},
        }
        
        # 连接权重矩阵
        self.connectivity_matrix = np.zeros((100, 100))  # 简化矩阵
        
        # 记忆编码/提取状态
        self.memory_state = {
            'encoding_mode': False,
            'retrieval_mode': False,
            'consolidation_strength': 0.0,
            'pattern_separation': 0.0,
            'pattern_completion': 0.0
        }
        
        # 神经振荡
        self.oscillations = {
            'theta': {'frequency': 8.0, 'amplitude': 0.0, 'phase': 0.0},
            'gamma': {'frequency': 40.0, 'amplitude': 0.0, 'phase': 0.0},
            'ripples': {'events': [], 'power': 0.0}
        }
        
        self._initialize_circuit()
    
    def _initialize_circuit(self):
        """初始化回路结构"""
        # 初始化海马各区域
        for region in self.hippocampus:
            neuron_count = {'DG': 200, 'CA3': 150, 'CA1': 180}[region]
            for i in range(neuron_count):
                self.hippocampus[region]['neurons'][i] = {
                    'membrane_potential': -70.0,
                    'firing_rate': 0.0,
                    'place_field': np.random.uniform(0, 100, 2),  # 位置场
                    'theta_preference': np.random.uniform(0, 2*np.pi),
                    'plasticity_trace': 0.0
                }
        
        # 初始化前额叶各区域
        for region in self.prefrontal_cortex:
            neuron_count = {'dlPFC': 300, 'vmPFC': 200, 'ACC': 150}[region]
            for i in range(neuron_count):
                self.prefrontal_cortex[region]['neurons'][i] = {
                    'membrane_potential': -70.0,
                    'firing_rate': 0.0,
                    'selectivity': np.random.choice(['spatial', 'temporal', 'abstract']),
                    'maintenance_strength': 0.0,
                    'plasticity_trace': 0.0
                }
        
        # 建立连接
        self._establish_connectivity()
    
    def _establish_connectivity(self):
        """建立海马-前额叶连接"""
        # CA1 -> dlPFC (记忆信息传递)
        ca1_neurons = len(self.hippocampus['CA1']['neurons'])
        dlpfc_neurons = len(self.prefrontal_cortex['dlPFC']['neurons'])
        
        # 稀疏连接矩阵
        connection_prob = 0.1
        for i in range(ca1_neurons):
            for j in range(dlpfc_neurons):
                if np.random.random() < connection_prob:
                    weight = np.random.normal(0.5, 0.1)
                    self.connectivity_matrix[i, j] = weight
    
    def encode_memory(self, stimulus: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """记忆编码过程"""
        self.memory_state['encoding_mode'] = True
        
        # 齿状回模式分离
        dg_activity = self._pattern_separation(stimulus)
        self.hippocampus['DG']['activity'] = dg_activity
        
        # CA3自联想网络
        ca3_pattern = self._auto_associative_recall(dg_activity, context)
        self.hippocampus['CA3']['activity'] = ca3_pattern
        
        # CA1序列编码
        ca1_sequence = self._sequence_encoding(ca3_pattern, context)
        self.hippocampus['CA1']['activity'] = ca1_sequence
        
        # 前额叶工作记忆维持
        wm_maintenance = self._working_memory_maintenance(ca1_sequence)
        self.prefrontal_cortex['dlPFC']['maintenance_activity'] = wm_maintenance
        
        # 更新连接权重（长时程增强）
        self._update_synaptic_weights(stimulus, context)
        
        return {
            'encoding_success': True,
            'pattern_separation_strength': dg_activity,
            'associative_strength': ca3_pattern,
            'sequence_strength': ca1_sequence,
            'working_memory_load': wm_maintenance
        }
    
    def retrieve_memory(self, cue: Dict[str, Any]) -> Dict[str, Any]:
        """记忆提取过程"""
        self.memory_state['retrieval_mode'] = True
        
        # CA3模式完成
        retrieved_pattern = self._pattern_completion(cue)
        self.hippocampus['CA3']['activity'] = retrieved_pattern
        
        # CA1序列重放
        sequence_replay = self._sequence_replay(retrieved_pattern)
        self.hippocampus['CA1']['sharp_waves'].append(sequence_replay)
        
        # 前额叶检索监控
        retrieval_confidence = self._retrieval_monitoring(sequence_replay)
        self.prefrontal_cortex['vmPFC']['decision_confidence'] = retrieval_confidence
        
        return {
            'retrieval_success': retrieved_pattern > 0.5,
            'pattern_completion_strength': retrieved_pattern,
            'replay_fidelity': sequence_replay,
            'confidence': retrieval_confidence
        }
    
    def _pattern_separation(self, stimulus: Dict[str, Any]) -> float:
        """齿状回模式分离"""
        # 稀疏编码实现模式分离
        input_vector = np.array(list(stimulus.values())[:10])  # 取前10个特征
        
        # 竞争性学习
        dg_neurons = list(self.hippocampus['DG']['neurons'].values())
        activations = []
        
        for neuron in dg_neurons[:50]:  # 取前50个神经元
            similarity = np.dot(input_vector, np.random.random(len(input_vector)))
            activation = max(0, similarity - 0.7)  # 阈值激活
            activations.append(activation)
        
        # 稀疏化（只有少数神经元激活）
        threshold = np.percentile(activations, 90)
        sparse_activity = np.mean([a for a in activations if a > threshold])
        
        return min(1.0, sparse_activity)
    
    def _auto_associative_recall(self, dg_input: float, context: Dict[str, Any]) -> float:
        """CA3自联想回忆"""
        # 递归网络动力学
        ca3_state = dg_input * 0.8  # DG输入权重
        
        # 上下文调制
        context_strength = len(context) / 10.0
        ca3_state += context_strength * 0.2
        
        # 自联想增强
        for _ in range(5):  # 迭代收敛
            ca3_state = np.tanh(ca3_state * 1.2)
        
        return min(1.0, ca3_state)
    
    def _sequence_encoding(self, ca3_input: float, context: Dict[str, Any]) -> float:
        """CA1序列编码"""
        # 时间序列编码
        temporal_weight = context.get('temporal_order', 0.5)
        sequence_strength = ca3_input * temporal_weight
        
        # Theta相位编码
        theta_phase = self.oscillations['theta']['phase']
        phase_modulation = np.cos(theta_phase) * 0.3
        
        return min(1.0, sequence_strength + phase_modulation)
    
    def _working_memory_maintenance(self, hippocampal_input: float) -> float:
        """前额叶工作记忆维持"""
        # 持续激活维持
        current_wm = self.prefrontal_cortex['dlPFC']['maintenance_activity']
        
        # 新信息整合
        integration_rate = 0.3
        new_wm = current_wm * (1 - integration_rate) + hippocampal_input * integration_rate
        
        # 容量限制
        capacity_limit = 0.8
        return min(capacity_limit, new_wm)
    
    def _pattern_completion(self, cue: Dict[str, Any]) -> float:
        """CA3模式完成"""
        # 部分线索激活完整模式
        cue_strength = len(cue) / 10.0
        
        # 自联想网络恢复
        completion_strength = cue_strength
        for _ in range(3):
            completion_strength = np.tanh(completion_strength * 1.5)
        
        return min(1.0, completion_strength)
    
    def _sequence_replay(self, pattern_strength: float) -> float:
        """CA1序列重放"""
        # 尖波涟漪事件
        if pattern_strength > 0.6:
            replay_fidelity = pattern_strength * np.random.uniform(0.8, 1.0)
            return replay_fidelity
        return 0.0
    
    def _retrieval_monitoring(self, replay_strength: float) -> float:
        """前额叶提取监控"""
        # 元认知监控
        confidence = replay_strength
        
        # 冲突检测
        conflict = abs(replay_strength - 0.5) * 2  # 距离0.5越远冲突越小
        confidence *= (1 + conflict * 0.2)
        
        return min(1.0, confidence)
    
    def _update_synaptic_weights(self, stimulus: Dict[str, Any], context: Dict[str, Any]):
        """更新突触权重"""
        # 赫布学习规则
        learning_rate = 0.01
        
        # 海马内部可塑性
        for region in self.hippocampus:
            for neuron in self.hippocampus[region]['neurons'].values():
                if neuron['firing_rate'] > 0.5:
                    neuron['plasticity_trace'] += learning_rate
                else:
                    neuron['plasticity_trace'] *= 0.99  # 衰减
    
    def update_oscillations(self, dt: float):
        """更新神经振荡"""
        # Theta节律 (海马)
        theta = self.oscillations['theta']
        theta['phase'] += 2 * np.pi * theta['frequency'] * dt / 1000.0
        theta['phase'] = theta['phase'] % (2 * np.pi)
        
        # 基于活动调制振幅
        hipp_activity = np.mean([self.hippocampus[r]['activity'] for r in self.hippocampus])
        theta['amplitude'] = hipp_activity * 0.8
        
        # Gamma节律 (皮层)
        gamma = self.oscillations['gamma']
        gamma['phase'] += 2 * np.pi * gamma['frequency'] * dt / 1000.0
        gamma['phase'] = gamma['phase'] % (2 * np.pi)
        
        pfc_activity = np.mean([self.prefrontal_cortex[r]['maintenance_activity'] 
                               for r in self.prefrontal_cortex])
        gamma['amplitude'] = pfc_activity * 0.6
    
    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """更新回路状态"""
        # 更新振荡
        self.update_oscillations(dt)
        
        # 处理输入
        results = {}
        
        if 'encode' in inputs:
            results['encoding'] = self.encode_memory(
                inputs['encode']['stimulus'], 
                inputs['encode']['context']
            )
        
        if 'retrieve' in inputs:
            results['retrieval'] = self.retrieve_memory(inputs['retrieve']['cue'])
        
        # 自发活动和巩固
        if not self.memory_state['encoding_mode'] and not self.memory_state['retrieval_mode']:
            consolidation = self._memory_consolidation(dt)
            results['consolidation'] = consolidation
        
        # 重置模式标志
        self.memory_state['encoding_mode'] = False
        self.memory_state['retrieval_mode'] = False
        
        return results
    
    def _memory_consolidation(self, dt: float) -> Dict[str, Any]:
        """记忆巩固过程"""
        # 系统巩固：海马到皮层的信息转移
        consolidation_rate = 0.001 * dt
        
        # 海马重放驱动皮层可塑性
        replay_events = len(self.hippocampus['CA1']['sharp_waves'])
        if replay_events > 0:
            # 增强海马-皮层连接
            for i in range(min(50, self.connectivity_matrix.shape[0])):
                for j in range(min(50, self.connectivity_matrix.shape[1])):
                    if self.connectivity_matrix[i, j] > 0:
                        self.connectivity_matrix[i, j] += consolidation_rate
            
            # 清除已处理的重放事件
            self.hippocampus['CA1']['sharp_waves'] = []
        
        return {
            'consolidation_strength': consolidation_rate * replay_events,
            'replay_events': replay_events
        }

class BasalGangliaCircuit:
    """基底节-皮层动作选择回路"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("BasalGangliaCircuit")
        
        # 基底节核团
        self.nuclei = {
            'striatum': {
                'direct_pathway': {'neurons': {}, 'activity': 0.0, 'dopamine_level': 1.0},
                'indirect_pathway': {'neurons': {}, 'activity': 0.0, 'dopamine_level': 1.0}
            },
            'gpe': {'neurons': {}, 'activity': 0.0, 'inhibition_strength': 0.0},
            'gpi_snr': {'neurons': {}, 'activity': 0.0, 'tonic_inhibition': 0.8},
            'stn': {'neurons': {}, 'activity': 0.0, 'hyperdirect_input': 0.0},
            'snc_vta': {'neurons': {}, 'dopamine_release': 1.0, 'reward_prediction': 0.0}
        }
        
        # 皮层区域
        self.cortical_areas = {
            'motor_cortex': {'neurons': {}, 'action_plans': [], 'execution_threshold': 0.7},
            'premotor_cortex': {'neurons': {}, 'movement_preparation': 0.0},
            'supplementary_motor': {'neurons': {}, 'sequence_planning': 0.0}
        }
        
        # 动作候选
        self.action_candidates = {}
        self.selected_action = None
        
        # 学习参数
        self.learning_params = {
            'reward_prediction_error': 0.0,
            'td_error': 0.0,
            'learning_rate': 0.01
        }
        
        self._initialize_circuit()
    
    def _initialize_circuit(self):
        """初始化基底节回路"""
        # 初始化纹状体
        for pathway in ['direct_pathway', 'indirect_pathway']:
            for i in range(200):
                self.nuclei['striatum'][pathway]['neurons'][i] = {
                    'membrane_potential': -70.0,
                    'firing_rate': 0.0,
                    'dopamine_sensitivity': np.random.uniform(0.5, 1.5),
                    'action_preference': np.random.randint(0, 10),  # 10个可能动作
                    'synaptic_weights': np.random.uniform(0, 1, 50)  # 皮层输入权重
                }
        
        # 初始化其他核团
        for nucleus in ['gpe', 'gpi_snr', 'stn', 'snc_vta']:
            neuron_count = {'gpe': 100, 'gpi_snr': 150, 'stn': 80, 'snc_vta': 60}[nucleus]
            for i in range(neuron_count):
                self.nuclei[nucleus]['neurons'][i] = {
                    'membrane_potential': -70.0,
                    'firing_rate': 0.0,
                    'baseline_activity': np.random.uniform(0.1, 0.5)
                }
        
        # 初始化皮层区域
        for area in self.cortical_areas:
            neuron_count = {'motor_cortex': 300, 'premotor_cortex': 250, 'supplementary_motor': 200}[area]
            for i in range(neuron_count):
                self.cortical_areas[area]['neurons'][i] = {
                    'membrane_potential': -70.0,
                    'firing_rate': 0.0,
                    'action_tuning': np.random.randint(0, 10),
                    'movement_direction': np.random.uniform(0, 2*np.pi)
                }
    
    def action_selection(self, action_values: Dict[int, float], context: Dict[str, Any]) -> Dict[str, Any]:
        """动作选择过程"""
        # 更新动作候选
        self.action_candidates = action_values
        
        # 皮层输入到纹状体
        cortical_input = self._compute_cortical_input(action_values, context)
        
        # 直接通路激活
        direct_activity = self._direct_pathway_activation(cortical_input, action_values)
        
        # 间接通路抑制
        indirect_activity = self._indirect_pathway_activation(cortical_input, action_values)
        
        # 超直接通路（皮层-STN）
        hyperdirect_activity = self._hyperdirect_pathway(cortical_input)
        
        # 基底节输出计算
        bg_output = self._compute_basal_ganglia_output(
            direct_activity, indirect_activity, hyperdirect_activity
        )
        
        # 动作选择（赢者通吃）
        selected_action_id = self._winner_take_all(bg_output)
        self.selected_action = selected_action_id
        
        # 执行信号到运动皮层
        execution_signal = self._generate_execution_signal(selected_action_id, bg_output)
        
        return {
            'selected_action': selected_action_id,
            'action_value': action_values.get(selected_action_id, 0.0),
            'execution_strength': execution_signal,
            'direct_pathway_activity': direct_activity,
            'indirect_pathway_activity': indirect_activity,
            'basal_ganglia_output': bg_output
        }
    
    def _compute_cortical_input(self, action_values: Dict[int, float], context: Dict[str, Any]) -> Dict[int, float]:
        """计算皮层输入"""
        cortical_input = {}
        
        for action_id, value in action_values.items():
            # 基础激活
            base_activation = value
            
            # 上下文调制
            context_modulation = context.get('urgency', 1.0) * context.get('confidence', 1.0)
            
            # 运动准备
            motor_preparation = self.cortical_areas['premotor_cortex']['movement_preparation']
            
            cortical_input[action_id] = base_activation * context_modulation + motor_preparation * 0.2
        
        return cortical_input
    
    def _direct_pathway_activation(self, cortical_input: Dict[int, float], action_values: Dict[int, float]) -> Dict[int, float]:
        """直接通路激活"""
        direct_activity = {}
        dopamine_level = self.nuclei['striatum']['direct_pathway']['dopamine_level']
        
        for action_id, cortical_strength in cortical_input.items():
            # 多巴胺增强直接通路
            da_modulation = 1.0 + (dopamine_level - 1.0) * 0.5
            
            # 纹状体神经元激活
            striatal_activation = cortical_strength * da_modulation
            
            # 直接抑制GPi/SNr
            direct_activity[action_id] = striatal_activation
        
        return direct_activity
    
    def _indirect_pathway_activation(self, cortical_input: Dict[int, float], action_values: Dict[int, float]) -> Dict[int, float]:
        """间接通路激活"""
        indirect_activity = {}
        dopamine_level = self.nuclei['striatum']['indirect_pathway']['dopamine_level']
        
        for action_id, cortical_strength in cortical_input.items():
            # 多巴胺抑制间接通路
            da_modulation = 1.0 - (dopamine_level - 1.0) * 0.5
            
            # 纹状体激活GPe
            striatal_to_gpe = cortical_strength * da_modulation
            
            # GPe抑制GPi/SNr（双重抑制=去抑制）
            gpe_inhibition = striatal_to_gpe * 0.8
            indirect_disinhibition = 1.0 - gpe_inhibition
            
            indirect_activity[action_id] = indirect_disinhibition
        
        return indirect_activity
    
    def _hyperdirect_pathway(self, cortical_input: Dict[int, float]) -> float:
        """超直接通路（全局抑制）"""
        # 皮层直接激活STN
        total_cortical = sum(cortical_input.values())
        stn_activation = min(1.0, total_cortical * 0.3)
        
        # STN激活GPi/SNr（全局抑制）
        hyperdirect_inhibition = stn_activation * 0.6
        
        return hyperdirect_inhibition
    
    def _compute_basal_ganglia_output(self, direct: Dict[int, float], 
                                    indirect: Dict[int, float], 
                                    hyperdirect: float) -> Dict[int, float]:
        """计算基底节输出"""
        bg_output = {}
        baseline_inhibition = self.nuclei['gpi_snr']['tonic_inhibition']
        
        for action_id in direct.keys():
            # 基线抑制
            total_inhibition = baseline_inhibition
            
            # 直接通路减少抑制
            total_inhibition -= direct[action_id] * 0.6
            
            # 间接通路增加抑制
            total_inhibition += (1.0 - indirect[action_id]) * 0.4
            
            # 超直接通路全局抑制
            total_inhibition += hyperdirect * 0.3
            
            # 输出=去抑制强度
            bg_output[action_id] = max(0.0, 1.0 - total_inhibition)
        
        return bg_output
    
    def _winner_take_all(self, bg_output: Dict[int, float]) -> Optional[int]:
        """赢者通吃选择"""
        if not bg_output:
            return None
        
        # 找到最大激活的动作
        max_action = max(bg_output.keys(), key=lambda x: bg_output[x])
        max_value = bg_output[max_action]
        
        # 检查是否超过执行阈值
        threshold = self.cortical_areas['motor_cortex']['execution_threshold']
        if max_value > threshold:
            return max_action
        
        return None
    
    def _generate_execution_signal(self, action_id: Optional[int], bg_output: Dict[int, float]) -> float:
        """生成执行信号"""
        if action_id is None:
            return 0.0
        
        # 基底节去抑制强度
        disinhibition = bg_output.get(action_id, 0.0)
        
        # 运动皮层激活
        execution_strength = disinhibition * 1.2
        
        return min(1.0, execution_strength)
    
    def reward_learning(self, reward: float, predicted_reward: float) -> Dict[str, Any]:
        """奖励学习（时间差分学习）"""
        # 计算预测误差
        td_error = reward - predicted_reward
        self.learning_params['td_error'] = td_error
        self.learning_params['reward_prediction_error'] = td_error
        
        # 多巴胺释放
        baseline_da = 1.0
        da_response = baseline_da + td_error * 0.5
        da_response = max(0.1, min(2.0, da_response))  # 限制范围
        
        # 更新多巴胺水平
        self.nuclei['striatum']['direct_pathway']['dopamine_level'] = da_response
        self.nuclei['striatum']['indirect_pathway']['dopamine_level'] = da_response
        self.nuclei['snc_vta']['dopamine_release'] = da_response
        
        # 更新动作价值（强化学习）
        if self.selected_action is not None:
            learning_rate = self.learning_params['learning_rate']
            
            # 更新纹状体突触权重
            for pathway in ['direct_pathway', 'indirect_pathway']:
                for neuron in self.nuclei['striatum'][pathway]['neurons'].values():
                    if neuron['action_preference'] == self.selected_action:
                        # 多巴胺调制的可塑性
                        plasticity_factor = (da_response - 1.0) * learning_rate
                        neuron['synaptic_weights'] += plasticity_factor * 0.1
                        neuron['synaptic_weights'] = np.clip(neuron['synaptic_weights'], 0, 2)
        
        return {
            'td_error': td_error,
            'dopamine_level': da_response,
            'learning_occurred': self.selected_action is not None
        }
    
    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """更新基底节回路"""
        results = {}
        
        # 动作选择
        if 'action_values' in inputs:
            selection_result = self.action_selection(
                inputs['action_values'], 
                inputs.get('context', {})
            )
            results['action_selection'] = selection_result
        
        # 奖励学习
        if 'reward' in inputs and 'predicted_reward' in inputs:
            learning_result = self.reward_learning(
                inputs['reward'], 
                inputs['predicted_reward']
            )
            results['reward_learning'] = learning_result
        
        # 运动准备更新
        if 'movement_cue' in inputs:
            preparation_strength = inputs['movement_cue'].get('strength', 0.0)
            self.cortical_areas['premotor_cortex']['movement_preparation'] = preparation_strength
        
        return results

class BrainstemModulationSystem:
    """脑干/丘脑调制系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("BrainstemModulationSystem")
        
        # 脑干核团
        self.brainstem_nuclei = {
            'locus_coeruleus': {  # 蓝斑核（去甲肾上腺素）
                'neurons': {},
                'norepinephrine_level': 1.0,
                'arousal_state': 0.5,
                'attention_modulation': 0.0
            },
            'raphe_nuclei': {  # 中缝核（血清素）
                'neurons': {},
                'serotonin_level': 1.0,
                'mood_state': 0.5,
                'sleep_wake_cycle': 0.5
            },
            'vta': {  # 腹侧被盖区（多巴胺）
                'neurons': {},
                'dopamine_level': 1.0,
                'motivation_level': 0.5,
                'reward_expectation': 0.0
            },
            'pedunculopontine': {  # 脚桥核（乙酰胆碱）
                'neurons': {},
                'acetylcholine_level': 1.0,
                'locomotion_state': 0.0,
                'rem_sleep_control': 0.0
            }
        }
        
        # 丘脑核团
        self.thalamic_nuclei = {
            'reticular_nucleus': {  # 网状核
                'neurons': {},
                'gating_strength': 0.5,
                'attention_filter': 0.0,
                'sleep_spindles': []
            },
            'intralaminar': {  # 板内核群
                'neurons': {},
                'arousal_projection': 0.0,
                'consciousness_level': 0.8
            },
            'midline': {  # 中线核群
                'neurons': {},
                'memory_modulation': 0.0,
                'emotional_regulation': 0.0
            }
        }
        
        # 全脑调制状态
        self.global_state = {
            'arousal_level': 0.5,
            'attention_focus': 0.5,
            'emotional_valence': 0.0,
            'sleep_stage': 'awake',
            'circadian_phase': 0.0
        }
        
        # 神经递质浓度
        self.neurotransmitter_levels = {
            'norepinephrine': 1.0,
            'serotonin': 1.0,
            'dopamine': 1.0,
            'acetylcholine': 1.0,
            'gaba': 1.0,
            'glutamate': 1.0
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """初始化调制系统"""
        # 初始化脑干核团
        for nucleus in self.brainstem_nuclei:
            neuron_count = {
                'locus_coeruleus': 50,
                'raphe_nuclei': 80,
                'vta': 100,
                'pedunculopontine': 60
            }[nucleus]
            
            for i in range(neuron_count):
                self.brainstem_nuclei[nucleus]['neurons'][i] = {
                    'membrane_potential': -70.0,
                    'firing_rate': np.random.uniform(0.1, 0.5),
                    'baseline_activity': np.random.uniform(0.2, 0.8),
                    'modulation_strength': np.random.uniform(0.5, 1.5)
                }
        
        # 初始化丘脑核团
        for nucleus in self.thalamic_nuclei:
            neuron_count = {
                'reticular_nucleus': 200,
                'intralaminar': 150,
                'midline': 100
            }[nucleus]
            
            for i in range(neuron_count):
                self.thalamic_nuclei[nucleus]['neurons'][i] = {
                    'membrane_potential': -70.0,
                    'firing_rate': np.random.uniform(0.1, 0.3),
                    'gating_threshold': np.random.uniform(0.3, 0.7),
                    'projection_strength': np.random.uniform(0.5, 1.0)
                }
    
    def arousal_modulation(self, stimulus_intensity: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """觉醒调制"""
        # 蓝斑核去甲肾上腺素系统
        lc_activity = self._locus_coeruleus_response(stimulus_intensity, context)
        
        # 更新觉醒水平
        arousal_change = (lc_activity - self.global_state['arousal_level']) * 0.1
        self.global_state['arousal_level'] += arousal_change
        self.global_state['arousal_level'] = np.clip(self.global_state['arousal_level'], 0.0, 1.0)
        
        # 去甲肾上腺素释放
        ne_release = lc_activity * 1.2
        self.neurotransmitter_levels['norepinephrine'] = ne_release
        self.brainstem_nuclei['locus_coeruleus']['norepinephrine_level'] = ne_release
        
        # 全脑调制效应
        attention_enhancement = self._compute_attention_modulation(ne_release)
        memory_consolidation = self._compute_memory_modulation(ne_release)
        
        return {
            'arousal_level': self.global_state['arousal_level'],
            'norepinephrine_level': ne_release,
            'attention_enhancement': attention_enhancement,
            'memory_consolidation': memory_consolidation,
            'locus_coeruleus_activity': lc_activity
        }
    
    def _locus_coeruleus_response(self, stimulus_intensity: float, context: Dict[str, Any]) -> float:
        """蓝斑核响应"""
        # 基础活动
        baseline = 0.3
        
        # 刺激强度响应
        stimulus_response = np.tanh(stimulus_intensity * 2.0) * 0.4
        
        # 上下文调制
        novelty = context.get('novelty', 0.0)
        threat = context.get('threat', 0.0)
        uncertainty = context.get('uncertainty', 0.0)
        
        context_modulation = (novelty + threat + uncertainty) / 3.0 * 0.3
        
        total_activity = baseline + stimulus_response + context_modulation
        return np.clip(total_activity, 0.0, 1.0)
    
    def _compute_attention_modulation(self, ne_level: float) -> float:
        """计算注意力调制"""
        # 倒U型曲线：中等水平的去甲肾上腺素最优
        optimal_level = 1.0
        deviation = abs(ne_level - optimal_level)
        attention_efficiency = 1.0 - deviation * 0.5
        
        return max(0.0, attention_efficiency)
    
    def _compute_memory_modulation(self, ne_level: float) -> float:
        """计算记忆调制"""
        # 去甲肾上腺素增强记忆巩固
        consolidation_strength = ne_level * 0.8
        return min(1.0, consolidation_strength)
    
    def mood_regulation(self, emotional_input: Dict[str, float]) -> Dict[str, Any]:
        """情绪调节（血清素系统）"""
        # 中缝核血清素响应
        serotonin_activity = self._raphe_nuclei_response(emotional_input)
        
        # 更新情绪状态
        valence_change = emotional_input.get('valence', 0.0) * 0.1
        self.global_state['emotional_valence'] += valence_change
        self.global_state['emotional_valence'] = np.clip(self.global_state['emotional_valence'], -1.0, 1.0)
        
        # 血清素释放
        serotonin_level = serotonin_activity * 1.1
        self.neurotransmitter_levels['serotonin'] = serotonin_level
        self.brainstem_nuclei['raphe_nuclei']['serotonin_level'] = serotonin_level
        
        # 情绪调节效应
        mood_stabilization = self._compute_mood_stabilization(serotonin_level)
        impulse_control = self._compute_impulse_control(serotonin_level)
        
        return {
            'emotional_valence': self.global_state['emotional_valence'],
            'serotonin_level': serotonin_level,
            'mood_stabilization': mood_stabilization,
            'impulse_control': impulse_control,
            'raphe_activity': serotonin_activity
        }
    
    def _raphe_nuclei_response(self, emotional_input: Dict[str, float]) -> float:
        """中缝核响应"""
        # 基础活动
        baseline = 0.4
        
        # 情绪输入响应
        valence = emotional_input.get('valence', 0.0)
        arousal = emotional_input.get('arousal', 0.0)
        
        # 负性情绪降低血清素活动
        valence_effect = valence * 0.2
        arousal_effect = arousal * 0.1
        
        total_activity = baseline + valence_effect + arousal_effect
        return np.clip(total_activity, 0.1, 1.0)
    
    def _compute_mood_stabilization(self, serotonin_level: float) -> float:
        """计算情绪稳定化"""
        # 血清素促进情绪稳定
        stabilization = serotonin_level * 0.9
        return min(1.0, stabilization)
    
    def _compute_impulse_control(self, serotonin_level: float) -> float:
        """计算冲动控制"""
        # 血清素增强冲动控制
        control_strength = serotonin_level * 0.8
        return min(1.0, control_strength)
    
    def sleep_wake_regulation(self, circadian_input: float, homeostatic_pressure: float) -> Dict[str, Any]:
        """睡眠-觉醒调节"""
        # 更新昼夜节律相位
        self.global_state['circadian_phase'] = circadian_input
        
        # 计算睡眠倾向
        sleep_propensity = self._compute_sleep_propensity(circadian_input, homeostatic_pressure)
        
        # 确定睡眠阶段
        sleep_stage = self._determine_sleep_stage(sleep_propensity)
        self.global_state['sleep_stage'] = sleep_stage
        
        # 脚桥核乙酰胆碱调制
        ach_modulation = self._pedunculopontine_modulation(sleep_stage)
        
        # 丘脑网状核门控
        thalamic_gating = self._thalamic_gating(sleep_stage)
        
        return {
            'sleep_stage': sleep_stage,
            'sleep_propensity': sleep_propensity,
            'acetylcholine_level': ach_modulation,
            'thalamic_gating': thalamic_gating,
            'circadian_phase': circadian_input
        }
    
    def _compute_sleep_propensity(self, circadian: float, homeostatic: float) -> float:
        """计算睡眠倾向"""
        # 昼夜节律成分（余弦函数）
        circadian_drive = (np.cos(circadian * 2 * np.pi) + 1) / 2
        
        # 稳态压力成分
        homeostatic_drive = homeostatic
        
        # 综合睡眠倾向
        sleep_propensity = (circadian_drive + homeostatic_drive) / 2
        return np.clip(sleep_propensity, 0.0, 1.0)
    
    def _determine_sleep_stage(self, sleep_propensity: float) -> str:
        """确定睡眠阶段"""
        if sleep_propensity < 0.3:
            return 'awake'
        elif sleep_propensity < 0.5:
            return 'drowsy'
        elif sleep_propensity < 0.7:
            return 'nrem_light'
        elif sleep_propensity < 0.9:
            return 'nrem_deep'
        else:
            return 'rem'
    
    def _pedunculopontine_modulation(self, sleep_stage: str) -> float:
        """脚桥核调制"""
        # 不同睡眠阶段的乙酰胆碱水平
        ach_levels = {
            'awake': 1.0,
            'drowsy': 0.8,
            'nrem_light': 0.4,
            'nrem_deep': 0.2,
            'rem': 1.2  # REM期间乙酰胆碱水平高
        }
        
        ach_level = ach_levels.get(sleep_stage, 1.0)
        self.neurotransmitter_levels['acetylcholine'] = ach_level
        
        return ach_level
    
    def _thalamic_gating(self, sleep_stage: str) -> float:
        """丘脑门控"""
        # 不同睡眠阶段的门控强度
        gating_strengths = {
            'awake': 0.2,      # 低门控，信息自由流动
            'drowsy': 0.4,
            'nrem_light': 0.6,
            'nrem_deep': 0.9,  # 高门控，阻断感觉输入
            'rem': 0.3         # 中等门控，允许内部活动
        }
        
        gating_strength = gating_strengths.get(sleep_stage, 0.2)
        self.thalamic_nuclei['reticular_nucleus']['gating_strength'] = gating_strength
        
        return gating_strength
    
    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """更新调制系统"""
        results = {}
        
        # 觉醒调制
        if 'stimulus_intensity' in inputs:
            arousal_result = self.arousal_modulation(
                inputs['stimulus_intensity'],
                inputs.get('context', {})
            )
            results['arousal'] = arousal_result
        
        # 情绪调节
        if 'emotional_input' in inputs:
            mood_result = self.mood_regulation(inputs['emotional_input'])
            results['mood'] = mood_result
        
        # 睡眠-觉醒调节
        if 'circadian_input' in inputs and 'homeostatic_pressure' in inputs:
            sleep_result = self.sleep_wake_regulation(
                inputs['circadian_input'],
                inputs['homeostatic_pressure']
            )
            results['sleep_wake'] = sleep_result
        
        # 更新全局神经递质水平
        self._update_global_neurotransmitters(dt)
        results['neurotransmitters'] = self.neurotransmitter_levels.copy()
        
        return results
    
    def _update_global_neurotransmitters(self, dt: float):
        """更新全局神经递质水平"""
        # 神经递质清除和重摄取
        clearance_rates = {
            'norepinephrine': 0.1,
            'serotonin': 0.05,
            'dopamine': 0.15,
            'acetylcholine': 0.2,
            'gaba': 0.3,
            'glutamate': 0.4
        }
        
        for nt, rate in clearance_rates.items():
            # 指数衰减
            self.neurotransmitter_levels[nt] *= np.exp(-rate * dt / 1000.0)
            # 维持最小基础水平
            self.neurotransmitter_levels[nt] = max(0.1, self.neurotransmitter_levels[nt])

class CognitiveBrainIntegrator:
    """认知-脑区整合器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("CognitiveBrainIntegrator")
        
        # 初始化各个回路
        self.hippocampal_pfc = HippocampalPFCCircuit(config)
        self.basal_ganglia = BasalGangliaCircuit(config)
        self.brainstem_modulation = BrainstemModulationSystem(config)
        
        # 认知功能映射
        self.cognitive_mappings = self._create_cognitive_mappings()
        
        # 跨回路交互
        self.circuit_interactions = {
            'memory_action': 0.0,      # 记忆对动作选择的影响
            'arousal_memory': 0.0,     # 觉醒对记忆的调制
            'emotion_decision': 0.0,   # 情绪对决策的影响
        }
        
        # 全局认知状态
        self.cognitive_state = {
            'attention_level': 0.5,
            'working_memory_load': 0.0,
            'cognitive_flexibility': 0.5,
            'executive_control': 0.5,
            'emotional_regulation': 0.5
        }
    
    def _create_cognitive_mappings(self) -> Dict[CognitiveFunction, CognitiveBrainMapping]:
        """创建认知功能到脑区的映射"""
        mappings = {}
        
        # 工作记忆
        mappings[CognitiveFunction.WORKING_MEMORY] = CognitiveBrainMapping(
            cognitive_function=CognitiveFunction.WORKING_MEMORY,
            primary_regions=['dlPFC', 'parietal_cortex'],
            supporting_regions=['ACC', 'thalamus'],
            circuit_type=BrainCircuit.HIPPOCAMPAL_PFC,
            neurotransmitters=['dopamine', 'glutamate'],
            oscillation_bands=['gamma', 'theta'],
            plasticity_mechanisms=['NMDA_dependent', 'dopamine_modulated']
        )
        
        # 情景记忆
        mappings[CognitiveFunction.EPISODIC_MEMORY] = CognitiveBrainMapping(
            cognitive_function=CognitiveFunction.EPISODIC_MEMORY,
            primary_regions=['hippocampus', 'medial_temporal_lobe'],
            supporting_regions=['retrosplenial_cortex', 'angular_gyrus'],
            circuit_type=BrainCircuit.HIPPOCAMPAL_PFC,
            neurotransmitters=['acetylcholine', 'glutamate'],
            oscillation_bands=['theta', 'gamma', 'ripples'],
            plasticity_mechanisms=['LTP', 'LTD', 'metaplasticity']
        )
        
        # 动作选择
        mappings[CognitiveFunction.ACTION_SELECTION] = CognitiveBrainMapping(
            cognitive_function=CognitiveFunction.ACTION_SELECTION,
            primary_regions=['striatum', 'motor_cortex'],
            supporting_regions=['STN', 'GPe', 'GPi'],
            circuit_type=BrainCircuit.BASAL_GANGLIA_CORTICAL,
            neurotransmitters=['dopamine', 'GABA'],
            oscillation_bands=['beta', 'gamma'],
            plasticity_mechanisms=['dopamine_dependent', 'corticostriatal']
        )
        
        # 注意力
        mappings[CognitiveFunction.ATTENTION] = CognitiveBrainMapping(
            cognitive_function=CognitiveFunction.ATTENTION,
            primary_regions=['parietal_cortex', 'frontal_eye_fields'],
            supporting_regions=['locus_coeruleus', 'thalamus'],
            circuit_type=BrainCircuit.THALAMO_CORTICAL,
            neurotransmitters=['norepinephrine', 'acetylcholine'],
            oscillation_bands=['alpha', 'gamma'],
            plasticity_mechanisms=['attention_dependent', 'neuromodulator_gated']
        )
        
        return mappings
    
    def process_cognitive_task(self, task_type: CognitiveFunction, 
                             task_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理认知任务"""
        
        # 获取任务映射
        mapping = self.cognitive_mappings.get(task_type)
        if not mapping:
            return {'error': f'Unknown cognitive function: {task_type}'}
        
        results = {'task_type': task_type.value, 'mapping': mapping}
        
        # 根据任务类型调用相应回路
        if mapping.circuit_type == BrainCircuit.HIPPOCAMPAL_PFC:
            circuit_result = self._process_memory_task(task_inputs)
            results['circuit_output'] = circuit_result
            
        elif mapping.circuit_type == BrainCircuit.BASAL_GANGLIA_CORTICAL:
            circuit_result = self._process_action_task(task_inputs)
            results['circuit_output'] = circuit_result
            
        elif mapping.circuit_type == BrainCircuit.BRAINSTEM_MODULATION:
            circuit_result = self._process_modulation_task(task_inputs)
            results['circuit_output'] = circuit_result
        
        # 更新认知状态
        self._update_cognitive_state(task_type, circuit_result)
        results['cognitive_state'] = self.cognitive_state.copy()
        
        return results
    
    def _process_memory_task(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理记忆任务"""
        return self.hippocampal_pfc.update(0.1, inputs)
    
    def _process_action_task(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理动作任务"""
        return self.basal_ganglia.update(0.1, inputs)
    
    def _process_modulation_task(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理调制任务"""
        return self.brainstem_modulation.update(0.1, inputs)
    
    def _update_cognitive_state(self, task_type: CognitiveFunction, circuit_result: Dict[str, Any]):
        """更新认知状态"""
        
        if task_type == CognitiveFunction.WORKING_MEMORY:
            if 'encoding' in circuit_result:
                wm_load = circuit_result['encoding'].get('working_memory_load', 0.0)
                self.cognitive_state['working_memory_load'] = wm_load
        
        elif task_type == CognitiveFunction.ACTION_SELECTION:
            if 'action_selection' in circuit_result:
                execution_strength = circuit_result['action_selection'].get('execution_strength', 0.0)
                self.cognitive_state['executive_control'] = execution_strength
        
        elif task_type == CognitiveFunction.ATTENTION:
            if 'arousal' in circuit_result:
                attention_level = circuit_result['arousal'].get('attention_enhancement', 0.5)
                self.cognitive_state['attention_level'] = attention_level
    
    def cross_circuit_interaction(self, dt: float) -> Dict[str, Any]:
        """跨回路交互"""
        
        # 记忆对动作选择的影响
        memory_confidence = getattr(self.hippocampal_pfc, 'memory_state', {}).get('consolidation_strength', 0.0)
        self.circuit_interactions['memory_action'] = memory_confidence * 0.3
        
        # 觉醒对记忆的调制
        arousal_level = self.brainstem_modulation.global_state['arousal_level']
        self.circuit_interactions['arousal_memory'] = arousal_level * 0.4
        
        # 情绪对决策的影响
        emotional_valence = self.brainstem_modulation.global_state['emotional_valence']
        self.circuit_interactions['emotion_decision'] = abs(emotional_valence) * 0.2
        
        return {
            'interactions': self.circuit_interactions.copy(),
            'global_coherence': np.mean(list(self.circuit_interactions.values()))
        }
    
    def update(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """更新整合器"""
        
        results = {
            'circuits': {},
            'interactions': {},
            'cognitive_state': {}
        }
        
        # 更新各个回路
        if 'hippocampal_pfc' in inputs:
            results['circuits']['hippocampal_pfc'] = self.hippocampal_pfc.update(dt, inputs['hippocampal_pfc'])
        
        if 'basal_ganglia' in inputs:
            results['circuits']['basal_ganglia'] = self.basal_ganglia.update(dt, inputs['basal_ganglia'])
        
        if 'brainstem_modulation' in inputs:
            results['circuits']['brainstem_modulation'] = self.brainstem_modulation.update(dt, inputs['brainstem_modulation'])
        
        # 跨回路交互
        interaction_result = self.cross_circuit_interaction(dt)
        results['interactions'] = interaction_result
        
        # 认知状态
        results['cognitive_state'] = self.cognitive_state.copy()
        
        return results

# 工厂函数
def create_cognitive_brain_integrator(config: Optional[Dict[str, Any]] = None) -> CognitiveBrainIntegrator:
    """创建认知-脑区整合器"""
    if config is None:
        config = {}
    
    return CognitiveBrainIntegrator(config)