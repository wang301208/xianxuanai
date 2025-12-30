"""
增强的皮层柱实现

在原有基础上增加：
- 更精细的层间连接模式
- 丘脑输入的层特异性处理
- 皮层内振荡和同步
- 可塑性和学习机制
- 与丘脑环路的紧密集成
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import random
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum

from .cortical_column import EnhancedCorticalColumn, ThalamicNucleus
from .enhanced_thalamocortical_loop import ThalamocorticalLoop, ThalamicNucleusType
from .neurons import Neuron, create_neuron
from .synapses import Synapse, create_synapse


class CorticalOscillationType(Enum):
    """皮层振荡类型"""
    GAMMA = "gamma"           # γ节律 (30-100 Hz)
    BETA = "beta"             # β节律 (13-30 Hz)
    ALPHA = "alpha"           # α节律 (8-13 Hz)
    THETA = "theta"           # θ节律 (4-8 Hz)
    DELTA = "delta"           # δ节律 (1-4 Hz)


@dataclass
class CorticalOscillationState:
    """皮层振荡状态"""
    gamma_frequency: float = 40.0      # γ节律频率
    beta_frequency: float = 20.0       # β节律频率
    alpha_frequency: float = 10.0      # α节律频率
    theta_frequency: float = 6.0       # θ节律频率
    
    # 当前状态
    gamma_phase: float = 0.0
    beta_phase: float = 0.0
    alpha_phase: float = 0.0
    theta_phase: float = 0.0
    
    # 振荡强度
    gamma_amplitude: float = 0.3
    beta_amplitude: float = 0.2
    alpha_amplitude: float = 0.4
    theta_amplitude: float = 0.1
    
    # 层特异性调节
    layer_modulation: Dict[str, float] = None
    
    def __post_init__(self):
        if self.layer_modulation is None:
            self.layer_modulation = {
                'L1': 0.2,      # L1主要受α调节
                'L2/3': 0.8,    # L2/3强γ振荡
                'L4': 0.6,      # L4中等γ振荡
                'L5': 0.7,      # L5强β-γ振荡
                'L6': 0.3       # L6弱振荡
            }
    
    def update_oscillations(self, dt: float) -> Dict[str, float]:
        """更新所有振荡相位"""
        # 更新相位
        self.gamma_phase += 2 * np.pi * self.gamma_frequency * dt / 1000.0
        self.beta_phase += 2 * np.pi * self.beta_frequency * dt / 1000.0
        self.alpha_phase += 2 * np.pi * self.alpha_frequency * dt / 1000.0
        self.theta_phase += 2 * np.pi * self.theta_frequency * dt / 1000.0
        
        # 保持相位在0-2π范围内
        self.gamma_phase = self.gamma_phase % (2 * np.pi)
        self.beta_phase = self.beta_phase % (2 * np.pi)
        self.alpha_phase = self.alpha_phase % (2 * np.pi)
        self.theta_phase = self.theta_phase % (2 * np.pi)
        
        # 计算各层的振荡调节
        layer_oscillations = {}
        for layer_name, base_modulation in self.layer_modulation.items():
            # 组合多个频率的振荡
            gamma_component = self.gamma_amplitude * np.sin(self.gamma_phase)
            beta_component = self.beta_amplitude * np.sin(self.beta_phase)
            alpha_component = self.alpha_amplitude * np.sin(self.alpha_phase)
            theta_component = self.theta_amplitude * np.sin(self.theta_phase)
            
            # 层特异性权重
            if layer_name in ['L2/3', 'L4']:
                # 浅层主要γ振荡
                combined = 0.6 * gamma_component + 0.2 * beta_component + 0.2 * alpha_component
            elif layer_name == 'L5':
                # L5强β-γ振荡
                combined = 0.4 * gamma_component + 0.4 * beta_component + 0.2 * alpha_component
            elif layer_name == 'L6':
                # L6主要α-θ振荡
                combined = 0.2 * gamma_component + 0.3 * alpha_component + 0.5 * theta_component
            else:  # L1
                # L1主要α振荡
                combined = 0.1 * gamma_component + 0.7 * alpha_component + 0.2 * theta_component
            
            layer_oscillations[layer_name] = base_modulation * combined
        
        return layer_oscillations


class EnhancedCorticalColumnWithLoop(EnhancedCorticalColumn):
    """与丘脑环路集成的增强皮层柱"""
    
    def __init__(self, config: Dict[str, Any], thalamocortical_loop: Optional[ThalamocorticalLoop] = None):
        # 扩展默认配置
        enhanced_config = {
            **config,
            'oscillation_enabled': config.get('oscillation_enabled', True),
            'plasticity_enabled': config.get('plasticity_enabled', True),
            'layer_specific_dynamics': config.get('layer_specific_dynamics', True)
        }
        
        super().__init__(enhanced_config)
        
        # 丘脑环路连接
        self.thalamocortical_loop = thalamocortical_loop
        
        # 皮层振荡状态
        self.oscillation_state = CorticalOscillationState()

        # 可塑性机制（需在创建增强连接前就绪）
        self.plasticity_traces: Dict[Tuple[int, int], float] = {}
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # 层间连接增强
        self._enhance_layer_connections()
        
        # 活动历史
        self.activity_history: Dict[str, deque] = {
            layer: deque(maxlen=100) for layer in ['L1', 'L2/3', 'L4', 'L5', 'L6']
        }
        
        # 同步状态
        self.inter_layer_synchrony: Dict[Tuple[str, str], float] = {}
        
    def _enhance_layer_connections(self):
        """增强层间连接模式"""
        # 添加更精细的层间连接
        enhanced_connections = {
            # 前馈连接增强
            'L4->L2/3': 0.5,    # 增强L4到L2/3
            'L2/3->L5': 0.4,    # 增强L2/3到L5
            'L4->L5': 0.3,      # 直接L4到L5
            'L5->L6': 0.4,      # 增强L5到L6
            
            # 反馈连接增强
            'L6->L4': 0.3,      # 增强L6到L4反馈
            'L5->L2/3': 0.3,    # 增强L5到L2/3反馈
            'L6->L2/3': 0.2,    # 新增L6到L2/3反馈
            
            # 横向连接
            'L2/3->L2/3': 0.15, # 增强L2/3内连接
            'L5->L5': 0.15,     # 增强L5内连接
            'L6->L6': 0.1,      # L6内连接
            
            # 跨层抑制
            'L2/3_inh->L5': 0.2,  # L2/3抑制性到L5
            'L5_inh->L2/3': 0.2   # L5抑制性到L2/3
        }
        
        # 更新连接概率
        self.column_config['connection_probabilities'].update(enhanced_connections)
        
        # 重新创建连接
        self._create_enhanced_connections()
    
    def _create_enhanced_connections(self):
        """创建增强的层间连接"""
        for connection, probability in self.column_config['connection_probabilities'].items():
            if '->' not in connection:
                continue
            
            parts = connection.split('->')
            if len(parts) != 2:
                continue
                
            pre_layer_name, post_layer_name = parts
            
            # 处理抑制性连接
            if '_inh' in pre_layer_name:
                pre_layer_base = pre_layer_name.replace('_inh', '')
                pre_layer = self.layers.get(f'{pre_layer_base}_inh')
                post_layer = self.layers.get(f'{post_layer_name}_exc')
                
                if pre_layer and post_layer:
                    self._connect_populations_enhanced(pre_layer, post_layer, probability, 'inhibitory')
            else:
                # 标准兴奋性连接
                pre_exc = self.layers.get(f'{pre_layer_name}_exc')
                post_exc = self.layers.get(f'{post_layer_name}_exc')
                post_inh = self.layers.get(f'{post_layer_name}_inh')
                
                if pre_exc and post_exc:
                    self._connect_populations_enhanced(pre_exc, post_exc, probability, 'excitatory')
                
                if pre_exc and post_inh:
                    self._connect_populations_enhanced(pre_exc, post_inh, probability * 0.8, 'excitatory')
    
    def _connect_populations_enhanced(self, pre_layer, post_layer, probability: float, synapse_type: str):
        """增强的群体连接方法"""
        for pre_neuron in pre_layer.neurons:
            for post_neuron in post_layer.neurons:
                if random.random() < probability:
                    # 距离依赖的连接概率
                    distance_factor = self._calculate_distance_factor(pre_neuron, post_neuron)
                    adjusted_probability = probability * distance_factor
                    
                    if random.random() < adjusted_probability:
                        weight = self._sample_enhanced_weight(synapse_type, pre_layer, post_layer)
                        delay = self._sample_enhanced_delay(pre_layer, post_layer)
                        
                        # 创建可塑性突触
                        if self.column_config.get('plasticity_enabled', True):
                            synapse_class = 'stdp'
                            params = {
                                'weight': weight,
                                'delay': delay,
                                'learning_rate': self.learning_rate,
                                'tau_plus': 20.0,
                                'tau_minus': 20.0,
                                'A_plus': 0.01,
                                'A_minus': 0.012
                            }
                        else:
                            synapse_class = 'static'
                            params = {'weight': weight, 'delay': delay}
                        
                        synapse = self.add_synapse(pre_neuron.id, post_neuron.id, synapse_class, params)
                        
                        # 初始化可塑性追踪
                        if self.column_config.get('plasticity_enabled', True):
                            self.plasticity_traces[(pre_neuron.id, post_neuron.id)] = 0.0
    
    def _calculate_distance_factor(self, pre_neuron: Neuron, post_neuron: Neuron) -> float:
        """计算距离依赖因子"""
        # 简化：基于神经元ID的伪距离
        distance = abs(pre_neuron.id - post_neuron.id) / 1000.0
        return np.exp(-distance / 2.0)  # 指数衰减
    
    def _sample_enhanced_weight(self, synapse_type: str, pre_layer, post_layer) -> float:
        """采样增强的突触权重"""
        base_weight = self._sample_weight(synapse_type)
        
        # 层特异性权重调节
        layer_factors = {
            ('L4_exc', 'L2/3_exc'): 1.2,    # 增强L4到L2/3
            ('L2/3_exc', 'L5_exc'): 1.1,    # 增强L2/3到L5
            ('L6_exc', 'L4_exc'): 0.8,      # 适中的L6反馈
            ('L5_exc', 'L2/3_exc'): 0.9     # 适中的L5反馈
        }
        
        factor = layer_factors.get((pre_layer.name, post_layer.name), 1.0)
        return base_weight * factor
    
    def _sample_enhanced_delay(self, pre_layer, post_layer) -> float:
        """采样增强的突触延迟"""
        # 层间距离影响延迟
        layer_distances = {
            ('L1', 'L2/3'): 0.5,
            ('L2/3', 'L4'): 0.8,
            ('L4', 'L5'): 1.0,
            ('L5', 'L6'): 1.2,
            ('L6', 'L4'): 2.0,   # 反馈连接延迟更长
            ('L5', 'L2/3'): 1.8
        }
        
        pre_base = pre_layer.name.split('_')[0]
        post_base = post_layer.name.split('_')[0]
        
        base_delay = layer_distances.get((pre_base, post_base), 1.0)
        return base_delay + random.uniform(0.2, 0.8)
    
    def process_thalamic_input(self, thalamic_inputs: Dict[ThalamicNucleusType, np.ndarray]):
        """处理来自丘脑的输入"""
        if not self.thalamocortical_loop:
            return
        
        # 将丘脑输入分发到相应的皮层层
        input_mapping = {
            ThalamicNucleusType.VPL: ['L4', 'L6'],      # 体感输入
            ThalamicNucleusType.VPM: ['L4', 'L6'],      # 面部感觉
            ThalamicNucleusType.LGN: ['L4', 'L6'],      # 视觉输入
            ThalamicNucleusType.MGN: ['L4', 'L6'],      # 听觉输入
            ThalamicNucleusType.MD: ['L2/3', 'L5'],     # 认知输入
            ThalamicNucleusType.PULVINAR: ['L1', 'L5']  # 注意力调节
        }
        
        for nucleus_type, input_data in thalamic_inputs.items():
            if nucleus_type in input_mapping:
                target_layers = input_mapping[nucleus_type]
                
                for layer_name in target_layers:
                    target_layer = self.layers.get(f'{layer_name}_exc')
                    if target_layer and len(input_data) > 0:
                        # 将输入分发到层内神经元
                        self._distribute_thalamic_input(target_layer, input_data, nucleus_type)
    
    def _distribute_thalamic_input(self, target_layer, input_data: np.ndarray, 
                                 nucleus_type: ThalamicNucleusType):
        """将丘脑输入分发到目标层"""
        # 根据核团类型调整输入强度
        nucleus_gains = {
            ThalamicNucleusType.VPL: 1.0,
            ThalamicNucleusType.LGN: 1.2,
            ThalamicNucleusType.MD: 0.8,
            ThalamicNucleusType.PULVINAR: 0.6
        }
        
        gain = nucleus_gains.get(nucleus_type, 1.0)
        
        # 将输入数据映射到神经元
        for i, neuron in enumerate(target_layer.neurons):
            if i < len(input_data):
                input_current = input_data[i] * gain
                # 添加到神经元的输入电流
                if hasattr(neuron, '_external_current'):
                    neuron._external_current += input_current
                else:
                    neuron._external_current = input_current
    
    def step(self, dt: float) -> Dict[str, Any]:
        """增强的步进函数"""
        # 更新振荡状态
        if self.column_config.get('oscillation_enabled', True):
            layer_oscillations = self.oscillation_state.update_oscillations(dt)
        else:
            layer_oscillations = {}
        
        # 应用振荡调节到各层
        self._apply_oscillation_modulation(layer_oscillations)
        
        # 调用父类的步进函数
        result = super().step(dt)
        
        # 更新活动历史
        self._update_activity_history()
        
        # 更新可塑性
        if self.column_config.get('plasticity_enabled', True):
            self._update_plasticity(dt)
        
        # 计算层间同步
        self._calculate_inter_layer_synchrony()
        
        # 添加增强信息到结果
        result.update({
            'layer_oscillations': layer_oscillations,
            'inter_layer_synchrony': self.inter_layer_synchrony,
            'plasticity_traces': dict(list(self.plasticity_traces.items())[:10])  # 前10个追踪
        })
        
        return result
    
    def _apply_oscillation_modulation(self, layer_oscillations: Dict[str, float]):
        """应用振荡调节到各层神经元"""
        for layer_name, modulation in layer_oscillations.items():
            exc_layer = self.layers.get(f'{layer_name}_exc')
            inh_layer = self.layers.get(f'{layer_name}_inh')
            
            # 调节兴奋性神经元
            if exc_layer:
                for neuron in exc_layer.neurons:
                    if hasattr(neuron, '_oscillation_modulation'):
                        neuron._oscillation_modulation = modulation
                    else:
                        setattr(neuron, '_oscillation_modulation', modulation)
            
            # 调节抑制性神经元（相位相反）
            if inh_layer:
                for neuron in inh_layer.neurons:
                    if hasattr(neuron, '_oscillation_modulation'):
                        neuron._oscillation_modulation = -modulation * 0.5
                    else:
                        setattr(neuron, '_oscillation_modulation', -modulation * 0.5)
    
    def _update_activity_history(self):
        """更新活动历史"""
        for layer_name in ['L1', 'L2/3', 'L4', 'L5', 'L6']:
            layer_activity = self.get_layer_activity(layer_name)
            if 'excitatory' in layer_activity:
                mean_activity = layer_activity['excitatory']['mean_voltage']
                self.activity_history[layer_name].append(mean_activity)
    
    def _update_plasticity(self, dt: float):
        """更新突触可塑性"""
        # 简化的STDP更新
        for (pre_id, post_id), trace in self.plasticity_traces.items():
            if (pre_id, post_id) in self.synapses:
                synapse = self.synapses[(pre_id, post_id)]
                
                # 获取神经元活动
                pre_neuron = self.neurons.get(pre_id)
                post_neuron = self.neurons.get(post_id)
                
                if pre_neuron and post_neuron:
                    # 简化的可塑性更新
                    pre_active = pre_neuron.voltage > -55.0
                    post_active = post_neuron.voltage > -55.0
                    
                    if pre_active and post_active:
                        # LTP
                        weight_change = self.learning_rate * 0.01
                    elif pre_active or post_active:
                        # LTD
                        weight_change = -self.learning_rate * 0.005
                    else:
                        weight_change = 0.0
                    
                    # 更新权重
                    if hasattr(synapse, 'weight'):
                        new_weight = synapse.weight + weight_change
                        synapse.weight = np.clip(new_weight, -2.0, 2.0)
                    
                    # 更新追踪
                    self.plasticity_traces[(pre_id, post_id)] = trace * 0.95 + weight_change
    
    def _calculate_inter_layer_synchrony(self):
        """计算层间同步性"""
        layer_names = ['L1', 'L2/3', 'L4', 'L5', 'L6']
        
        for i in range(len(layer_names)):
            for j in range(i + 1, len(layer_names)):
                layer1, layer2 = layer_names[i], layer_names[j]
                
                if len(self.activity_history[layer1]) > 10 and len(self.activity_history[layer2]) > 10:
                    # 计算最近10个时间步的相关性
                    activity1 = list(self.activity_history[layer1])[-10:]
                    activity2 = list(self.activity_history[layer2])[-10:]
                    
                    correlation = np.corrcoef(activity1, activity2)[0, 1]
                    if np.isnan(correlation) or not np.isfinite(correlation):
                        correlation = 0.0
                    self.inter_layer_synchrony[(layer1, layer2)] = float(correlation)
    
    def get_cortical_feedback_for_thalamus(self) -> Dict[str, float]:
        """获取发送给丘脑的皮层反馈"""
        feedback = {}
        
        # L6主要提供反馈
        l6_activity = self.get_layer_activity('L6')
        if 'excitatory' in l6_activity:
            feedback['primary'] = l6_activity['excitatory']['mean_voltage']
        
        # L5提供次要反馈
        l5_activity = self.get_layer_activity('L5')
        if 'excitatory' in l5_activity:
            feedback['secondary'] = l5_activity['excitatory']['mean_voltage']
        
        return feedback
    
    def get_oscillation_coherence(self) -> Dict[str, float]:
        """获取振荡相干性"""
        coherence = {}
        
        # 计算各频段的相干性
        coherence['gamma'] = self.oscillation_state.gamma_amplitude
        coherence['beta'] = self.oscillation_state.beta_amplitude
        coherence['alpha'] = self.oscillation_state.alpha_amplitude
        coherence['theta'] = self.oscillation_state.theta_amplitude
        
        return coherence
