"""
增强的丘脑-皮层环路系统

实现完整的丘脑-皮层-丘脑闭环连接，包括：
- 多个丘脑核团（VPL, VPM, MD, LGN等）
- 皮层柱的6层结构与丘脑的双向连接
- 丘脑内抑制网络（网状核）
- 振荡模式和注意力调节
- 睡眠-觉醒状态调节
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import random
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging

from .neurons import Neuron, create_neuron
from .synapses import Synapse, create_synapse
from .network import NeuralNetwork, Layer


class ThalamicNucleusType(Enum):
    """丘脑核团类型"""
    VPL = "ventral_posterior_lateral"      # 腹后外侧核（体感）
    VPM = "ventral_posterior_medial"       # 腹后内侧核（面部感觉）
    LGN = "lateral_geniculate"             # 外侧膝状体（视觉）
    MGN = "medial_geniculate"              # 内侧膝状体（听觉）
    MD = "mediodorsal"                     # 背内侧核（认知）
    VA = "ventral_anterior"                # 腹前核（运动）
    VL = "ventral_lateral"                 # 腹外侧核（运动）
    LP = "lateral_posterior"               # 后外侧核（视觉关联）
    PULVINAR = "pulvinar"                  # 枕核（注意力）
    RETICULAR = "reticular"                # 网状核（抑制调节）


@dataclass
class ThalamicOscillationState:
    """丘脑振荡状态"""
    # 基本振荡参数
    alpha_frequency: float = 10.0          # α节律频率 (8-12 Hz)
    spindle_frequency: float = 12.0        # 睡眠纺锤波频率 (11-15 Hz)
    delta_frequency: float = 2.0           # δ节律频率 (1-4 Hz)
    
    # 当前状态
    current_phase: float = 0.0             # 当前相位
    oscillation_amplitude: float = 1.0     # 振荡幅度
    
    # 状态调节
    arousal_level: float = 0.8             # 觉醒水平 (0-1)
    attention_focus: float = 0.5           # 注意力聚焦度 (0-1)
    
    # 睡眠状态
    sleep_stage: int = 0                   # 0=觉醒, 1=N1, 2=N2, 3=N3, 4=REM
    spindle_active: bool = False           # 纺锤波是否激活
    
    def update_oscillation(self, dt: float) -> float:
        """更新振荡状态并返回调节因子"""
        # 根据觉醒水平选择主导频率
        if self.arousal_level > 0.7:
            # 觉醒状态：α节律
            dominant_freq = self.alpha_frequency
            self.oscillation_amplitude = 0.3 + 0.4 * self.attention_focus
        elif self.arousal_level > 0.3:
            # 浅睡眠：纺锤波
            dominant_freq = self.spindle_frequency
            self.oscillation_amplitude = 0.6
            self.spindle_active = True
        else:
            # 深睡眠：δ节律
            dominant_freq = self.delta_frequency
            self.oscillation_amplitude = 0.8
            self.spindle_active = False
        
        # 更新相位
        self.current_phase += 2 * np.pi * dominant_freq * dt / 1000.0
        self.current_phase = self.current_phase % (2 * np.pi)
        
        # 计算调节因子
        modulation = self.oscillation_amplitude * np.sin(self.current_phase)
        
        # 添加注意力调节
        attention_boost = 1.0 + 0.5 * self.attention_focus * np.sin(self.current_phase * 2)
        
        return modulation * attention_boost


class ThalamicNucleus:
    """增强的丘脑核团"""
    
    def __init__(self, nucleus_type: ThalamicNucleusType, size: int, 
                 position: Tuple[float, float, float], config: Dict[str, Any]):
        self.nucleus_type = nucleus_type
        self.size = size
        self.position = position
        self.config = config
        self.neurons: List[Neuron] = []
        self.interneurons: List[Neuron] = []  # 局部抑制性中间神经元
        
        # 振荡状态
        self.oscillation_state = ThalamicOscillationState()
        
        # 输入缓冲
        self.sensory_input: Optional[np.ndarray] = None
        self.cortical_feedback: Dict[int, float] = {}
        self.reticular_inhibition: Dict[int, float] = {}
        
        # 连接权重
        self.cortical_weights: Dict[int, float] = {}
        self.reticular_weights: Dict[int, float] = {}
        
        # 创建神经元
        self._create_neurons()
        
        # 日志
        self.logger = logging.getLogger(f"ThalamicNucleus_{nucleus_type.value}")
        self._current_time = 0.0
        
    def _create_neurons(self):
        """创建丘脑神经元"""
        # 主要中继神经元（兴奋性）
        for i in range(self.size):
            neuron_params = self._get_nucleus_specific_params()
            neuron_params['id'] = i + 20000 + self.nucleus_type.value.__hash__() % 1000
            
            # 根据核团类型选择神经元模型
            if self.nucleus_type in [ThalamicNucleusType.VPL, ThalamicNucleusType.VPM]:
                # 感觉中继核：快速响应
                neuron = create_neuron('lif', neuron_params['id'], {
                    'threshold': -50.0,
                    'reset': -70.0,
                    'tau_m': 15.0,
                    'refractory_period': 2.0
                })
            elif self.nucleus_type == ThalamicNucleusType.MD:
                # 认知核：复杂动力学
                neuron = create_neuron('adex', neuron_params['id'], {
                    'threshold': -50.0,
                    'reset': -70.0,
                    'tau_m': 20.0,
                    'adaptation': 0.1
                })
            else:
                # 其他核团：标准LIF
                neuron = create_neuron('lif', neuron_params['id'], neuron_params)
            
            self.neurons.append(neuron)
        
        # 局部抑制性中间神经元（约20%）
        interneuron_count = max(1, int(self.size * 0.2))
        for i in range(interneuron_count):
            neuron_params = {
                'threshold': -55.0,
                'reset': -70.0,
                'tau_m': 10.0,
                'refractory_period': 1.0
            }
            neuron_id = i + 25000 + self.nucleus_type.value.__hash__() % 1000
            neuron = create_neuron('lif', neuron_id, neuron_params)
            self.interneurons.append(neuron)
    
    def _get_nucleus_specific_params(self) -> Dict[str, Any]:
        """获取核团特异性参数"""
        base_params = {
            'threshold': -50.0,
            'reset': -70.0,
            'tau_m': 20.0,
            'refractory_period': 2.0
        }
        
        # 核团特异性调整
        if self.nucleus_type == ThalamicNucleusType.LGN:
            # 视觉核：快速时间常数
            base_params['tau_m'] = 12.0
            base_params['threshold'] = -48.0
        elif self.nucleus_type == ThalamicNucleusType.VPL:
            # 体感核：中等响应
            base_params['tau_m'] = 15.0
        elif self.nucleus_type == ThalamicNucleusType.MD:
            # 认知核：慢时间常数
            base_params['tau_m'] = 25.0
            base_params['threshold'] = -52.0
        elif self.nucleus_type == ThalamicNucleusType.RETICULAR:
            # 网状核：抑制性
            base_params['tau_m'] = 18.0
            base_params['threshold'] = -53.0
        
        return base_params
    
    def set_sensory_input(self, input_data: np.ndarray):
        """设置感觉输入"""
        if len(input_data) != self.size:
            # 调整输入大小
            if len(input_data) > self.size:
                input_data = input_data[:self.size]
            else:
                input_data = np.pad(input_data, (0, self.size - len(input_data)))
        
        self.sensory_input = input_data.copy()
    
    def set_cortical_feedback(self, feedback: Dict[int, float]):
        """设置皮层反馈"""
        self.cortical_feedback = feedback.copy()
    
    def set_reticular_inhibition(self, inhibition: Dict[int, float]):
        """设置网状核抑制"""
        self.reticular_inhibition = inhibition.copy()
    
    def update_arousal(self, arousal_level: float):
        """更新觉醒水平"""
        self.oscillation_state.arousal_level = np.clip(arousal_level, 0.0, 1.0)
    
    def update_attention(self, attention_focus: float):
        """更新注意力聚焦度"""
        self.oscillation_state.attention_focus = np.clip(attention_focus, 0.0, 1.0)
    
    def step(self, dt: float) -> Dict[str, Any]:
        """更新丘脑核团状态"""
        # 更新振荡状态
        current_time = float(getattr(self, "_current_time", 0.0))
        oscillation_modulation = self.oscillation_state.update_oscillation(dt)
        
        # 更新主要神经元
        relay_spikes = []
        for i, neuron in enumerate(self.neurons):
            # 计算总输入电流
            input_current = 0.0
            
            # 感觉输入
            if self.sensory_input is not None:
                sensory_gain = self._get_sensory_gain()
                input_current += self.sensory_input[i] * sensory_gain
            
            # 皮层反馈
            if neuron.id in self.cortical_feedback:
                feedback_gain = self._get_feedback_gain()
                input_current += self.cortical_feedback[neuron.id] * feedback_gain
            
            # 网状核抑制
            if neuron.id in self.reticular_inhibition:
                input_current -= self.reticular_inhibition[neuron.id] * 2.0
            
            # 振荡调节
            input_current *= (1.0 + 0.3 * oscillation_modulation)
            
            # 更新神经元
            step_fn = getattr(neuron, "step", None)
            if callable(step_fn):
                if step_fn(float(dt), float(input_current), current_time=current_time):
                    relay_spikes.append(neuron.id)
            else:
                if neuron.update(input_current, dt):
                    relay_spikes.append(neuron.id)
        
        # 更新中间神经元
        interneuron_spikes = []
        for neuron in self.interneurons:
            # 中间神经元接收来自主要神经元的输入
            input_current = 0.0
            for main_neuron in self.neurons:
                if main_neuron.voltage > -60.0:  # 活跃神经元
                    input_current += 0.1
            
            # 振荡调节
            input_current *= (1.0 + 0.2 * oscillation_modulation)
            
            step_fn = getattr(neuron, "step", None)
            if callable(step_fn):
                if step_fn(float(dt), float(input_current), current_time=current_time):
                    interneuron_spikes.append(neuron.id)
            else:
                if neuron.update(input_current, dt):
                    interneuron_spikes.append(neuron.id)
        
        self._current_time = current_time + float(dt)
        return {
            'relay_spikes': relay_spikes,
            'interneuron_spikes': interneuron_spikes,
            'oscillation_phase': self.oscillation_state.current_phase,
            'oscillation_amplitude': self.oscillation_state.oscillation_amplitude,
            'arousal_level': self.oscillation_state.arousal_level,
            'attention_focus': self.oscillation_state.attention_focus
        }
    
    def _get_sensory_gain(self) -> float:
        """获取感觉输入增益"""
        base_gain = 1.0
        
        # 根据核团类型调整
        if self.nucleus_type == ThalamicNucleusType.LGN:
            base_gain = 1.2  # 视觉增强
        elif self.nucleus_type in [ThalamicNucleusType.VPL, ThalamicNucleusType.VPM]:
            base_gain = 1.0  # 体感标准
        elif self.nucleus_type == ThalamicNucleusType.MGN:
            base_gain = 1.1  # 听觉增强
        
        # 觉醒水平调节
        arousal_modulation = 0.5 + 0.5 * self.oscillation_state.arousal_level
        
        return base_gain * arousal_modulation
    
    def _get_feedback_gain(self) -> float:
        """获取皮层反馈增益"""
        base_gain = 0.3
        
        # 注意力调节
        attention_modulation = 0.5 + 0.5 * self.oscillation_state.attention_focus
        
        return base_gain * attention_modulation


class ThalamocorticalLoop:
    """完整的丘脑-皮层环路系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thalamic_nuclei: Dict[ThalamicNucleusType, ThalamicNucleus] = {}
        self.cortical_columns: Dict[int, Any] = {}  # 将由外部设置
        
        # 连接矩阵
        self.thalamo_cortical_connections: Dict[Tuple[ThalamicNucleusType, int], List[Tuple[int, int, float]]] = {}
        self.cortico_thalamic_connections: Dict[Tuple[int, ThalamicNucleusType], List[Tuple[int, int, float]]] = {}
        self.reticular_connections: Dict[ThalamicNucleusType, List[Tuple[int, int, float]]] = {}
        
        # 全局状态
        self.global_arousal: float = 0.8
        self.global_attention: Dict[str, float] = {}
        
        # 创建丘脑核团
        self._create_thalamic_nuclei()
        
        # 日志
        self.logger = logging.getLogger("ThalamocorticalLoop")
    
    def _create_thalamic_nuclei(self):
        """创建丘脑核团"""
        nucleus_configs = {
            ThalamicNucleusType.VPL: {'size': 150, 'position': (0, -5, 0)},
            ThalamicNucleusType.VPM: {'size': 120, 'position': (2, -5, 0)},
            ThalamicNucleusType.LGN: {'size': 200, 'position': (-3, -6, 0)},
            ThalamicNucleusType.MGN: {'size': 100, 'position': (3, -6, 0)},
            ThalamicNucleusType.MD: {'size': 180, 'position': (0, -3, 0)},
            ThalamicNucleusType.PULVINAR: {'size': 160, 'position': (0, -7, 0)},
            ThalamicNucleusType.RETICULAR: {'size': 300, 'position': (0, -4, 0)}
        }
        
        for nucleus_type, nucleus_config in nucleus_configs.items():
            if self.config.get('enabled_nuclei', {}).get(nucleus_type.value, True):
                nucleus = ThalamicNucleus(
                    nucleus_type=nucleus_type,
                    size=nucleus_config['size'],
                    position=nucleus_config['position'],
                    config=self.config.get('nucleus_params', {})
                )
                self.thalamic_nuclei[nucleus_type] = nucleus
    
    def add_cortical_column(self, column_id: int, cortical_column):
        """添加皮层柱"""
        self.cortical_columns[column_id] = cortical_column
        
        # 自动创建连接
        self._create_thalamocortical_connections(column_id)
        self._create_corticothalamic_connections(column_id)
    
    def _create_thalamocortical_connections(self, column_id: int):
        """创建丘脑到皮层的连接"""
        cortical_column = self.cortical_columns[column_id]
        
        # 定义核团到皮层层的连接模式
        connection_patterns = {
            ThalamicNucleusType.VPL: [('L4', 0.4), ('L6', 0.2)],      # 体感输入
            ThalamicNucleusType.VPM: [('L4', 0.4), ('L6', 0.2)],      # 面部感觉
            ThalamicNucleusType.LGN: [('L4', 0.5), ('L6', 0.3)],      # 视觉输入
            ThalamicNucleusType.MGN: [('L4', 0.4), ('L6', 0.2)],      # 听觉输入
            ThalamicNucleusType.MD: [('L2/3', 0.3), ('L5', 0.2)],     # 认知输入
            ThalamicNucleusType.PULVINAR: [('L1', 0.2), ('L5', 0.3)]  # 注意力调节
        }
        
        for nucleus_type, patterns in connection_patterns.items():
            if nucleus_type not in self.thalamic_nuclei:
                continue
                
            nucleus = self.thalamic_nuclei[nucleus_type]
            connections = []
            
            for layer_name, probability in patterns:
                target_layer = cortical_column.layers.get(f'{layer_name}_exc')
                if not target_layer:
                    continue
                
                # 创建连接
                for thal_neuron in nucleus.neurons:
                    for cort_neuron in target_layer.neurons:
                        if random.random() < probability:
                            weight = random.uniform(0.2, 0.8)
                            delay = random.uniform(1.0, 4.0)
                            connections.append((thal_neuron.id, cort_neuron.id, weight))
            
            self.thalamo_cortical_connections[(nucleus_type, column_id)] = connections
    
    def _create_corticothalamic_connections(self, column_id: int):
        """创建皮层到丘脑的连接"""
        cortical_column = self.cortical_columns[column_id]
        
        # 定义皮层层到核团的连接模式
        connection_patterns = {
            'L6': [(ThalamicNucleusType.VPL, 0.3), (ThalamicNucleusType.LGN, 0.3), 
                   (ThalamicNucleusType.MD, 0.2)],  # L6主要反馈
            'L5': [(ThalamicNucleusType.MD, 0.2), (ThalamicNucleusType.PULVINAR, 0.2)]  # L5次要反馈
        }
        
        for layer_name, patterns in connection_patterns.items():
            source_layer = cortical_column.layers.get(f'{layer_name}_exc')
            if not source_layer:
                continue
            
            for nucleus_type, probability in patterns:
                if nucleus_type not in self.thalamic_nuclei:
                    continue
                
                nucleus = self.thalamic_nuclei[nucleus_type]
                connections = []
                
                # 创建连接
                for cort_neuron in source_layer.neurons:
                    for thal_neuron in nucleus.neurons:
                        if random.random() < probability:
                            weight = random.uniform(0.1, 0.4)
                            delay = random.uniform(2.0, 6.0)
                            connections.append((cort_neuron.id, thal_neuron.id, weight))
                
                self.cortico_thalamic_connections[(column_id, nucleus_type)] = connections
        
        # 创建到网状核的连接（抑制调节）
        if ThalamicNucleusType.RETICULAR in self.thalamic_nuclei:
            reticular_nucleus = self.thalamic_nuclei[ThalamicNucleusType.RETICULAR]
            connections = []
            
            # L6到网状核的强连接
            l6_layer = cortical_column.layers.get('L6_exc')
            if l6_layer:
                for cort_neuron in l6_layer.neurons:
                    for ret_neuron in reticular_nucleus.neurons:
                        if random.random() < 0.4:
                            weight = random.uniform(0.3, 0.7)
                            connections.append((cort_neuron.id, ret_neuron.id, weight))
            
            self.reticular_connections[column_id] = connections
    
    def set_sensory_input(self, nucleus_type: ThalamicNucleusType, input_data: np.ndarray):
        """设置特定核团的感觉输入"""
        if nucleus_type in self.thalamic_nuclei:
            self.thalamic_nuclei[nucleus_type].set_sensory_input(input_data)
    
    def update_global_arousal(self, arousal_level: float):
        """更新全局觉醒水平"""
        self.global_arousal = np.clip(arousal_level, 0.0, 1.0)
        
        # 更新所有核团的觉醒水平
        for nucleus in self.thalamic_nuclei.values():
            nucleus.update_arousal(self.global_arousal)
    
    def update_attention_focus(self, region: str, focus_level: float):
        """更新特定区域的注意力聚焦"""
        self.global_attention[region] = np.clip(focus_level, 0.0, 1.0)
        
        # 根据区域更新相应核团的注意力
        region_nucleus_mapping = {
            'visual': ThalamicNucleusType.LGN,
            'somatosensory': ThalamicNucleusType.VPL,
            'auditory': ThalamicNucleusType.MGN,
            'cognitive': ThalamicNucleusType.MD,
            'attention': ThalamicNucleusType.PULVINAR
        }
        
        if region in region_nucleus_mapping:
            nucleus_type = region_nucleus_mapping[region]
            if nucleus_type in self.thalamic_nuclei:
                self.thalamic_nuclei[nucleus_type].update_attention(focus_level)
    
    def step(self, dt: float) -> Dict[str, Any]:
        """更新整个丘脑-皮层环路"""
        results = {
            'thalamic_activity': {},
            'cortical_feedback': {},
            'reticular_activity': {},
            'oscillation_states': {}
        }
        
        # 1. 更新丘脑核团
        for nucleus_type, nucleus in self.thalamic_nuclei.items():
            nucleus_result = nucleus.step(dt)
            results['thalamic_activity'][nucleus_type.value] = nucleus_result
            results['oscillation_states'][nucleus_type.value] = {
                'phase': nucleus_result['oscillation_phase'],
                'amplitude': nucleus_result['oscillation_amplitude']
            }
        
        # 2. 计算皮层反馈
        for column_id, cortical_column in self.cortical_columns.items():
            # 获取皮层活动
            cortical_activity = self._get_cortical_activity(cortical_column)
            results['cortical_feedback'][column_id] = cortical_activity
            
            # 将反馈发送到丘脑核团
            for nucleus_type, nucleus in self.thalamic_nuclei.items():
                feedback_key = (column_id, nucleus_type)
                if feedback_key in self.cortico_thalamic_connections:
                    feedback = self._calculate_cortical_feedback(
                        cortical_activity, 
                        self.cortico_thalamic_connections[feedback_key]
                    )
                    nucleus.set_cortical_feedback(feedback)
        
        # 3. 计算网状核抑制
        if ThalamicNucleusType.RETICULAR in self.thalamic_nuclei:
            reticular_nucleus = self.thalamic_nuclei[ThalamicNucleusType.RETICULAR]
            reticular_activity = results['thalamic_activity']['reticular']
            
            # 网状核抑制其他丘脑核团
            for nucleus_type, nucleus in self.thalamic_nuclei.items():
                if nucleus_type != ThalamicNucleusType.RETICULAR:
                    inhibition = self._calculate_reticular_inhibition(
                        reticular_activity['relay_spikes'],
                        nucleus
                    )
                    nucleus.set_reticular_inhibition(inhibition)
        
        return results
    
    def _get_cortical_activity(self, cortical_column) -> Dict[str, float]:
        """获取皮层柱的活动状态"""
        activity = {}
        
        for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
            exc_layer = cortical_column.layers.get(f'{layer_name}_exc')
            if exc_layer:
                # 计算层的平均活动
                voltages = [cortical_column.neurons[nid].voltage for nid in exc_layer.neuron_ids]
                mean_voltage = np.mean(voltages)
                activity[layer_name] = max(0.0, (mean_voltage + 70.0) / 20.0)  # 归一化到0-1
        
        return activity
    
    def _calculate_cortical_feedback(self, cortical_activity: Dict[str, float], 
                                   connections: List[Tuple[int, int, float]]) -> Dict[int, float]:
        """计算皮层对丘脑的反馈"""
        feedback = {}
        
        for pre_id, post_id, weight in connections:
            # 简化：使用平均皮层活动作为反馈强度
            avg_activity = np.mean(list(cortical_activity.values()))
            feedback[post_id] = avg_activity * weight
        
        return feedback
    
    def _calculate_reticular_inhibition(self, reticular_spikes: List[int], 
                                      target_nucleus: ThalamicNucleus) -> Dict[int, float]:
        """计算网状核对目标核团的抑制"""
        inhibition = {}
        
        # 网状核活动越强，抑制越强
        inhibition_strength = len(reticular_spikes) * 0.1
        
        for neuron in target_nucleus.neurons:
            # 随机分配抑制强度
            if random.random() < 0.3:  # 30%的神经元受到抑制
                inhibition[neuron.id] = inhibition_strength * random.uniform(0.5, 1.5)
        
        return inhibition
    
    def get_synchronization_index(self) -> Dict[str, float]:
        """计算丘脑核团间的同步化指数"""
        sync_indices = {}
        
        nucleus_phases = {}
        for nucleus_type, nucleus in self.thalamic_nuclei.items():
            nucleus_phases[nucleus_type.value] = nucleus.oscillation_state.current_phase
        
        # 计算两两之间的相位同步
        nucleus_names = list(nucleus_phases.keys())
        for i in range(len(nucleus_names)):
            for j in range(i + 1, len(nucleus_names)):
                name1, name2 = nucleus_names[i], nucleus_names[j]
                phase_diff = abs(nucleus_phases[name1] - nucleus_phases[name2])
                phase_diff = min(phase_diff, 2 * np.pi - phase_diff)  # 取最小相位差
                sync_index = 1.0 - phase_diff / np.pi  # 归一化到0-1
                sync_indices[f"{name1}_{name2}"] = sync_index
        
        return sync_indices
    
    def simulate_sleep_transition(self, target_stage: int):
        """模拟睡眠阶段转换"""
        stage_arousal_mapping = {
            0: 0.8,   # 觉醒
            1: 0.6,   # N1浅睡眠
            2: 0.4,   # N2睡眠
            3: 0.2,   # N3深睡眠
            4: 0.5    # REM睡眠
        }
        
        target_arousal = stage_arousal_mapping.get(target_stage, 0.8)
        self.update_global_arousal(target_arousal)
        
        # 更新所有核团的睡眠阶段
        for nucleus in self.thalamic_nuclei.values():
            nucleus.oscillation_state.sleep_stage = target_stage
        
        self.logger.info(f"睡眠阶段转换到: {target_stage}, 觉醒水平: {target_arousal}")
