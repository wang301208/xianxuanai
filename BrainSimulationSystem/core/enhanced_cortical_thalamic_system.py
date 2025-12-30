"""
完善的皮层柱+丘脑环路集成系统

这个模块提供了一个完整的皮层-丘脑环路系统，包括：
- 多层皮层柱架构（L1-L6）
- 多个丘脑核团的精确建模
- 双向皮层-丘脑连接
- 振荡同步和注意力调节
- 可塑性学习机制
- 睡眠-觉醒状态调节
- 多模态感觉输入处理
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import random
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import time

from .enhanced_cortical_column import EnhancedCorticalColumnWithLoop, CorticalOscillationType
from .enhanced_thalamocortical_loop import ThalamocorticalLoop, ThalamicNucleusType, ThalamicNucleus
from .neurons import Neuron, create_neuron
from .synapses import Synapse, create_synapse


class SensoryModalityType(Enum):
    """感觉模态类型"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    SOMATOSENSORY = "somatosensory"
    GUSTATORY = "gustatory"
    OLFACTORY = "olfactory"


class AttentionType(Enum):
    """注意力类型"""
    SPATIAL = "spatial"          # 空间注意力
    FEATURE = "feature"          # 特征注意力
    TEMPORAL = "temporal"        # 时间注意力
    EXECUTIVE = "executive"      # 执行注意力


@dataclass
class CorticalThalamicConfig:
    """皮层-丘脑系统配置"""
    # 皮层柱配置
    num_columns: int = 4
    neurons_per_column: int = 2000
    column_spacing: float = 1.0  # mm
    
    # 丘脑配置
    enabled_nuclei: Dict[str, bool] = field(default_factory=lambda: {
        'VPL': True, 'VPM': True, 'LGN': True, 'MGN': True,
        'MD': True, 'PULVINAR': True, 'RETICULAR': True
    })
    
    # 连接配置
    thalamo_cortical_strength: float = 0.8
    cortico_thalamic_strength: float = 0.6
    inter_column_strength: float = 0.3
    
    # 振荡配置
    oscillation_enabled: bool = True
    synchronization_enabled: bool = True
    
    # 可塑性配置
    plasticity_enabled: bool = True
    learning_rate: float = 0.001
    
    # 注意力配置
    attention_enabled: bool = True
    attention_modulation_strength: float = 0.5
    
    # 睡眠配置
    sleep_regulation_enabled: bool = True
    circadian_modulation: bool = True


class CorticalThalamicSystem:
    """完善的皮层柱+丘脑环路系统"""
    
    def __init__(self, config: CorticalThalamicConfig):
        self.config = config
        self.logger = logging.getLogger("CorticalThalamicSystem")
        
        # 核心组件
        self.thalamocortical_loop: Optional[ThalamocorticalLoop] = None
        self.cortical_columns: Dict[int, EnhancedCorticalColumnWithLoop] = {}
        
        # 感觉输入处理
        self.sensory_processors: Dict[SensoryModalityType, Any] = {}
        self.sensory_buffers: Dict[SensoryModalityType, deque] = {}
        
        # 注意力系统
        self.attention_state: Dict[AttentionType, float] = {
            AttentionType.SPATIAL: 0.5,
            AttentionType.FEATURE: 0.5,
            AttentionType.TEMPORAL: 0.5,
            AttentionType.EXECUTIVE: 0.5
        }
        self.attention_targets: Dict[str, float] = {}
        
        # 全局状态
        self.global_arousal: float = 0.8
        self.sleep_stage: int = 0  # 0=觉醒, 1-4=睡眠阶段
        self.circadian_phase: float = 0.0  # 0-2π
        
        # 性能监控
        self.performance_metrics: Dict[str, Any] = {}
        self.synchronization_metrics: Dict[str, float] = {}
        
        # 初始化系统
        self._initialize_system()
        
        self.logger.info(f"皮层-丘脑系统初始化完成: {config.num_columns}个皮层柱")
    
    def _initialize_system(self):
        """初始化整个系统"""
        # 1. 创建丘脑环路
        self._create_thalamocortical_loop()
        
        # 2. 创建皮层柱
        self._create_cortical_columns()
        
        # 3. 建立连接
        self._establish_connections()
        
        # 4. 初始化感觉处理器
        self._initialize_sensory_processors()
        
        # 5. 初始化注意力系统
        self._initialize_attention_system()
    
    def _create_thalamocortical_loop(self):
        """创建丘脑-皮层环路"""
        loop_config = {
            'enabled_nuclei': self.config.enabled_nuclei,
            'oscillation_enabled': self.config.oscillation_enabled,
            'plasticity_enabled': self.config.plasticity_enabled
        }
        
        self.thalamocortical_loop = ThalamocorticalLoop(loop_config)
        self.logger.info("丘脑-皮层环路创建完成")
    
    def _create_cortical_columns(self):
        """创建皮层柱"""
        for column_id in range(self.config.num_columns):
            # 计算皮层柱位置
            x = (column_id % 2) * self.config.column_spacing
            y = (column_id // 2) * self.config.column_spacing
            z = 0.0
            
            # 皮层柱配置
            column_config = {
                'total_neurons': self.config.neurons_per_column,
                'position': (x, y, z),
                'oscillation_enabled': self.config.oscillation_enabled,
                'plasticity_enabled': self.config.plasticity_enabled,
                'learning_rate': self.config.learning_rate,
                'layer_proportions': {
                    'L1': 0.05,
                    'L2/3': 0.35,
                    'L4': 0.25,
                    'L5': 0.25,
                    'L6': 0.10
                }
            }
            
            # 创建增强皮层柱
            cortical_column = EnhancedCorticalColumnWithLoop(
                config=column_config,
                thalamocortical_loop=self.thalamocortical_loop
            )
            
            self.cortical_columns[column_id] = cortical_column
            
            # 将皮层柱添加到丘脑环路
            if self.thalamocortical_loop:
                self.thalamocortical_loop.add_cortical_column(column_id, cortical_column)
        
        self.logger.info(f"创建了 {len(self.cortical_columns)} 个皮层柱")
    
    def _establish_connections(self):
        """建立系统内连接"""
        # 1. 皮层柱间连接
        self._create_inter_column_connections()
        
        # 2. 丘脑-皮层连接（已在ThalamocorticalLoop中处理）
        
        # 3. 全局调节连接
        self._create_global_modulation_connections()
    
    def _create_inter_column_connections(self):
        """创建皮层柱间连接"""
        connection_strength = self.config.inter_column_strength
        
        for col1_id, col1 in self.cortical_columns.items():
            for col2_id, col2 in self.cortical_columns.items():
                if col1_id != col2_id:
                    # 计算距离
                    pos1 = col1.column_config.get('position', (0, 0, 0))
                    pos2 = col2.column_config.get('position', (0, 0, 0))
                    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
                    
                    # 距离依赖的连接概率
                    connection_prob = connection_strength * np.exp(-distance / 2.0)
                    
                    if connection_prob > 0.1:
                        self._connect_columns(col1, col2, connection_prob)
    
    def _connect_columns(self, col1: EnhancedCorticalColumnWithLoop, 
                        col2: EnhancedCorticalColumnWithLoop, probability: float):
        """连接两个皮层柱"""
        # L2/3 到 L2/3 的长程连接
        l23_1 = col1.layers.get('L2/3_exc')
        l23_2 = col2.layers.get('L2/3_exc')
        
        if l23_1 and l23_2:
            for neuron1 in l23_1.neurons[:10]:  # 限制连接数量
                for neuron2 in l23_2.neurons[:10]:
                    if random.random() < probability:
                        weight = random.uniform(0.1, 0.3)
                        delay = random.uniform(2.0, 5.0)  # 长程连接延迟更长
                        
                        synapse_params = {
                            'weight': weight,
                            'delay': delay,
                            'learning_rate': self.config.learning_rate * 0.5
                        }
                        
                        col1.add_synapse(neuron1.id, neuron2.id, 'stdp', synapse_params)
    
    def _create_global_modulation_connections(self):
        """创建全局调节连接"""
        # 这里可以添加全局神经调质系统的连接
        # 如胆碱能、多巴胺能、去甲肾上腺素能系统
        pass
    
    def _initialize_sensory_processors(self):
        """初始化感觉处理器"""
        for modality in SensoryModalityType:
            self.sensory_buffers[modality] = deque(maxlen=100)
            
            # 创建简化的感觉处理器
            processor_config = {
                'modality': modality,
                'buffer_size': 100,
                'preprocessing_enabled': True
            }
            
            self.sensory_processors[modality] = SensoryProcessor(processor_config)
    
    def _initialize_attention_system(self):
        """初始化注意力系统"""
        if not self.config.attention_enabled:
            return
        
        # 初始化注意力目标
        for column_id in self.cortical_columns.keys():
            self.attention_targets[f"column_{column_id}"] = 0.5
        
        for modality in SensoryModalityType:
            self.attention_targets[modality.value] = 0.5
    
    def process_sensory_input(self, modality: SensoryModalityType, 
                            input_data: np.ndarray) -> bool:
        """处理感觉输入"""
        try:
            # 预处理感觉数据
            if modality in self.sensory_processors:
                processed_data = self.sensory_processors[modality].process(input_data)
            else:
                processed_data = input_data
            
            # 存储到缓冲区
            self.sensory_buffers[modality].append(processed_data)
            
            # 路由到相应的丘脑核团
            nucleus_mapping = {
                SensoryModalityType.VISUAL: ThalamicNucleusType.LGN,
                SensoryModalityType.AUDITORY: ThalamicNucleusType.MGN,
                SensoryModalityType.SOMATOSENSORY: ThalamicNucleusType.VPL
            }
            
            if modality in nucleus_mapping and self.thalamocortical_loop:
                nucleus_type = nucleus_mapping[modality]
                
                # 应用注意力调节
                attention_gain = self._get_attention_gain(modality.value)
                modulated_data = processed_data * attention_gain
                
                # 发送到丘脑
                self.thalamocortical_loop.set_sensory_input(nucleus_type, modulated_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理感觉输入失败 ({modality.value}): {e}")
            return False
    
    def _get_attention_gain(self, target: str) -> float:
        """获取注意力增益"""
        base_gain = 1.0
        
        if target in self.attention_targets:
            attention_level = self.attention_targets[target]
            modulation_strength = self.config.attention_modulation_strength
            
            # 注意力增益：0.5 + 0.5 * attention_level * modulation_strength
            gain = base_gain + modulation_strength * (attention_level - 0.5)
            return max(0.1, min(2.0, gain))  # 限制在合理范围内
        
        return base_gain
    
    def update_attention(self, attention_type: AttentionType, level: float):
        """更新注意力状态"""
        self.attention_state[attention_type] = np.clip(level, 0.0, 1.0)
        
        # 更新丘脑核团的注意力
        if self.thalamocortical_loop:
            if attention_type == AttentionType.SPATIAL:
                self.thalamocortical_loop.update_attention_focus('visual', level)
            elif attention_type == AttentionType.FEATURE:
                self.thalamocortical_loop.update_attention_focus('cognitive', level)
    
    def set_attention_target(self, target: str, focus_level: float):
        """设置特定目标的注意力聚焦"""
        self.attention_targets[target] = np.clip(focus_level, 0.0, 1.0)
        
        # 更新相关系统
        if self.thalamocortical_loop:
            self.thalamocortical_loop.update_attention_focus(target, focus_level)
    
    def update_arousal(self, arousal_level: float):
        """更新全局觉醒水平"""
        self.global_arousal = np.clip(arousal_level, 0.0, 1.0)
        
        # 更新丘脑环路
        if self.thalamocortical_loop:
            self.thalamocortical_loop.update_global_arousal(self.global_arousal)
        
        # 更新皮层柱
        for column in self.cortical_columns.values():
            if hasattr(column, 'oscillation_state'):
                column.oscillation_state.alpha_amplitude = 0.2 + 0.3 * self.global_arousal
    
    def simulate_sleep_transition(self, target_stage: int):
        """模拟睡眠阶段转换"""
        if not self.config.sleep_regulation_enabled:
            return
        
        self.sleep_stage = target_stage
        
        # 更新觉醒水平
        stage_arousal_mapping = {
            0: 0.8,   # 觉醒
            1: 0.6,   # N1浅睡眠
            2: 0.4,   # N2睡眠
            3: 0.2,   # N3深睡眠
            4: 0.5    # REM睡眠
        }
        
        target_arousal = stage_arousal_mapping.get(target_stage, 0.8)
        self.update_arousal(target_arousal)
        
        # 更新丘脑环路的睡眠状态
        if self.thalamocortical_loop:
            self.thalamocortical_loop.simulate_sleep_transition(target_stage)
        
        self.logger.info(f"睡眠阶段转换: {target_stage}")
    
    def update_circadian_rhythm(self, time_of_day: float):
        """更新昼夜节律（时间以小时为单位，0-24）"""
        if not self.config.circadian_modulation:
            return
        
        # 转换为相位（0-2π）
        self.circadian_phase = (time_of_day / 24.0) * 2 * np.pi
        
        # 昼夜节律对觉醒的影响
        circadian_arousal = 0.6 + 0.3 * np.sin(self.circadian_phase - np.pi/2)
        
        # 调节全局觉醒水平
        modulated_arousal = self.global_arousal * circadian_arousal
        self.update_arousal(modulated_arousal)
    
    def step(self, dt: float) -> Dict[str, Any]:
        """系统步进"""
        results = {
            'timestamp': time.time(),
            'thalamic_results': {},
            'cortical_results': {},
            'synchronization': {},
            'attention_state': self.attention_state.copy(),
            'global_state': {
                'arousal': self.global_arousal,
                'sleep_stage': self.sleep_stage,
                'circadian_phase': self.circadian_phase
            }
        }
        
        try:
            # 1. 更新丘脑环路
            if self.thalamocortical_loop:
                thalamic_result = self.thalamocortical_loop.step(dt)
                results['thalamic_results'] = thalamic_result
                
                # 处理丘脑输入到皮层
                self._process_thalamic_to_cortical(thalamic_result)
            
            # 2. 更新皮层柱
            for column_id, column in self.cortical_columns.items():
                column_result = column.step(dt)
                results['cortical_results'][column_id] = column_result
            
            # 3. 计算同步化指标
            if self.config.synchronization_enabled:
                sync_metrics = self._calculate_synchronization_metrics()
                results['synchronization'] = sync_metrics
                self.synchronization_metrics.update(sync_metrics)
            
            # 4. 更新性能指标
            self._update_performance_metrics(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"系统步进失败: {e}")
            return results
    
    def _process_thalamic_to_cortical(self, thalamic_result: Dict[str, Any]):
        """处理丘脑到皮层的信号传递"""
        if 'thalamic_activity' not in thalamic_result:
            return
        
        # 构建丘脑输入字典
        thalamic_inputs = {}
        
        for nucleus_name, activity in thalamic_result['thalamic_activity'].items():
            try:
                nucleus_type = ThalamicNucleusType(nucleus_name)
                
                # 从尖峰活动构建输入数组
                if 'relay_spikes' in activity:
                    spike_count = len(activity['relay_spikes'])
                    # 简化：将尖峰数转换为输入强度
                    input_strength = min(spike_count * 0.1, 1.0)
                    
                    # 创建输入数组
                    if nucleus_type in self.thalamocortical_loop.thalamic_nuclei:
                        nucleus = self.thalamocortical_loop.thalamic_nuclei[nucleus_type]
                        input_array = np.full(nucleus.size, input_strength)
                        thalamic_inputs[nucleus_type] = input_array
                        
            except ValueError:
                continue  # 跳过无效的核团名称
        
        # 将输入发送到所有皮层柱
        for column in self.cortical_columns.values():
            column.process_thalamic_input(thalamic_inputs)
    
    def _calculate_synchronization_metrics(self) -> Dict[str, float]:
        """计算同步化指标"""
        metrics = {}
        
        # 1. 丘脑核团间同步
        if self.thalamocortical_loop:
            thalamic_sync = self.thalamocortical_loop.get_synchronization_index()
            metrics['thalamic_synchrony'] = np.mean(list(thalamic_sync.values()))
        
        # 2. 皮层柱间同步
        column_phases = []
        for column in self.cortical_columns.values():
            if hasattr(column, 'oscillation_state'):
                column_phases.append(column.oscillation_state.gamma_phase)
        
        if len(column_phases) > 1:
            # 计算相位同步
            phase_diffs = []
            for i in range(len(column_phases)):
                for j in range(i + 1, len(column_phases)):
                    diff = abs(column_phases[i] - column_phases[j])
                    diff = min(diff, 2 * np.pi - diff)
                    phase_diffs.append(1.0 - diff / np.pi)
            
            metrics['cortical_synchrony'] = np.mean(phase_diffs)
        
        # 3. 皮层-丘脑同步
        if self.thalamocortical_loop and column_phases:
            thalamic_phases = []
            for nucleus in self.thalamocortical_loop.thalamic_nuclei.values():
                thalamic_phases.append(nucleus.oscillation_state.current_phase)
            
            if thalamic_phases:
                # 计算皮层-丘脑相位同步
                ct_sync = []
                for c_phase in column_phases:
                    for t_phase in thalamic_phases:
                        diff = abs(c_phase - t_phase)
                        diff = min(diff, 2 * np.pi - diff)
                        ct_sync.append(1.0 - diff / np.pi)
                
                metrics['cortico_thalamic_synchrony'] = np.mean(ct_sync)
        
        return metrics
    
    def _update_performance_metrics(self, results: Dict[str, Any]):
        """更新性能指标"""
        # 计算系统活动水平
        total_activity = 0.0
        neuron_count = 0
        
        for column_result in results['cortical_results'].values():
            if 'layer_activities' in column_result:
                for layer_activity in column_result['layer_activities'].values():
                    if 'excitatory' in layer_activity:
                        total_activity += layer_activity['excitatory'].get('mean_voltage', -70.0)
                        neuron_count += 1
        
        if neuron_count > 0:
            self.performance_metrics['mean_cortical_activity'] = total_activity / neuron_count
        
        # 计算丘脑活动水平
        if 'thalamic_results' in results and 'thalamic_activity' in results['thalamic_results']:
            thalamic_spikes = 0
            for activity in results['thalamic_results']['thalamic_activity'].values():
                thalamic_spikes += len(activity.get('relay_spikes', []))
            
            self.performance_metrics['thalamic_spike_rate'] = thalamic_spikes
        
        # 计算整体同步水平
        if 'synchronization' in results:
            sync_values = list(results['synchronization'].values())
            if sync_values:
                self.performance_metrics['overall_synchrony'] = np.mean(sync_values)
    
    def get_system_state(self) -> Dict[str, Any]:
        """获取系统完整状态"""
        state = {
            'config': {
                'num_columns': self.config.num_columns,
                'neurons_per_column': self.config.neurons_per_column,
                'enabled_nuclei': self.config.enabled_nuclei
            },
            'global_state': {
                'arousal': self.global_arousal,
                'sleep_stage': self.sleep_stage,
                'circadian_phase': self.circadian_phase
            },
            'attention_state': self.attention_state.copy(),
            'attention_targets': self.attention_targets.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'synchronization_metrics': self.synchronization_metrics.copy()
        }
        
        # 添加皮层柱状态
        state['cortical_columns'] = {}
        for column_id, column in self.cortical_columns.items():
            column_state = {
                'position': column.column_config.get('position', (0, 0, 0)),
                'neuron_count': sum(len(layer.neurons) for layer in column.layers.values()),
                'oscillation_coherence': column.get_oscillation_coherence() if hasattr(column, 'get_oscillation_coherence') else {}
            }
            state['cortical_columns'][column_id] = column_state
        
        # 添加丘脑状态
        if self.thalamocortical_loop:
            state['thalamic_nuclei'] = {}
            for nucleus_type, nucleus in self.thalamocortical_loop.thalamic_nuclei.items():
                nucleus_state = {
                    'size': nucleus.size,
                    'position': nucleus.position,
                    'arousal_level': nucleus.oscillation_state.arousal_level,
                    'attention_focus': nucleus.oscillation_state.attention_focus,
                    'oscillation_amplitude': nucleus.oscillation_state.oscillation_amplitude
                }
                state['thalamic_nuclei'][nucleus_type.value] = nucleus_state
        
        return state
    
    def save_state(self, filepath: str):
        """保存系统状态到文件"""
        try:
            state = self.get_system_state()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"系统状态已保存到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存系统状态失败: {e}")
    
    def reset_system(self):
        """重置整个系统"""
        # 重置全局状态
        self.global_arousal = 0.8
        self.sleep_stage = 0
        self.circadian_phase = 0.0
        
        # 重置注意力状态
        for attention_type in self.attention_state:
            self.attention_state[attention_type] = 0.5
        
        for target in self.attention_targets:
            self.attention_targets[target] = 0.5
        
        # 重置皮层柱
        for column in self.cortical_columns.values():
            if hasattr(column, 'reset'):
                column.reset()
        
        # 重置丘脑环路
        if self.thalamocortical_loop:
            for nucleus in self.thalamocortical_loop.thalamic_nuclei.values():
                nucleus.oscillation_state.current_phase = 0.0
                nucleus.oscillation_state.arousal_level = 0.8
                nucleus.oscillation_state.attention_focus = 0.5
        
        # 清空缓冲区
        for buffer in self.sensory_buffers.values():
            buffer.clear()
        
        # 重置指标
        self.performance_metrics.clear()
        self.synchronization_metrics.clear()
        
        self.logger.info("系统已重置")


class SensoryProcessor:
    """感觉输入处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.modality = config['modality']
        self.buffer_size = config.get('buffer_size', 100)
        self.preprocessing_enabled = config.get('preprocessing_enabled', True)
        
        # 预处理参数
        self.normalization_params = {'mean': 0.0, 'std': 1.0}
        self.filter_params = {'cutoff': 0.1, 'order': 2}
        
    def process(self, input_data: np.ndarray) -> np.ndarray:
        """处理感觉输入数据"""
        if not self.preprocessing_enabled:
            return input_data
        
        processed_data = input_data.copy()
        
        # 1. 归一化
        if len(processed_data) > 0:
            mean_val = np.mean(processed_data)
            std_val = np.std(processed_data) + 1e-6
            processed_data = (processed_data - mean_val) / std_val
        
        # 2. 模态特异性处理
        if self.modality == SensoryModalityType.VISUAL:
            # 视觉：对比度增强
            processed_data = np.tanh(processed_data * 2.0)
        elif self.modality == SensoryModalityType.AUDITORY:
            # 听觉：频率滤波
            processed_data = self._apply_frequency_filter(processed_data)
        elif self.modality == SensoryModalityType.SOMATOSENSORY:
            # 体感：空间滤波
            processed_data = self._apply_spatial_filter(processed_data)
        
        # 3. 幅度限制
        processed_data = np.clip(processed_data, -2.0, 2.0)
        
        return processed_data
    
    def _apply_frequency_filter(self, data: np.ndarray) -> np.ndarray:
        """应用频率滤波"""
        # 简化的低通滤波
        if len(data) > 1:
            filtered = np.zeros_like(data)
            filtered[0] = data[0]
            for i in range(1, len(data)):
                filtered[i] = 0.7 * filtered[i-1] + 0.3 * data[i]
            return filtered
        return data
    
    def _apply_spatial_filter(self, data: np.ndarray) -> np.ndarray:
        """应用空间滤波"""
        # 简化的平滑滤波
        if len(data) > 2:
            filtered = np.zeros_like(data)
            filtered[0] = data[0]
            filtered[-1] = data[-1]
            for i in range(1, len(data) - 1):
                filtered[i] = 0.25 * data[i-1] + 0.5 * data[i] + 0.25 * data[i+1]
            return filtered
        return data


# 工厂函数
def create_cortical_thalamic_system(config: Optional[Dict[str, Any]] = None) -> CorticalThalamicSystem:
    """创建皮层-丘脑系统"""
    if config is None:
        config = {}
    
    # 使用默认配置
    system_config = CorticalThalamicConfig(**config)
    
    return CorticalThalamicSystem(system_config)


def create_default_config() -> CorticalThalamicConfig:
    """创建默认配置"""
    return CorticalThalamicConfig()


def create_minimal_config() -> CorticalThalamicConfig:
    """创建最小配置（用于测试）"""
    return CorticalThalamicConfig(
        num_columns=2,
        neurons_per_column=500,
        enabled_nuclei={
            'VPL': True, 'LGN': True, 'MD': True, 'RETICULAR': True,
            'VPM': False, 'MGN': False, 'PULVINAR': False
        },
        oscillation_enabled=True,
        plasticity_enabled=False,
        attention_enabled=True
    )