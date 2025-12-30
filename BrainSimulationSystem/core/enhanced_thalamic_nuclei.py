"""
增强的丘脑核团系统

这个模块提供了详细的丘脑核团建模，包括：
- 多种丘脑核团的精确建模
- 核团间的复杂连接
- 丘脑网状核的门控功能
- 振荡生成和同步
- 注意力和觉醒调节
- 感觉信息的中继和处理
- 皮层-丘脑-基底节环路
"""

from typing import DefaultDict, Dict, List, Optional, Any, Tuple, Set
import numpy as np
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import time

from .neurons import Neuron, create_neuron
from .synapses import Synapse, create_synapse


class ThalamicCellType(Enum):
    """丘脑细胞类型"""
    RELAY = "relay"                    # 中继细胞
    INTERNEURON = "interneuron"        # 中间神经元
    RETICULAR = "reticular"            # 网状核细胞
    MATRIX = "matrix"                  # 基质细胞
    CORE = "core"                      # 核心细胞


class ThalamicOscillationType(Enum):
    """丘脑振荡类型"""
    SPINDLE = "spindle"                # 睡眠纺锤波 (7-14 Hz)
    DELTA = "delta"                    # δ波 (1-4 Hz)
    ALPHA = "alpha"                    # α波 (8-12 Hz)
    BETA = "beta"                      # β波 (13-30 Hz)
    GAMMA = "gamma"                    # γ波 (30-100 Hz)


@dataclass
class ThalamicNucleusConfig:
    """丘脑核团配置"""
    nucleus_type: str
    size: int = 1000
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # 细胞类型比例
    cell_type_ratios: Dict[ThalamicCellType, float] = field(default_factory=lambda: {
        ThalamicCellType.RELAY: 0.75,
        ThalamicCellType.INTERNEURON: 0.20,
        ThalamicCellType.MATRIX: 0.05
    })
    
    # 连接参数
    internal_connectivity: float = 0.1
    external_connectivity: float = 0.3
    
    # 振荡参数
    intrinsic_frequency: float = 10.0  # Hz
    oscillation_amplitude: float = 0.5
    
    # 功能参数
    sensory_modality: Optional[str] = None
    cortical_targets: List[str] = field(default_factory=list)
    subcortical_inputs: List[str] = field(default_factory=list)


@dataclass
class InternalSynapse:
    """Lightweight internal synapse record for thalamic nuclei."""

    source_id: int
    target_id: int
    weight: float
    delay: float
    synapse_type: str = "excitatory"
    learning_rate: float = 0.0


class EnhancedThalamicNucleus:
    """增强的丘脑核团"""
    
    def __init__(self, config: ThalamicNucleusConfig):
        self.config = config
        self.nucleus_type = config.nucleus_type
        self.size = config.size
        self.position = config.position
        
        self.logger = logging.getLogger(f"ThalamicNucleus_{self.nucleus_type}")
        
        # 神经元群体
        self.cell_populations: Dict[ThalamicCellType, List[Neuron]] = {}
        self.all_neurons: List[Neuron] = []
        
        # 内部连接
        self.internal_synapses: List[InternalSynapse] = []
        self.connectivity_matrix: DefaultDict[int, Dict[int, float]] | None = None
        
        # 振荡状态
        self.oscillation_state = ThalamicOscillationState()
        
        # 功能状态
        self.arousal_level: float = 0.8
        self.attention_focus: float = 0.5
        self.gating_state: float = 1.0  # 门控状态 (0=完全抑制, 1=完全开放)
        
        # 输入输出
        self.sensory_input: Optional[np.ndarray] = None
        self.cortical_feedback: Optional[np.ndarray] = None
        self.subcortical_input: Optional[np.ndarray] = None
        
        # 活动监控
        self.activity_history: deque = deque(maxlen=1000)
        self.spike_trains: Dict[int, deque] = {}
        
        # 可塑性
        self.plasticity_enabled: bool = True
        self.learning_rate: float = 0.001
        
        # 初始化核团
        self._initialize_nucleus()
        
        self.logger.info(f"丘脑核团 {self.nucleus_type} 初始化完成: {self.size} 个神经元")
    
    def _initialize_nucleus(self):
        """初始化核团"""
        # 1. 创建神经元群体
        self._create_cell_populations()
        
        # 2. 建立内部连接
        self._create_internal_connections()
        
        # 3. 初始化振荡状态
        self._initialize_oscillations()
        
        # 4. 设置功能特性
        self._setup_functional_properties()
    
    def _create_cell_populations(self):
        """创建细胞群体"""
        neuron_id = 0
        
        for cell_type, ratio in self.config.cell_type_ratios.items():
            population_size = int(self.size * ratio)
            population = []
            
            for i in range(population_size):
                # 根据细胞类型创建神经元
                neuron_params = self._get_neuron_params(cell_type)
                neuron_type = neuron_params.get("neuron_type", "lif")
                neuron = create_neuron(neuron_type, neuron_id, neuron_params)
                
                population.append(neuron)
                self.all_neurons.append(neuron)
                
                # 初始化尖峰记录
                self.spike_trains[neuron_id] = deque(maxlen=100)
                
                neuron_id += 1
            
            self.cell_populations[cell_type] = population
            self.logger.debug(f"创建 {cell_type.value} 群体: {population_size} 个神经元")
    
    def _get_neuron_params(self, cell_type: ThalamicCellType) -> Dict[str, Any]:
        """获取神经元参数"""
        base_params = {
            'neuron_type': 'lif',
            'tau_m': 20.0,
            'v_rest': -70.0,
            'v_threshold': -55.0,
            'v_reset': -75.0,
            'refractory_period': 2.0
        }
        
        if cell_type == ThalamicCellType.RELAY:
            # 中继细胞：具有T型钙通道，支持爆发放电
            base_params.update({
                'neuron_type': 'thalamic_relay',
                'tau_m': 25.0,
                'has_t_channel': True,
                'burst_threshold': -65.0,
                'burst_duration': 5.0
            })
        elif cell_type == ThalamicCellType.INTERNEURON:
            # 中间神经元：快速抑制性
            base_params.update({
                'neuron_type': 'fast_spiking',
                'tau_m': 10.0,
                'v_threshold': -50.0,
                'inhibitory': True
            })
        elif cell_type == ThalamicCellType.RETICULAR:
            # 网状核细胞：具有强烈的爆发特性
            base_params.update({
                'neuron_type': 'reticular',
                'tau_m': 30.0,
                'has_t_channel': True,
                'burst_threshold': -60.0,
                'burst_duration': 10.0,
                'inhibitory': True
            })
        elif cell_type == ThalamicCellType.MATRIX:
            # 基质细胞：广泛投射
            base_params.update({
                'neuron_type': 'matrix',
                'tau_m': 35.0,
                'diffuse_projection': True
            })
        elif cell_type == ThalamicCellType.CORE:
            # 核心细胞：特异性投射
            base_params.update({
                'neuron_type': 'core',
                'tau_m': 20.0,
                'specific_projection': True
            })
        
        return base_params
    
    def _create_internal_connections(self):
        """创建内部连接"""
        connectivity_prob = self.config.internal_connectivity
        
        # 使用稀疏连接结构，避免为大核团分配 O(N^2) 的矩阵
        n_neurons = len(self.all_neurons)
        self.connectivity_matrix = defaultdict(dict)
        
        # 不同细胞类型间的连接规则
        connection_rules = self._get_connection_rules()
        
        for source_type, source_pop in self.cell_populations.items():
            for target_type, target_pop in self.cell_populations.items():
                if (source_type, target_type) in connection_rules:
                    rule = connection_rules[(source_type, target_type)]
                    self._connect_populations(source_pop, target_pop, rule)
    
    def _get_connection_rules(self) -> Dict[Tuple[ThalamicCellType, ThalamicCellType], Dict[str, Any]]:
        """获取连接规则"""
        return {
            # 中继细胞 -> 中继细胞 (弱兴奋性)
            (ThalamicCellType.RELAY, ThalamicCellType.RELAY): {
                'probability': 0.05,
                'weight': 0.2,
                'delay': 1.0,
                'synapse_type': 'excitatory'
            },
            
            # 中继细胞 -> 中间神经元 (兴奋性)
            (ThalamicCellType.RELAY, ThalamicCellType.INTERNEURON): {
                'probability': 0.3,
                'weight': 0.5,
                'delay': 0.5,
                'synapse_type': 'excitatory'
            },
            
            # 中间神经元 -> 中继细胞 (抑制性)
            (ThalamicCellType.INTERNEURON, ThalamicCellType.RELAY): {
                'probability': 0.4,
                'weight': -0.8,
                'delay': 0.5,
                'synapse_type': 'inhibitory'
            },
            
            # 网状核 -> 中继细胞 (强抑制性)
            (ThalamicCellType.RETICULAR, ThalamicCellType.RELAY): {
                'probability': 0.6,
                'weight': -1.2,
                'delay': 1.0,
                'synapse_type': 'inhibitory'
            },
            
            # 中继细胞 -> 网状核 (兴奋性)
            (ThalamicCellType.RELAY, ThalamicCellType.RETICULAR): {
                'probability': 0.2,
                'weight': 0.4,
                'delay': 1.0,
                'synapse_type': 'excitatory'
            }
        }
    
    def _connect_populations(self, source_pop: List[Neuron], target_pop: List[Neuron], 
                           rule: Dict[str, Any]):
        """连接神经元群体"""
        probability = rule['probability']
        weight = rule['weight']
        delay = rule['delay']
        synapse_type = rule['synapse_type']
        
        synapse_count = 0

        if not source_pop or not target_pop or probability <= 0.0:
            return

        source_ids = [int(n.id) for n in source_pop]
        target_ids = [int(n.id) for n in target_pop]
        total_pairs = len(source_ids) * len(target_ids)
        expected = int(probability * total_pairs)
        expected = max(0, min(expected, total_pairs))

        # 采样建立连接，避免 O(N^2) 遍历导致的性能/内存问题
        for _ in range(expected):
            src = int(np.random.choice(source_ids))
            tgt = int(np.random.choice(target_ids))
            if src == tgt:
                continue
            syn = InternalSynapse(
                source_id=src,
                target_id=tgt,
                weight=float(weight * np.random.uniform(0.8, 1.2)),
                delay=float(max(0.0, delay + np.random.uniform(-0.2, 0.2))),
                synapse_type=str(synapse_type),
                learning_rate=float(self.learning_rate) if self.plasticity_enabled else 0.0,
            )
            self.internal_synapses.append(syn)
            if self.connectivity_matrix is not None:
                self.connectivity_matrix[src][tgt] = float(weight)
            synapse_count += 1
        
        self.logger.debug(f"创建连接: {len(source_pop)} -> {len(target_pop)}, {synapse_count} 个突触")
    
    def _initialize_oscillations(self):
        """初始化振荡"""
        self.oscillation_state.intrinsic_frequency = self.config.intrinsic_frequency
        self.oscillation_state.oscillation_amplitude = self.config.oscillation_amplitude
        
        # 根据核团类型设置振荡特性
        if self.nucleus_type in ['VPL', 'VPM', 'LGN', 'MGN']:
            # 感觉中继核：支持多种振荡
            self.oscillation_state.supported_oscillations = [
                ThalamicOscillationType.ALPHA,
                ThalamicOscillationType.BETA,
                ThalamicOscillationType.GAMMA
            ]
        elif self.nucleus_type == 'RETICULAR':
            # 网状核：主要产生睡眠纺锤波
            self.oscillation_state.supported_oscillations = [
                ThalamicOscillationType.SPINDLE,
                ThalamicOscillationType.DELTA
            ]
        elif self.nucleus_type in ['MD', 'PULVINAR']:
            # 高级核团：支持认知相关振荡
            self.oscillation_state.supported_oscillations = [
                ThalamicOscillationType.ALPHA,
                ThalamicOscillationType.BETA,
                ThalamicOscillationType.GAMMA
            ]
    
    def _setup_functional_properties(self):
        """设置功能特性"""
        # 根据核团类型设置功能
        if self.nucleus_type == 'LGN':
            self.config.sensory_modality = 'visual'
            self.config.cortical_targets = ['V1']
        elif self.nucleus_type == 'MGN':
            self.config.sensory_modality = 'auditory'
            self.config.cortical_targets = ['A1']
        elif self.nucleus_type in ['VPL', 'VPM']:
            self.config.sensory_modality = 'somatosensory'
            self.config.cortical_targets = ['S1']
        elif self.nucleus_type == 'MD':
            self.config.cortical_targets = ['PFC']
        elif self.nucleus_type == 'PULVINAR':
            self.config.cortical_targets = ['PPC', 'IT']
    
    def set_sensory_input(self, input_data: np.ndarray):
        """设置感觉输入"""
        self.sensory_input = input_data
        
        # 将输入分发到中继细胞
        if ThalamicCellType.RELAY in self.cell_populations:
            relay_cells = self.cell_populations[ThalamicCellType.RELAY]
            
            # 简化：将输入均匀分发
            input_per_cell = len(input_data) // len(relay_cells) if len(relay_cells) > 0 else 0
            
            for i, neuron in enumerate(relay_cells):
                start_idx = i * input_per_cell
                end_idx = min(start_idx + input_per_cell, len(input_data))
                
                if start_idx < len(input_data):
                    cell_input = np.mean(input_data[start_idx:end_idx])
                    # 将输入转换为电流注入
                    neuron.external_current = cell_input * 10.0  # pA
    
    def set_cortical_feedback(self, feedback_data: np.ndarray):
        """设置皮层反馈"""
        self.cortical_feedback = feedback_data
        
        # 皮层反馈主要影响网状核和中继细胞
        if ThalamicCellType.RETICULAR in self.cell_populations:
            reticular_cells = self.cell_populations[ThalamicCellType.RETICULAR]
            
            for i, neuron in enumerate(reticular_cells):
                if i < len(feedback_data):
                    # 反馈调节网状核的抑制强度
                    feedback_strength = feedback_data[i]
                    neuron.external_current = feedback_strength * 5.0

    def set_subcortical_input(self, input_data: np.ndarray):
        """设置来自其他丘脑/皮层下结构的输入（用于核团间传递）。"""
        self.subcortical_input = input_data

        if input_data is None:
            return
        try:
            input_array = np.asarray(input_data, dtype=float)
        except Exception:
            return

        drive = float(np.mean(input_array)) if input_array.size else 0.0
        for neuron in self.all_neurons:
            neuron.external_current += drive * 2.0
    
    def update_arousal(self, arousal_level: float):
        """更新觉醒水平"""
        self.arousal_level = np.clip(arousal_level, 0.0, 1.0)
        
        # 觉醒水平影响振荡特性
        self.oscillation_state.arousal_level = self.arousal_level
        
        # 高觉醒：增强gamma振荡，减少delta/spindle
        if self.arousal_level > 0.7:
            self.oscillation_state.current_oscillation = ThalamicOscillationType.GAMMA
            self.oscillation_state.oscillation_amplitude = 0.3 + 0.4 * self.arousal_level
        elif self.arousal_level > 0.4:
            self.oscillation_state.current_oscillation = ThalamicOscillationType.ALPHA
            self.oscillation_state.oscillation_amplitude = 0.5
        else:
            # 低觉醒：睡眠相关振荡
            if self.nucleus_type == 'RETICULAR':
                self.oscillation_state.current_oscillation = ThalamicOscillationType.SPINDLE
            else:
                self.oscillation_state.current_oscillation = ThalamicOscillationType.DELTA
            self.oscillation_state.oscillation_amplitude = 0.8
    
    def update_attention_focus(self, focus_level: float):
        """更新注意力聚焦"""
        self.attention_focus = np.clip(focus_level, 0.0, 1.0)
        
        # 注意力影响门控状态
        if self.nucleus_type == 'RETICULAR':
            # 网状核：高注意力 -> 低抑制（更开放的门控）
            self.gating_state = 0.3 + 0.7 * self.attention_focus
        else:
            # 其他核团：高注意力 -> 增强传递
            self.gating_state = 0.5 + 0.5 * self.attention_focus
        
        # 更新振荡状态
        self.oscillation_state.attention_focus = self.attention_focus
    
    def step(self, dt: float) -> Dict[str, Any]:
        """核团步进"""
        results = {
            'timestamp': time.time(),
            'nucleus_type': self.nucleus_type,
            'cell_activities': {},
            'oscillation_state': {},
            'functional_state': {},
            'spike_counts': {}
        }
        
        try:
            # 1. 更新振荡状态
            self._update_oscillations(dt)
            
            # 2. 更新神经元
            self._update_neurons(dt)
            
            # 3. 处理突触传递
            self._process_synaptic_transmission(dt)
            
            # 4. 应用门控
            self._apply_gating()
            
            # 5. 收集结果
            self._collect_results(results)
            
            # 6. 记录活动历史
            mean_activity = np.mean([neuron.membrane_potential for neuron in self.all_neurons])
            self.activity_history.append(mean_activity)
            
            return results
            
        except Exception as e:
            self.logger.error(f"核团步进失败: {e}")
            return results
    
    def _update_oscillations(self, dt: float):
        """更新振荡状态"""
        # 更新相位
        frequency = self.oscillation_state.get_current_frequency()
        self.oscillation_state.current_phase += 2 * np.pi * frequency * dt / 1000.0
        
        # 保持相位在[0, 2π]范围内
        self.oscillation_state.current_phase = self.oscillation_state.current_phase % (2 * np.pi)
        
        # 计算振荡驱动
        oscillation_drive = (
            self.oscillation_state.oscillation_amplitude * 
            np.sin(self.oscillation_state.current_phase)
        )
        
        # 应用振荡驱动到神经元
        for neuron in self.all_neurons:
            neuron.external_current += oscillation_drive * 2.0  # pA
    
    def _update_neurons(self, dt: float):
        """更新神经元"""
        for neuron in self.all_neurons:
            # 应用门控
            gated_current = neuron.external_current * self.gating_state
            
            # 更新神经元状态
            old_potential = neuron.membrane_potential
            neuron.step(dt, gated_current)
            
            # 检测尖峰
            if (old_potential < neuron.v_threshold and 
                neuron.membrane_potential >= neuron.v_threshold):
                
                self.spike_trains[neuron.id].append(time.time())
            
            # 重置外部电流
            neuron.external_current = 0.0
    
    def _process_synaptic_transmission(self, dt: float):
        """处理突触传递"""
        for synapse in self.internal_synapses:
            # 检查源神经元是否发放尖峰
            source_neuron = self._get_neuron_by_id(synapse.source_id)
            target_neuron = self._get_neuron_by_id(synapse.target_id)
            
            if source_neuron and target_neuron:
                # 简化的突触传递
                if (source_neuron.membrane_potential >= source_neuron.v_threshold and
                    synapse.delay <= dt):
                    
                    # 计算突触电流
                    synaptic_current = synapse.weight * self.gating_state
                    
                    # 应用到目标神经元
                    target_neuron.external_current += synaptic_current
                    
                    # 可塑性更新
                    if self.plasticity_enabled and synapse.learning_rate > 0:
                        self._update_synapse_plasticity(synapse, source_neuron, target_neuron)
    
    def _apply_gating(self):
        """应用门控机制"""
        # 网状核特殊的门控逻辑
        if self.nucleus_type == 'RETICULAR':
            # 网状核通过抑制来实现门控
            for neuron in self.cell_populations.get(ThalamicCellType.RETICULAR, []):
                # 门控状态低 -> 网状核活跃 -> 更多抑制
                inhibition_strength = (1.0 - self.gating_state) * 10.0
                neuron.external_current += inhibition_strength
        
        # 其他核团的门控通过调节传递效率实现（已在_update_neurons中处理）
    
    def _update_synapse_plasticity(self, synapse: Synapse, source_neuron: Neuron, target_neuron: Neuron):
        """更新突触可塑性"""
        # 简化的STDP规则
        if hasattr(synapse, 'last_pre_spike') and hasattr(synapse, 'last_post_spike'):
            dt_spike = synapse.last_post_spike - synapse.last_pre_spike
            
            if abs(dt_spike) < 50.0:  # 50ms窗口
                if dt_spike > 0:  # 突触前先于突触后
                    weight_change = synapse.learning_rate * np.exp(-dt_spike / 20.0)
                else:  # 突触后先于突触前
                    weight_change = -synapse.learning_rate * np.exp(dt_spike / 20.0)
                
                # 更新权重
                synapse.weight += weight_change
                synapse.weight = np.clip(synapse.weight, -2.0, 2.0)
    
    def _get_neuron_by_id(self, neuron_id: int) -> Optional[Neuron]:
        """根据ID获取神经元"""
        for neuron in self.all_neurons:
            if neuron.id == neuron_id:
                return neuron
        return None
    
    def _collect_results(self, results: Dict[str, Any]):
        """收集结果"""
        # 细胞群体活动
        for cell_type, population in self.cell_populations.items():
            potentials = [neuron.membrane_potential for neuron in population]
            
            results['cell_activities'][cell_type.value] = {
                'mean_potential': np.mean(potentials),
                'std_potential': np.std(potentials),
                'active_cells': sum(1 for p in potentials if p > -60.0),
                'population_size': len(population)
            }
        
        # 振荡状态
        results['oscillation_state'] = {
            'current_oscillation': self.oscillation_state.current_oscillation.value if self.oscillation_state.current_oscillation else None,
            'current_phase': self.oscillation_state.current_phase,
            'frequency': self.oscillation_state.get_current_frequency(),
            'amplitude': self.oscillation_state.oscillation_amplitude
        }
        
        # 功能状态
        results['functional_state'] = {
            'arousal_level': self.arousal_level,
            'attention_focus': self.attention_focus,
            'gating_state': self.gating_state,
            'sensory_input_present': self.sensory_input is not None,
            'cortical_feedback_present': self.cortical_feedback is not None
        }
        
        # 尖峰统计
        for cell_type, population in self.cell_populations.items():
            spike_count = 0
            for neuron in population:
                if neuron.id in self.spike_trains:
                    # 计算最近100ms内的尖峰数
                    recent_spikes = [
                        spike_time for spike_time in self.spike_trains[neuron.id]
                        if time.time() - spike_time < 0.1
                    ]
                    spike_count += len(recent_spikes)
            
            results['spike_counts'][cell_type.value] = spike_count
    
    def get_output_activity(self) -> np.ndarray:
        """获取输出活动"""
        # 主要从中继细胞获取输出
        if ThalamicCellType.RELAY in self.cell_populations:
            relay_cells = self.cell_populations[ThalamicCellType.RELAY]
            
            output = np.array([
                neuron.membrane_potential for neuron in relay_cells
            ])
            
            # 应用门控
            return output * self.gating_state
        
        return np.array([])
    
    def reset(self):
        """重置核团状态"""
        # 重置神经元
        for neuron in self.all_neurons:
            neuron.membrane_potential = neuron.v_rest
            neuron.external_current = 0.0
        
        # 重置振荡状态
        self.oscillation_state.current_phase = 0.0
        
        # 重置功能状态
        self.arousal_level = 0.8
        self.attention_focus = 0.5
        self.gating_state = 1.0
        
        # 清空输入
        self.sensory_input = None
        self.cortical_feedback = None
        self.subcortical_input = None
        
        # 清空历史
        self.activity_history.clear()
        for spike_train in self.spike_trains.values():
            spike_train.clear()
        
        self.logger.info(f"核团 {self.nucleus_type} 已重置")


class ThalamicOscillationState:
    """丘脑振荡状态"""
    
    def __init__(self):
        # 当前振荡
        self.current_oscillation: Optional[ThalamicOscillationType] = ThalamicOscillationType.ALPHA
        self.current_phase: float = 0.0
        
        # 振荡参数
        self.intrinsic_frequency: float = 10.0
        self.oscillation_amplitude: float = 0.5
        
        # 支持的振荡类型
        self.supported_oscillations: List[ThalamicOscillationType] = [
            ThalamicOscillationType.ALPHA,
            ThalamicOscillationType.BETA,
            ThalamicOscillationType.GAMMA
        ]
        
        # 状态调节
        self.arousal_level: float = 0.8
        self.attention_focus: float = 0.5
        
        # 频率映射
        self.frequency_mapping = {
            ThalamicOscillationType.DELTA: 2.5,
            ThalamicOscillationType.SPINDLE: 10.0,
            ThalamicOscillationType.ALPHA: 10.0,
            ThalamicOscillationType.BETA: 20.0,
            ThalamicOscillationType.GAMMA: 40.0
        }
    
    def get_current_frequency(self) -> float:
        """获取当前频率"""
        if self.current_oscillation:
            base_freq = self.frequency_mapping[self.current_oscillation]
            
            # 觉醒和注意力调节频率
            arousal_modulation = 1.0 + 0.2 * (self.arousal_level - 0.5)
            attention_modulation = 1.0 + 0.1 * (self.attention_focus - 0.5)
            
            return base_freq * arousal_modulation * attention_modulation
        
        return self.intrinsic_frequency
    
    def switch_oscillation(self, new_oscillation: ThalamicOscillationType):
        """切换振荡类型"""
        # 允许在运行时扩展支持的振荡类型（测试/睡眠阶段切换需要）
        if new_oscillation not in self.supported_oscillations:
            self.supported_oscillations.append(new_oscillation)
        self.current_oscillation = new_oscillation
        # 重置相位以避免突变
        self.current_phase = 0.0


# 工厂函数
def create_enhanced_thalamic_nucleus(nucleus_type: str, config: Optional[Dict[str, Any]] = None) -> EnhancedThalamicNucleus:
    """创建增强丘脑核团"""
    if config is None:
        config = {}
    
    # 使用默认配置
    nucleus_config = ThalamicNucleusConfig(nucleus_type=nucleus_type, **config)
    
    return EnhancedThalamicNucleus(nucleus_config)


def create_standard_thalamic_nuclei(size_hint: Optional[int] = None) -> Dict[str, EnhancedThalamicNucleus]:
    """创建标准丘脑核团集合

    Args:
        size_hint: 当提供时，按提示规模缩小核团大小以适配测试/轻量运行。
    """
    nuclei = {}

    if size_hint is not None:
        base = int(max(80, min(300, int(size_hint) // 2 or 80)))
        sizes = {
            'LGN': base,
            'MGN': max(60, int(base * 0.8)),
            'VPL': max(60, int(base * 0.9)),
            'VPM': max(60, int(base * 0.7)),
            'MD': max(60, int(base * 0.9)),
            'PULVINAR': max(60, int(base * 0.9)),
            'RETICULAR': max(60, int(base * 0.8)),
        }
    else:
        sizes = {
            'LGN': 1200,
            'MGN': 800,
            'VPL': 1000,
            'VPM': 600,
            'MD': 1500,
            'PULVINAR': 2000,
            'RETICULAR': 800,
        }
    
    # 感觉中继核
    nuclei['LGN'] = create_enhanced_thalamic_nucleus('LGN', {
        'size': sizes['LGN'],
        'position': (-2.0, 0.0, 0.0),
        'sensory_modality': 'visual'
    })
    
    nuclei['MGN'] = create_enhanced_thalamic_nucleus('MGN', {
        'size': sizes['MGN'],
        'position': (-1.0, -2.0, 0.0),
        'sensory_modality': 'auditory'
    })
    
    nuclei['VPL'] = create_enhanced_thalamic_nucleus('VPL', {
        'size': sizes['VPL'],
        'position': (0.0, -1.0, 0.0),
        'sensory_modality': 'somatosensory'
    })
    
    nuclei['VPM'] = create_enhanced_thalamic_nucleus('VPM', {
        'size': sizes['VPM'],
        'position': (0.0, -0.5, 0.0),
        'sensory_modality': 'somatosensory'
    })
    
    # 高级核团
    nuclei['MD'] = create_enhanced_thalamic_nucleus('MD', {
        'size': sizes['MD'],
        'position': (1.0, 0.0, 0.0),
        'cortical_targets': ['PFC']
    })
    
    nuclei['PULVINAR'] = create_enhanced_thalamic_nucleus('PULVINAR', {
        'size': sizes['PULVINAR'],
        'position': (-1.0, 1.0, 0.0),
        'cortical_targets': ['PPC', 'IT']
    })
    
    # 网状核
    nuclei['RETICULAR'] = create_enhanced_thalamic_nucleus('RETICULAR', {
        'size': sizes['RETICULAR'],
        'position': (0.0, 0.0, 1.0),
        'cell_type_ratios': {
            ThalamicCellType.RETICULAR: 1.0  # 纯网状核细胞
        }
    })
    
    return nuclei
