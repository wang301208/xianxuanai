"""
增强的突触模型库

扩展原有 synapses.py，添加：
- 多巴胺/乙酰胆碱调制
- 慢性可塑性（L-LTP, L-LTD）
- 胶质细胞交互
- 长程延迟和体积传导
- 代谢耦合
- 突触标记和追踪
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import time

logger = logging.getLogger(__name__)

class NeuromodulatorType(Enum):
    """神经调质类型"""
    DOPAMINE = "dopamine"
    ACETYLCHOLINE = "acetylcholine"
    SEROTONIN = "serotonin"
    NOREPINEPHRINE = "norepinephrine"
    GABA = "gaba"
    GLUTAMATE = "glutamate"
    ADENOSINE = "adenosine"
    NITRIC_OXIDE = "nitric_oxide"

class PlasticityType(Enum):
    """可塑性类型"""
    STDP = "stdp"  # 尖峰时序依赖可塑性
    HOMEOSTATIC = "homeostatic"  # 稳态可塑性
    METAPLASTICITY = "metaplasticity"  # 元可塑性
    L_LTP = "l_ltp"  # 晚期长时程增强
    L_LTD = "l_ltd"  # 晚期长时程抑制
    PROTEIN_SYNTHESIS = "protein_synthesis"  # 蛋白质合成依赖
    STRUCTURAL = "structural"  # 结构可塑性

class SynapseState(Enum):
    """突触状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    POTENTIATED = "potentiated"
    DEPRESSED = "depressed"
    TAGGED = "tagged"  # 突触标记
    CAPTURED = "captured"  # 突触捕获

@dataclass
class NeuromodulatorConfig:
    """神经调质配置"""
    type: NeuromodulatorType
    baseline_concentration: float  # 基线浓度 (μM)
    release_rate: float  # 释放速率 (μM/ms)
    decay_tau: float  # 衰减时间常数 (ms)
    diffusion_coefficient: float  # 扩散系数 (μm²/ms)
    receptor_affinity: float  # 受体亲和力
    max_effect: float  # 最大效应
    
    # 调制参数
    modulation_targets: List[str] = field(default_factory=list)
    dose_response_curve: str = "sigmoid"  # sigmoid, linear, exponential
    cooperativity: float = 1.0

@dataclass
class GlialConfig:
    """胶质细胞配置"""
    astrocyte_density: float  # 星形胶质细胞密度 (cells/mm³)
    microglia_density: float  # 小胶质细胞密度 (cells/mm³)
    oligodendrocyte_density: float  # 少突胶质细胞密度 (cells/mm³)
    
    # 功能参数
    glutamate_uptake_rate: float  # 谷氨酸摄取速率
    k_buffering_capacity: float  # 钾离子缓冲能力
    calcium_wave_speed: float  # 钙波传播速度 (μm/ms)
    atp_release_threshold: float  # ATP释放阈值
    
    # 代谢参数
    glucose_consumption: float  # 葡萄糖消耗率
    lactate_production: float  # 乳酸产生率
    oxygen_consumption: float  # 氧气消耗率

@dataclass
class VolumeTransmissionConfig:
    """体积传导配置"""
    enabled: bool = True
    diffusion_space_fraction: float = 0.2  # 细胞外空间分数
    tortuosity: float = 1.6  # 迂曲度
    clearance_rate: float = 0.1  # 清除速率 (1/ms)
    
    # 扩散参数
    molecular_weight_dependence: bool = True
    temperature_dependence: bool = True
    ph_dependence: bool = True

class NeuromodulatorSystem:
    """神经调质系统"""
    
    def __init__(self):
        self.modulators = {}
        self.release_sites = {}
        self.concentration_maps = {}
        self.receptor_distributions = {}
        
    def add_modulator(self, name: str, config: NeuromodulatorConfig):
        """添加神经调质"""
        self.modulators[name] = config
        self.concentration_maps[name] = np.zeros((100, 100, 100))  # 3D浓度图
        
    def add_release_site(self, modulator: str, location: Tuple[float, float, float], 
                        strength: float):
        """添加释放位点"""
        if modulator not in self.release_sites:
            self.release_sites[modulator] = []
        
        self.release_sites[modulator].append({
            'location': location,
            'strength': strength,
            'last_release': 0.0
        })
    
    def release_modulator(self, modulator: str, site_id: int, amount: float, 
                         current_time: float):
        """释放神经调质"""
        if modulator not in self.release_sites:
            return
        
        if site_id >= len(self.release_sites[modulator]):
            return
        
        site = self.release_sites[modulator][site_id]
        site['last_release'] = current_time
        
        # 更新浓度图
        self._update_concentration_map(modulator, site['location'], amount)
    
    def _update_concentration_map(self, modulator: str, location: Tuple[float, float, float], 
                                 amount: float):
        """更新浓度图"""
        config = self.modulators[modulator]
        x, y, z = location
        
        # 简化的扩散模型
        sigma = np.sqrt(2 * config.diffusion_coefficient * config.decay_tau)
        
        # 在3D网格中添加高斯分布
        grid_x, grid_y, grid_z = np.meshgrid(
            np.arange(100), np.arange(100), np.arange(100)
        )
        
        distance_sq = (grid_x - x)**2 + (grid_y - y)**2 + (grid_z - z)**2
        concentration_add = amount * np.exp(-distance_sq / (2 * sigma**2))
        
        self.concentration_maps[modulator] += concentration_add
    
    def get_concentration(self, modulator: str, location: Tuple[float, float, float]) -> float:
        """获取指定位置的调质浓度"""
        if modulator not in self.concentration_maps:
            return 0.0
        
        x, y, z = [int(coord) for coord in location]
        x = np.clip(x, 0, 99)
        y = np.clip(y, 0, 99)
        z = np.clip(z, 0, 99)
        
        return float(self.concentration_maps[modulator][x, y, z])
    
    def update_concentrations(self, dt: float):
        """更新所有调质浓度（衰减）"""
        for modulator, config in self.modulators.items():
            decay_factor = np.exp(-dt / config.decay_tau)
            self.concentration_maps[modulator] *= decay_factor
    
    def compute_modulation_effect(self, modulator: str, target: str, 
                                 location: Tuple[float, float, float]) -> float:
        """计算调制效应"""
        if modulator not in self.modulators:
            return 1.0
        
        config = self.modulators[modulator]
        if target not in config.modulation_targets:
            return 1.0
        
        concentration = self.get_concentration(modulator, location)
        
        # 剂量-反应曲线
        if config.dose_response_curve == "sigmoid":
            effect = config.max_effect / (1 + np.exp(-config.cooperativity * 
                                                   (concentration - config.baseline_concentration)))
        elif config.dose_response_curve == "linear":
            effect = config.max_effect * concentration / config.baseline_concentration
        elif config.dose_response_curve == "exponential":
            effect = config.max_effect * (1 - np.exp(-concentration / config.baseline_concentration))
        else:
            effect = 1.0
        
        return max(0.0, min(effect, config.max_effect))

class GlialSystem:
    """胶质细胞系统"""
    
    def __init__(self, config: GlialConfig):
        self.config = config
        self.astrocyte_locations = []
        self.microglia_locations = []
        self.oligodendrocyte_locations = []
        
        # 代谢状态
        self.glucose_levels = {}
        self.lactate_levels = {}
        self.atp_levels = {}
        self.calcium_levels = {}
        
        # 活动状态
        self.astrocyte_activation = {}
        self.microglia_activation = {}
        
    def initialize_glial_cells(self, volume: Tuple[float, float, float]):
        """初始化胶质细胞分布"""
        vol_x, vol_y, vol_z = volume
        total_volume = vol_x * vol_y * vol_z  # mm³
        
        # 计算细胞数量
        n_astrocytes = int(total_volume * self.config.astrocyte_density)
        n_microglia = int(total_volume * self.config.microglia_density)
        n_oligodendrocytes = int(total_volume * self.config.oligodendrocyte_density)
        
        # 随机分布
        self.astrocyte_locations = [
            (np.random.uniform(0, vol_x), 
             np.random.uniform(0, vol_y), 
             np.random.uniform(0, vol_z))
            for _ in range(n_astrocytes)
        ]
        
        self.microglia_locations = [
            (np.random.uniform(0, vol_x), 
             np.random.uniform(0, vol_y), 
             np.random.uniform(0, vol_z))
            for _ in range(n_microglia)
        ]
        
        self.oligodendrocyte_locations = [
            (np.random.uniform(0, vol_x), 
             np.random.uniform(0, vol_y), 
             np.random.uniform(0, vol_z))
            for _ in range(n_oligodendrocytes)
        ]
        
        # 初始化代谢状态
        for i, loc in enumerate(self.astrocyte_locations):
            self.glucose_levels[f"astro_{i}"] = 5.0  # mM
            self.lactate_levels[f"astro_{i}"] = 1.0  # mM
            self.atp_levels[f"astro_{i}"] = 3.0  # mM
            self.calcium_levels[f"astro_{i}"] = 0.1  # μM
            self.astrocyte_activation[f"astro_{i}"] = 0.0
        
        for i, loc in enumerate(self.microglia_locations):
            self.microglia_activation[f"micro_{i}"] = 0.0
    
    def glutamate_uptake(self, location: Tuple[float, float, float], 
                        glutamate_concentration: float) -> float:
        """星形胶质细胞谷氨酸摄取"""
        # 找到最近的星形胶质细胞
        min_distance = float('inf')
        nearest_astrocyte = None
        
        for i, astro_loc in enumerate(self.astrocyte_locations):
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(location, astro_loc)))
            if distance < min_distance:
                min_distance = distance
                nearest_astrocyte = f"astro_{i}"
        
        if nearest_astrocyte is None or min_distance > 50.0:  # μm
            return glutamate_concentration
        
        # 米氏动力学
        km = 10.0  # μM
        vmax = self.config.glutamate_uptake_rate
        
        uptake_rate = vmax * glutamate_concentration / (km + glutamate_concentration)
        
        # 考虑距离衰减
        distance_factor = np.exp(-min_distance / 20.0)
        effective_uptake = uptake_rate * distance_factor
        
        return max(0.0, glutamate_concentration - effective_uptake)
    
    def potassium_buffering(self, location: Tuple[float, float, float], 
                           k_concentration: float) -> float:
        """钾离子缓冲"""
        # 找到附近的星形胶质细胞
        buffering_capacity = 0.0
        
        for i, astro_loc in enumerate(self.astrocyte_locations):
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(location, astro_loc)))
            if distance < 30.0:  # μm
                distance_factor = np.exp(-distance / 15.0)
                buffering_capacity += self.config.k_buffering_capacity * distance_factor
        
        # 缓冲效应
        baseline_k = 3.0  # mM
        excess_k = k_concentration - baseline_k
        buffered_k = baseline_k + excess_k * np.exp(-buffering_capacity)
        
        return max(baseline_k, buffered_k)
    
    def calcium_wave_propagation(self, trigger_location: Tuple[float, float, float], 
                               intensity: float, current_time: float):
        """钙波传播"""
        wave_speed = self.config.calcium_wave_speed
        
        for i, astro_loc in enumerate(self.astrocyte_locations):
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(trigger_location, astro_loc)))
            arrival_time = distance / wave_speed
            
            if arrival_time < 1000.0:  # 1秒内到达
                # 计算钙浓度增加
                amplitude = intensity * np.exp(-distance / 100.0)  # 距离衰减
                
                astro_id = f"astro_{i}"
                if astro_id in self.calcium_levels:
                    self.calcium_levels[astro_id] += amplitude
                    
                    # 如果超过阈值，释放ATP
                    if self.calcium_levels[astro_id] > self.config.atp_release_threshold:
                        self._release_atp(astro_loc, amplitude)
    
    def _release_atp(self, location: Tuple[float, float, float], amount: float):
        """释放ATP"""
        # ATP可以激活附近的小胶质细胞
        for i, micro_loc in enumerate(self.microglia_locations):
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(location, micro_loc)))
            if distance < 50.0:  # μm
                activation = amount * np.exp(-distance / 25.0)
                micro_id = f"micro_{i}"
                if micro_id in self.microglia_activation:
                    self.microglia_activation[micro_id] += activation
    
    def metabolic_support(self, location: Tuple[float, float, float], 
                         energy_demand: float) -> float:
        """代谢支持"""
        # 找到最近的星形胶质细胞
        min_distance = float('inf')
        nearest_astrocyte = None
        
        for i, astro_loc in enumerate(self.astrocyte_locations):
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(location, astro_loc)))
            if distance < min_distance:
                min_distance = distance
                nearest_astrocyte = f"astro_{i}"
        
        if nearest_astrocyte is None or min_distance > 30.0:
            return 0.0
        
        # 检查葡萄糖和ATP水平
        glucose = self.glucose_levels.get(nearest_astrocyte, 0.0)
        atp = self.atp_levels.get(nearest_astrocyte, 0.0)
        
        # 计算可提供的能量
        available_energy = min(glucose * 0.1, atp * 0.2)  # 简化的能量转换
        provided_energy = min(energy_demand, available_energy)
        
        # 更新代谢状态
        if provided_energy > 0:
            self.glucose_levels[nearest_astrocyte] -= provided_energy / 0.1
            self.atp_levels[nearest_astrocyte] -= provided_energy / 0.2
            
            # 产生乳酸
            lactate_production = provided_energy * 0.5
            self.lactate_levels[nearest_astrocyte] += lactate_production
        
        return provided_energy
    
    def update_metabolism(self, dt: float):
        """更新代谢状态"""
        for astro_id in self.astrocyte_activation.keys():
            # 葡萄糖消耗
            glucose_consumption = self.config.glucose_consumption * dt
            if astro_id in self.glucose_levels:
                self.glucose_levels[astro_id] = max(0.0, 
                    self.glucose_levels[astro_id] - glucose_consumption)
            
            # ATP再生
            if astro_id in self.glucose_levels and self.glucose_levels[astro_id] > 0:
                atp_generation = glucose_consumption * 30  # ATP/glucose比例
                if astro_id in self.atp_levels:
                    self.atp_levels[astro_id] = min(5.0,  # 最大ATP浓度
                        self.atp_levels[astro_id] + atp_generation)
            
            # 钙离子衰减
            if astro_id in self.calcium_levels:
                decay_rate = 0.01  # 1/ms
                self.calcium_levels[astro_id] *= np.exp(-decay_rate * dt)

class VolumeTransmissionSystem:
    """体积传导系统"""
    
    def __init__(self, config: VolumeTransmissionConfig):
        self.config = config
        self.transmitter_concentrations = {}
        self.diffusion_grid = None
        self.grid_size = (100, 100, 100)
        
        if config.enabled:
            self._initialize_diffusion_grid()
    
    def _initialize_diffusion_grid(self):
        """初始化扩散网格"""
        self.diffusion_grid = {}
        
        # 为每种神经递质创建3D浓度网格
        transmitters = ['glutamate', 'gaba', 'dopamine', 'acetylcholine', 
                       'serotonin', 'norepinephrine']
        
        for transmitter in transmitters:
            self.diffusion_grid[transmitter] = np.zeros(self.grid_size)
    
    def add_point_source(self, transmitter: str, location: Tuple[float, float, float], 
                        amount: float, molecular_weight: float = 150.0):
        """添加点源释放"""
        if not self.config.enabled or transmitter not in self.diffusion_grid:
            return
        
        # 转换为网格坐标
        x, y, z = [int(coord) for coord in location]
        x = np.clip(x, 0, self.grid_size[0] - 1)
        y = np.clip(y, 0, self.grid_size[1] - 1)
        z = np.clip(z, 0, self.grid_size[2] - 1)
        
        # 计算扩散系数
        diffusion_coeff = self._calculate_diffusion_coefficient(molecular_weight)
        
        # 添加到网格
        self.diffusion_grid[transmitter][x, y, z] += amount
    
    def _calculate_diffusion_coefficient(self, molecular_weight: float) -> float:
        """计算扩散系数"""
        # Stokes-Einstein方程的简化版本
        base_diffusion = 1e-6  # cm²/s
        
        # 分子量依赖性
        if self.config.molecular_weight_dependence:
            mw_factor = (150.0 / molecular_weight) ** 0.5  # 参考分子量150 Da
        else:
            mw_factor = 1.0
        
        # 迂曲度和细胞外空间分数
        effective_diffusion = (base_diffusion * mw_factor * 
                             self.config.diffusion_space_fraction / 
                             self.config.tortuosity)
        
        return effective_diffusion
    
    def update_diffusion(self, dt: float):
        """更新扩散过程"""
        if not self.config.enabled:
            return
        
        for transmitter in self.diffusion_grid.keys():
            # 3D扩散方程的简化数值解
            concentration = self.diffusion_grid[transmitter]
            
            # 拉普拉斯算子（6点模板）
            laplacian = np.zeros_like(concentration)
            
            # x方向
            laplacian[1:-1, :, :] += (concentration[2:, :, :] - 
                                     2*concentration[1:-1, :, :] + 
                                     concentration[:-2, :, :])
            
            # y方向
            laplacian[:, 1:-1, :] += (concentration[:, 2:, :] - 
                                     2*concentration[:, 1:-1, :] + 
                                     concentration[:, :-2, :])
            
            # z方向
            laplacian[:, :, 1:-1] += (concentration[:, :, 2:] - 
                                     2*concentration[:, :, 1:-1] + 
                                     concentration[:, :, :-2])
            
            # 扩散系数（假设所有递质相同）
            D = 1e-6  # cm²/s
            
            # 更新浓度
            diffusion_term = D * laplacian * dt
            clearance_term = -self.config.clearance_rate * concentration * dt
            
            self.diffusion_grid[transmitter] += diffusion_term + clearance_term
            
            # 确保非负
            self.diffusion_grid[transmitter] = np.maximum(
                self.diffusion_grid[transmitter], 0.0
            )
    
    def get_concentration(self, transmitter: str, 
                         location: Tuple[float, float, float]) -> float:
        """获取指定位置的递质浓度"""
        if (not self.config.enabled or 
            transmitter not in self.diffusion_grid):
            return 0.0
        
        x, y, z = [int(coord) for coord in location]
        x = np.clip(x, 0, self.grid_size[0] - 1)
        y = np.clip(y, 0, self.grid_size[1] - 1)
        z = np.clip(z, 0, self.grid_size[2] - 1)
        
        return float(self.diffusion_grid[transmitter][x, y, z])

class EnhancedSynapse:
    """增强的突触模型"""
    
    def __init__(self, synapse_id: str, pre_neuron_id: str, post_neuron_id: str,
                 synapse_type: str = "glutamate", location: Tuple[float, float, float] = (0, 0, 0)):
        self.synapse_id = synapse_id
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.synapse_type = synapse_type
        self.location = location
        
        # 基本参数
        self.weight = 1.0
        self.delay = 1.0  # ms
        self.state = SynapseState.ACTIVE
        
        # 可塑性相关
        self.plasticity_types = []
        self.tag_strength = 0.0
        self.tag_decay_tau = 3600000.0  # 1小时
        self.protein_synthesis_threshold = 0.5
        
        # 调制相关
        self.modulation_sensitivity = {}
        self.baseline_release_probability = 0.5
        self.current_release_probability = 0.5
        
        # 代谢相关
        self.energy_cost = 1.0
        self.vesicle_pool_size = 100
        self.available_vesicles = 100
        self.vesicle_refill_rate = 10.0  # vesicles/s
        
        # 历史记录
        self.spike_times_pre = []
        self.spike_times_post = []
        self.weight_history = []
        self.activity_history = []
        
        # 分子机制
        self.calcium_concentration = 0.1  # μM
        self.camp_concentration = 1.0  # μM
        self.protein_levels = {
            'CaMKII': 1.0,
            'PKA': 1.0,
            'CREB': 1.0,
            'Arc': 0.0,
            'Homer': 1.0
        }
    
    def add_plasticity_rule(self, plasticity_type: PlasticityType, 
                           parameters: Dict[str, Any]):
        """添加可塑性规则"""
        self.plasticity_types.append({
            'type': plasticity_type,
            'parameters': parameters,
            'enabled': True
        })
    
    def set_modulation_sensitivity(self, modulator: NeuromodulatorType, 
                                  sensitivity: float):
        """设置调制敏感性"""
        self.modulation_sensitivity[modulator] = sensitivity
    
    def process_spike(self, spike_time: float, is_presynaptic: bool, 
                     neuromodulator_system: Optional[NeuromodulatorSystem] = None,
                     glial_system: Optional[GlialSystem] = None) -> float:
        """处理尖峰事件"""
        if is_presynaptic:
            self.spike_times_pre.append(spike_time)
            
            # 检查囊泡可用性
            if self.available_vesicles <= 0:
                return 0.0
            
            # 计算释放概率
            release_prob = self._calculate_release_probability(
                neuromodulator_system, spike_time
            )
            
            # 随机释放
            if np.random.random() < release_prob:
                self.available_vesicles -= 1
                
                # 计算突触后电流
                psc = self._calculate_postsynaptic_current(
                    neuromodulator_system, glial_system
                )
                
                # 更新可塑性
                self._update_plasticity(spike_time, True)
                
                return psc
        else:
            self.spike_times_post.append(spike_time)
            self._update_plasticity(spike_time, False)
        
        return 0.0
    
    def _calculate_release_probability(self, neuromodulator_system: Optional[NeuromodulatorSystem],
                                     spike_time: float) -> float:
        """计算释放概率"""
        prob = self.baseline_release_probability
        
        # 神经调质调制
        if neuromodulator_system:
            for modulator, sensitivity in self.modulation_sensitivity.items():
                modulation = neuromodulator_system.compute_modulation_effect(
                    modulator.value, "release_probability", self.location
                )
                prob *= (1.0 + sensitivity * (modulation - 1.0))
        
        # 钙依赖性
        calcium_factor = self.calcium_concentration / 1.0  # 归一化
        prob *= calcium_factor
        
        # 囊泡耗竭
        depletion_factor = self.available_vesicles / self.vesicle_pool_size
        prob *= depletion_factor
        
        return np.clip(prob, 0.0, 1.0)
    
    def _calculate_postsynaptic_current(self, neuromodulator_system: Optional[NeuromodulatorSystem],
                                       glial_system: Optional[GlialSystem]) -> float:
        """计算突触后电流"""
        # 基础电流
        base_current = self.weight * 10.0  # pA
        
        # 神经调质调制
        if neuromodulator_system:
            for modulator, sensitivity in self.modulation_sensitivity.items():
                modulation = neuromodulator_system.compute_modulation_effect(
                    modulator.value, "synaptic_strength", self.location
                )
                base_current *= (1.0 + sensitivity * (modulation - 1.0))
        
        # 胶质细胞调制
        if glial_system and self.synapse_type == "glutamate":
            # 谷氨酸摄取
            glutamate_conc = base_current / 10.0  # 简化转换
            reduced_conc = glial_system.glutamate_uptake(self.location, glutamate_conc)
            base_current *= (reduced_conc / glutamate_conc) if glutamate_conc > 0 else 1.0
        
        return base_current
    
    def _update_plasticity(self, spike_time: float, is_presynaptic: bool):
        """更新可塑性"""
        for plasticity_rule in self.plasticity_types:
            if not plasticity_rule['enabled']:
                continue
            
            ptype = plasticity_rule['type']
            params = plasticity_rule['parameters']
            
            if ptype == PlasticityType.STDP:
                self._update_stdp(spike_time, is_presynaptic, params)
            elif ptype == PlasticityType.HOMEOSTATIC:
                self._update_homeostatic(spike_time, params)
            elif ptype == PlasticityType.L_LTP:
                self._update_late_ltp(spike_time, params)
            elif ptype == PlasticityType.L_LTD:
                self._update_late_ltd(spike_time, params)
            elif ptype == PlasticityType.METAPLASTICITY:
                self._update_metaplasticity(spike_time, params)
    
    def _update_stdp(self, spike_time: float, is_presynaptic: bool, params: Dict[str, Any]):
        """更新STDP"""
        tau_plus = params.get('tau_plus', 20.0)
        tau_minus = params.get('tau_minus', 20.0)
        a_plus = params.get('a_plus', 0.01)
        a_minus = params.get('a_minus', 0.012)
        
        if is_presynaptic:
            # 查找最近的突触后尖峰
            recent_post_spikes = [t for t in self.spike_times_post 
                                if abs(t - spike_time) < 5 * tau_minus]
            
            for post_time in recent_post_spikes:
                dt = post_time - spike_time
                if dt > 0:  # 突触后在前
                    weight_change = a_plus * np.exp(-dt / tau_plus)
                    self.weight += weight_change
                    
                    # 标记突触
                    if weight_change > 0.005:
                        self.tag_strength = max(self.tag_strength, weight_change)
                        self.state = SynapseState.TAGGED
        else:
            # 查找最近的突触前尖峰
            recent_pre_spikes = [t for t in self.spike_times_pre 
                               if abs(t - spike_time) < 5 * tau_plus]
            
            for pre_time in recent_pre_spikes:
                dt = spike_time - pre_time
                if dt > 0:  # 突触前在前
                    weight_change = -a_minus * np.exp(-dt / tau_minus)
                    self.weight += weight_change
                    
                    # 标记突触
                    if abs(weight_change) > 0.005:
                        self.tag_strength = max(self.tag_strength, abs(weight_change))
                        self.state = SynapseState.TAGGED
        
        # 限制权重范围
        self.weight = np.clip(self.weight, 0.0, 5.0)
    
    def _update_homeostatic(self, spike_time: float, params: Dict[str, Any]):
        """更新稳态可塑性"""
        target_rate = params.get('target_rate', 5.0)  # Hz
        tau_homeostatic = params.get('tau', 1000.0)  # ms
        
        # 计算最近的活动率
        recent_window = 1000.0  # ms
        recent_spikes = len([t for t in self.spike_times_post 
                           if spike_time - t < recent_window])
        current_rate = recent_spikes / (recent_window / 1000.0)  # Hz
        
        # 稳态调整
        rate_error = target_rate - current_rate
        weight_change = rate_error * 0.001 * (1000.0 / tau_homeostatic)
        
        self.weight += weight_change
        self.weight = np.clip(self.weight, 0.1, 3.0)
    
    def _update_late_ltp(self, spike_time: float, params: Dict[str, Any]):
        """更新晚期LTP"""
        if self.state != SynapseState.TAGGED:
            return
        
        # 检查是否满足蛋白质合成条件
        if self.tag_strength > self.protein_synthesis_threshold:
            # 模拟蛋白质合成
            synthesis_strength = params.get('synthesis_strength', 2.0)
            
            # 增加蛋白质水平
            self.protein_levels['Arc'] += synthesis_strength * self.tag_strength
            self.protein_levels['CaMKII'] += synthesis_strength * self.tag_strength * 0.5
            
            # 结构性权重增加
            structural_increase = synthesis_strength * self.tag_strength
            self.weight += structural_increase
            
            # 状态转换
            self.state = SynapseState.POTENTIATED
            
            # 重置标记
            self.tag_strength = 0.0
    
    def _update_late_ltd(self, spike_time: float, params: Dict[str, Any]):
        """更新晚期LTD"""
        if self.state != SynapseState.TAGGED:
            return
        
        # 检查抑制条件
        if self.tag_strength > params.get('depression_threshold', 0.3):
            depression_strength = params.get('depression_strength', 0.5)
            
            # 减少蛋白质水平
            self.protein_levels['Homer'] -= depression_strength * self.tag_strength
            
            # 结构性权重减少
            structural_decrease = depression_strength * self.tag_strength
            self.weight -= structural_decrease
            
            # 状态转换
            self.state = SynapseState.DEPRESSED
            
            # 重置标记
            self.tag_strength = 0.0
        
        self.weight = max(0.1, self.weight)
    
    def _update_metaplasticity(self, spike_time: float, params: Dict[str, Any]):
        """更新元可塑性"""
        # 计算历史活动
        history_window = params.get('history_window', 10000.0)  # ms
        recent_activity = len([t for t in self.spike_times_pre + self.spike_times_post
                             if spike_time - t < history_window])
        
        # 调整可塑性阈值
        if recent_activity > params.get('high_activity_threshold', 100):
            # 高活动：提高LTD阈值，降低LTP阈值
            for rule in self.plasticity_types:
                if rule['type'] == PlasticityType.STDP:
                    rule['parameters']['a_plus'] *= 0.95
                    rule['parameters']['a_minus'] *= 1.05
        elif recent_activity < params.get('low_activity_threshold', 20):
            # 低活动：降低LTD阈值，提高LTP阈值
            for rule in self.plasticity_types:
                if rule['type'] == PlasticityType.STDP:
                    rule['parameters']['a_plus'] *= 1.05
                    rule['parameters']['a_minus'] *= 0.95
    
    def update_vesicle_pool(self, dt: float):
        """更新囊泡池"""
        # 囊泡补充
        refill = self.vesicle_refill_rate * dt / 1000.0  # 转换为ms
        self.available_vesicles = min(self.vesicle_pool_size, 
                                    self.available_vesicles + refill)
    
    def update_tag_decay(self, dt: float):
        """更新标记衰减"""
        if self.tag_strength > 0:
            decay_factor = np.exp(-dt / self.tag_decay_tau)
            self.tag_strength *= decay_factor
            
            if self.tag_strength < 0.01:
                self.tag_strength = 0.0
                if self.state == SynapseState.TAGGED:
                    self.state = SynapseState.ACTIVE
    
    def get_energy_consumption(self) -> float:
        """获取能量消耗"""
        # 基础代谢
        base_cost = 0.1  # ATP/ms
        
        # 活动相关消耗
        activity_cost = len(self.spike_times_pre) * 0.01  # 每个尖峰的成本
        
        # 可塑性相关消耗
        plasticity_cost = 0.0
        if self.state in [SynapseState.TAGGED, SynapseState.POTENTIATED]:
            plasticity_cost = 0.05
        
        return base_cost + activity_cost + plasticity_cost
    
    def export_state(self) -> Dict[str, Any]:
        """导出突触状态"""
        return {
            'synapse_id': self.synapse_id,
            'pre_neuron_id': self.pre_neuron_id,
            'post_neuron_id': self.post_neuron_id,
            'synapse_type': self.synapse_type,
            'location': self.location,
            'weight': self.weight,
            'delay': self.delay,
            'state': self.state.value,
            'tag_strength': self.tag_strength,
            'available_vesicles': self.available_vesicles,
            'protein_levels': self.protein_levels.copy(),
            'plasticity_types': [rule['type'].value for rule in self.plasticity_types],
            'energy_consumption': self.get_energy_consumption()
        }

class EnhancedSynapseManager:
    """增强的突触管理器"""
    
    def __init__(self):
        self.synapses = {}
        self.neuromodulator_system = NeuromodulatorSystem()
        self.glial_system = None
        self.volume_transmission = None
        
        # 性能监控
        self.update_times = []
        self.memory_usage = []
        
        # 并行处理
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
    def initialize_systems(self, glial_config: Optional[GlialConfig] = None,
                          volume_config: Optional[VolumeTransmissionConfig] = None,
                          tissue_volume: Tuple[float, float, float] = (1000, 1000, 1000)):
        """初始化辅助系统"""
        
        # 初始化胶质系统
        if glial_config:
            self.glial_system = GlialSystem(glial_config)
            self.glial_system.initialize_glial_cells(tissue_volume)
        
        # 初始化体积传导系统
        if volume_config:
            self.volume_transmission = VolumeTransmissionSystem(volume_config)
        
        # 添加常见神经调质
        self._setup_neuromodulators()
    
    def _setup_neuromodulators(self):
        """设置神经调质系统"""
        # 多巴胺
        dopamine_config = NeuromodulatorConfig(
            type=NeuromodulatorType.DOPAMINE,
            baseline_concentration=0.1,  # μM
            release_rate=0.5,
            decay_tau=1000.0,  # ms
            diffusion_coefficient=0.76e-6,  # cm²/s
            receptor_affinity=0.1,
            max_effect=2.0,
            modulation_targets=['release_probability', 'synaptic_strength', 'plasticity'],
            dose_response_curve="sigmoid",
            cooperativity=2.0
        )
        self.neuromodulator_system.add_modulator("dopamine", dopamine_config)
        
        # 乙酰胆碱
        ach_config = NeuromodulatorConfig(
            type=NeuromodulatorType.ACETYLCHOLINE,
            baseline_concentration=0.05,
            release_rate=1.0,
            decay_tau=100.0,  # 快速降解
            diffusion_coefficient=1.2e-6,
            receptor_affinity=0.05,
            max_effect=1.5,
            modulation_targets=['attention', 'plasticity', 'excitability'],
            dose_response_curve="sigmoid",
            cooperativity=1.5
        )
        self.neuromodulator_system.add_modulator("acetylcholine", ach_config)
        
        # 5-羟色胺
        serotonin_config = NeuromodulatorConfig(
            type=NeuromodulatorType.SEROTONIN,
            baseline_concentration=0.02,
            release_rate=0.3,
            decay_tau=2000.0,
            diffusion_coefficient=0.6e-6,
            receptor_affinity=0.01,
            max_effect=1.8,
            modulation_targets=['mood', 'sleep', 'plasticity'],
            dose_response_curve="sigmoid",
            cooperativity=1.0
        )
        self.neuromodulator_system.add_modulator("serotonin", serotonin_config)
    
    def create_synapse(self, synapse_id: str, pre_neuron_id: str, post_neuron_id: str,
                      synapse_type: str = "glutamate", 
                      location: Tuple[float, float, float] = (0, 0, 0),
                      plasticity_rules: Optional[List[Dict[str, Any]]] = None) -> EnhancedSynapse:
        """创建增强突触"""
        
        synapse = EnhancedSynapse(synapse_id, pre_neuron_id, post_neuron_id, 
                                synapse_type, location)
        
        # 添加可塑性规则
        if plasticity_rules:
            for rule in plasticity_rules:
                ptype = PlasticityType(rule['type'])
                params = rule.get('parameters', {})
                synapse.add_plasticity_rule(ptype, params)
        else:
            # 默认STDP
            synapse.add_plasticity_rule(PlasticityType.STDP, {
                'tau_plus': 20.0,
                'tau_minus': 20.0,
                'a_plus': 0.01,
                'a_minus': 0.012
            })
        
        # 设置神经调质敏感性
        if synapse_type == "glutamate":
            synapse.set_modulation_sensitivity(NeuromodulatorType.DOPAMINE, 0.5)
            synapse.set_modulation_sensitivity(NeuromodulatorType.ACETYLCHOLINE, 0.3)
        elif synapse_type == "gaba":
            synapse.set_modulation_sensitivity(NeuromodulatorType.SEROTONIN, 0.4)
            synapse.set_modulation_sensitivity(NeuromodulatorType.NOREPINEPHRINE, 0.2)
        
        self.synapses[synapse_id] = synapse
        return synapse
    
    def process_spike_event(self, neuron_id: str, spike_time: float, 
                           is_presynaptic: bool = True) -> List[Tuple[str, float, float]]:
        """处理尖峰事件"""
        postsynaptic_currents = []
        
        # 找到相关突触
        relevant_synapses = []
        for synapse in self.synapses.values():
            if is_presynaptic and synapse.pre_neuron_id == neuron_id:
                relevant_synapses.append(synapse)
            elif not is_presynaptic and synapse.post_neuron_id == neuron_id:
                relevant_synapses.append(synapse)
        
        # 并行处理突触
        def process_synapse(synapse):
            current = synapse.process_spike(
                spike_time, is_presynaptic, 
                self.neuromodulator_system, 
                self.glial_system
            )
            
            if current > 0 and is_presynaptic:
                # 添加到体积传导
                if self.volume_transmission:
                    self.volume_transmission.add_point_source(
                        synapse.synapse_type, synapse.location, current * 0.1
                    )
                
                return (synapse.post_neuron_id, current, spike_time + synapse.delay)
            return None
        
        # 使用线程池处理
        futures = [self.thread_pool.submit(process_synapse, synapse) 
                  for synapse in relevant_synapses]
        
        for future in futures:
            result = future.result()
            if result:
                postsynaptic_currents.append(result)
        
        return postsynaptic_currents
    
    def release_neuromodulator(self, modulator: str, location: Tuple[float, float, float],
                              amount: float, current_time: float):
        """释放神经调质"""
        # 找到最近的释放位点
        if modulator in self.neuromodulator_system.release_sites:
            sites = self.neuromodulator_system.release_sites[modulator]
            if sites:
                # 找到最近的位点
                min_distance = float('inf')
                nearest_site = 0
                
                for i, site in enumerate(sites):
                    distance = np.sqrt(sum((a - b)**2 for a, b in 
                                         zip(location, site['location'])))
                    if distance < min_distance:
                        min_distance = distance
                        nearest_site = i
                
                self.neuromodulator_system.release_modulator(
                    modulator, nearest_site, amount, current_time
                )
        else:
            # 添加新的释放位点
            self.neuromodulator_system.add_release_site(modulator, location, 1.0)
            self.neuromodulator_system.release_modulator(modulator, 0, amount, current_time)
    
    def update_systems(self, dt: float, current_time: float):
        """更新所有系统"""
        start_time = time.time()
        
        # 更新神经调质浓度
        self.neuromodulator_system.update_concentrations(dt)
        
        # 更新胶质细胞代谢
        if self.glial_system:
            self.glial_system.update_metabolism(dt)
        
        # 更新体积传导
        if self.volume_transmission:
            self.volume_transmission.update_diffusion(dt)
        
        # 更新所有突触
        def update_synapse(synapse):
            synapse.update_vesicle_pool(dt)
            synapse.update_tag_decay(dt)
        
        # 并行更新突触
        futures = [self.thread_pool.submit(update_synapse, synapse) 
                  for synapse in self.synapses.values()]
        
        for future in futures:
            future.result()
        
        # 记录性能
        update_time = time.time() - start_time
        self.update_times.append(update_time)
        
        # 保持最近1000次记录
        if len(self.update_times) > 1000:
            self.update_times = self.update_times[-1000:]
    
    def induce_long_term_potentiation(self, synapse_ids: List[str], 
                                     stimulation_pattern: str = "theta_burst"):
        """诱导长时程增强"""
        for synapse_id in synapse_ids:
            if synapse_id in self.synapses:
                synapse = self.synapses[synapse_id]
                
                # 添加L-LTP规则
                synapse.add_plasticity_rule(PlasticityType.L_LTP, {
                    'synthesis_strength': 2.0,
                    'protein_threshold': 0.3
                })
                
                # 强化标记
                synapse.tag_strength = 0.8
                synapse.state = SynapseState.TAGGED
                
                # 增加钙浓度
                synapse.calcium_concentration = 2.0
                
                # 触发胶质细胞钙波
                if self.glial_system:
                    self.glial_system.calcium_wave_propagation(
                        synapse.location, 1.0, 0.0
                    )
    
    def induce_long_term_depression(self, synapse_ids: List[str]):
        """诱导长时程抑制"""
        for synapse_id in synapse_ids:
            if synapse_id in self.synapses:
                synapse = self.synapses[synapse_id]
                
                # 添加L-LTD规则
                synapse.add_plasticity_rule(PlasticityType.L_LTD, {
                    'depression_strength': 0.5,
                    'depression_threshold': 0.3
                })
                
                # 设置抑制标记
                synapse.tag_strength = 0.6
                synapse.state = SynapseState.TAGGED
                
                # 降低钙浓度
                synapse.calcium_concentration = 0.05
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        if not self.synapses:
            return {}
        
        # 基本统计
        total_synapses = len(self.synapses)
        active_synapses = sum(1 for s in self.synapses.values() 
                            if s.state == SynapseState.ACTIVE)
        tagged_synapses = sum(1 for s in self.synapses.values() 
                            if s.state == SynapseState.TAGGED)
        
        # 权重统计
        weights = [s.weight for s in self.synapses.values()]
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        
        # 能量消耗
        total_energy = sum(s.get_energy_consumption() for s in self.synapses.values())
        
        # 性能统计
        avg_update_time = np.mean(self.update_times) if self.update_times else 0.0
        
        # 神经调质统计
        modulator_stats = {}
        for modulator, concentration_map in self.neuromodulator_system.concentration_maps.items():
            modulator_stats[modulator] = {
                'mean_concentration': float(np.mean(concentration_map)),
                'max_concentration': float(np.max(concentration_map)),
                'active_volume': float(np.sum(concentration_map > 0.01))
            }
        
        return {
            'total_synapses': total_synapses,
            'active_synapses': active_synapses,
            'tagged_synapses': tagged_synapses,
            'potentiated_synapses': sum(1 for s in self.synapses.values() 
                                      if s.state == SynapseState.POTENTIATED),
            'depressed_synapses': sum(1 for s in self.synapses.values() 
                                    if s.state == SynapseState.DEPRESSED),
            'mean_weight': mean_weight,
            'std_weight': std_weight,
            'total_energy_consumption': total_energy,
            'avg_update_time_ms': avg_update_time * 1000,
            'neuromodulator_stats': modulator_stats,
            'glial_cells': {
                'astrocytes': len(self.glial_system.astrocyte_locations) if self.glial_system else 0,
                'microglia': len(self.glial_system.microglia_locations) if self.glial_system else 0,
                'oligodendrocytes': len(self.glial_system.oligodendrocyte_locations) if self.glial_system else 0
            } if self.glial_system else {}
        }
    
    def export_network_state(self, filepath: str):
        """导出网络状态"""
        network_state = {
            'synapses': {sid: synapse.export_state() 
                        for sid, synapse in self.synapses.items()},
            'neuromodulators': {
                modulator: {
                    'concentration_map': concentration_map.tolist(),
                    'release_sites': self.neuromodulator_system.release_sites.get(modulator, [])
                }
                for modulator, concentration_map in 
                self.neuromodulator_system.concentration_maps.items()
            },
            'statistics': self.get_network_statistics(),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(network_state, f, indent=2, ensure_ascii=False)
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)

def create_enhanced_synapse_manager(glial_config: Optional[GlialConfig] = None,
                                   volume_config: Optional[VolumeTransmissionConfig] = None,
                                   tissue_volume: Tuple[float, float, float] = (1000, 1000, 1000)) -> EnhancedSynapseManager:
    """创建增强突触管理器的便捷函数"""
    
    manager = EnhancedSynapseManager()
    
    # 使用默认配置
    if glial_config is None:
        glial_config = GlialConfig(
            astrocyte_density=50000,  # cells/mm³
            microglia_density=5000,
            oligodendrocyte_density=10000,
            glutamate_uptake_rate=100.0,
            k_buffering_capacity=0.8,
            calcium_wave_speed=20.0,  # μm/ms
            atp_release_threshold=1.0,
            glucose_consumption=0.1,
            lactate_production=0.05,
            oxygen_consumption=0.02
        )
    
    if volume_config is None:
        volume_config = VolumeTransmissionConfig(
            enabled=True,
            diffusion_space_fraction=0.2,
            tortuosity=1.6,
            clearance_rate=0.1
        )
    
    manager.initialize_systems(glial_config, volume_config, tissue_volume)
    
    return manager

if __name__ == "__main__":
    # 测试增强突触系统
    logging.basicConfig(level=logging.INFO)
    
    # 创建管理器
    manager = create_enhanced_synapse_manager()
    
    # 创建一些测试突触
    for i in range(100):
        synapse_id = f"synapse_{i}"
        pre_id = f"neuron_{i}"
        post_id = f"neuron_{i+100}"
        location = (np.random.rand() * 1000, np.random.rand() * 1000, np.random.rand() * 1000)
        
        plasticity_rules = [
            {'type': 'stdp', 'parameters': {'tau_plus': 20.0, 'tau_minus': 20.0}},
            {'type': 'homeostatic', 'parameters': {'target_rate': 5.0, 'tau': 1000.0}}
        ]
        
        manager.create_synapse(synapse_id, pre_id, post_id, "glutamate", 
                             location, plasticity_rules)
    
    # 模拟一些活动
    for t in range(1000):  # 1秒仿真
        # 随机尖峰
        if np.random.random() < 0.1:
            neuron_id = f"neuron_{np.random.randint(0, 100)}"
            manager.process_spike_event(neuron_id, float(t), True)
        
        # 随机神经调质释放
        if np.random.random() < 0.01:
            location = (np.random.rand() * 1000, np.random.rand() * 1000, np.random.rand() * 1000)
            manager.release_neuromodulator("dopamine", location, 1.0, float(t))
        
        # 更新系统
        manager.update_systems(1.0, float(t))
    
    # 获取统计信息
    stats = manager.get_network_statistics()
    print("增强突触系统测试完成:")
    print(f"  总突触数: {stats['total_synapses']}")
    print(f"  活跃突触数: {stats['active_synapses']}")
    print(f"  标记突触数: {stats['tagged_synapses']}")
    print(f"  平均权重: {stats['mean_weight']:.3f}")
    print(f"  总能量消耗: {stats['total_energy_consumption']:.2f}")
    print(f"  平均更新时间: {stats['avg_update_time_ms']:.2f} ms")
    
    # 导出状态
    manager.export_network_state("enhanced_synapse_network.json")
    
    # 清理
    manager.cleanup()
    
    print("测试完成，状态已导出到 enhanced_synapse_network.json")