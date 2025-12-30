"""
详细的多室神经元模型
Detailed Multi-Compartment Neuron Model
"""
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

from .enums import CellType
from .parameters import get_cell_parameters

# 兼容无 numba 环境：优先尝试导入，失败则提供空装饰器与占位对象
try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        # 空装饰器，直接返回原函数
        def deco(func):
            return func
        return deco

class DetailedNeuron:
    """详细的多室神经元模型"""
    
    def __init__(self, neuron_id: int, cell_type: CellType, position: Tuple[float, float, float]):
        self.neuron_id = neuron_id
        self.cell_type = cell_type
        self.position = position
        
        # 从数据库获取参数
        self.params = get_cell_parameters(cell_type)
        
        # 多室结构
        self.compartments = {
            'soma': self._create_soma(),
            'basal_dendrites': self._create_basal_dendrites(),
            'apical_dendrite': self._create_apical_dendrite(),
            'axon': self._create_axon()
        }
        
        # 状态变量
        self.membrane_potential = self.params.resting_potential
        self.calcium_concentration = 0.1  # μM
        self.spike_times = []
        self.last_spike_time = -np.inf
        
        # 离子通道状态
        self.ion_channels = self._initialize_ion_channels()
        
        # 突触连接
        self.input_synapses = []
        self.output_synapses = []
        
        # 代谢状态
        self.atp_level = 1.0
        self.glucose_level = 1.0
        self.oxygen_level = 1.0
        
        self.logger = logging.getLogger(f"Neuron_{neuron_id}")
    

    
    def _create_soma(self) -> Dict[str, Any]:
        """创建胞体室"""
        return {
            'diameter': self.params.soma_diameter,
            'length': self.params.soma_diameter,
            'capacitance': self.params.membrane_capacitance,
            'voltage': self.params.resting_potential,
            'ion_channels': {}
        }
    
    def _create_basal_dendrites(self) -> Dict[str, Any]:
        """创建基树突室"""
        return {
            'total_length': self.params.dendritic_length * 0.6,
            'diameter': 2.0,
            'spine_count': int(self.params.dendritic_length * 0.6 * self.params.spine_density),
            'voltage': self.params.resting_potential,
            'calcium': 0.1
        }
    
    def _create_apical_dendrite(self) -> Dict[str, Any]:
        """创建顶树突室"""
        return {
            'total_length': self.params.dendritic_length * 0.4,
            'diameter': 3.0,
            'spine_count': int(self.params.dendritic_length * 0.4 * self.params.spine_density),
            'voltage': self.params.resting_potential,
            'calcium': 0.1
        }
    
    def _create_axon(self) -> Dict[str, Any]:
        """创建轴突室"""
        return {
            'length': self.params.axonal_length,
            'diameter': 1.0,
            'voltage': self.params.resting_potential,
            'conduction_velocity': 1.0  # m/s
        }
    
    def _initialize_ion_channels(self) -> Dict[str, Any]:
        """初始化离子通道"""
        return {
            'na_channels': {
                'density': self.params.na_channel_density,
                'activation': 0.0,
                'inactivation': 1.0,
                'current': 0.0
            },
            'k_channels': {
                'density': self.params.k_channel_density,
                'activation': 0.0,
                'current': 0.0
            },
            'ca_channels': {
                'density': self.params.ca_channel_density,
                'activation': 0.0,
                'current': 0.0
            }
        }
    
    @jit(nopython=True)
    def _hodgkin_huxley_dynamics(self, voltage: float, dt: float) -> Tuple[float, float, float, float, float, float]:
        """Hodgkin-Huxley离子通道动力学（JIT编译加速）"""
        
        # 钠通道
        alpha_m = 0.1 * (voltage + 40) / (1 - np.exp(-(voltage + 40) / 10))
        beta_m = 4 * np.exp(-(voltage + 65) / 18)
        alpha_h = 0.07 * np.exp(-(voltage + 65) / 20)
        beta_h = 1 / (1 + np.exp(-(voltage + 35) / 10))
        
        # 钾通道
        alpha_n = 0.01 * (voltage + 55) / (1 - np.exp(-(voltage + 55) / 10))
        beta_n = 0.125 * np.exp(-(voltage + 65) / 80)
        
        return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n
    
    def update(self, dt: float, synaptic_inputs: List[float]) -> Dict[str, Any]:
        """更新神经元状态"""
        
        current_time = len(self.spike_times) * dt if self.spike_times else 0.0
        
        # 检查不应期
        if current_time - self.last_spike_time < self.params.refractory_period:
            return {
                'spike': False,
                'voltage': self.params.reset_potential,
                'calcium': self.calcium_concentration,
                'atp': self.atp_level
            }
        
        # 计算突触电流
        total_synaptic_current = sum(synaptic_inputs)
        
        # 计算离子通道电流
        na_current = self._calculate_na_current()
        k_current = self._calculate_k_current()
        ca_current = self._calculate_ca_current()
        
        # 总电流
        total_current = total_synaptic_current + na_current + k_current + ca_current
        
        # 更新膜电位
        dv_dt = total_current / self.params.membrane_capacitance
        self.membrane_potential += dv_dt * dt
        
        # 检查发放
        spike_occurred = False
        if self.membrane_potential >= self.params.threshold:
            spike_occurred = True
            self.spike_times.append(current_time)
            self.last_spike_time = current_time
            self.membrane_potential = self.params.reset_potential
            
            # 钙内流
            self.calcium_concentration += 0.5
        
        # 钙浓度衰减
        self.calcium_concentration *= np.exp(-dt / 50.0)
        
        # 代谢更新
        self._update_metabolism(dt, spike_occurred)
        
        return {
            'spike': spike_occurred,
            'voltage': self.membrane_potential,
            'calcium': self.calcium_concentration,
            'atp': self.atp_level,
            'glucose': self.glucose_level,
            'oxygen': self.oxygen_level
        }
    
    def _calculate_na_current(self) -> float:
        """计算钠电流"""
        channels = self.ion_channels['na_channels']
        e_na = 50.0  # mV
        g_na = channels['density'] * channels['activation']**3 * channels['inactivation']
        return -g_na * (self.membrane_potential - e_na)
    
    def _calculate_k_current(self) -> float:
        """计算钾电流"""
        channels = self.ion_channels['k_channels']
        e_k = -77.0  # mV
        g_k = channels['density'] * channels['activation']**4
        return -g_k * (self.membrane_potential - e_k)
    
    def _calculate_ca_current(self) -> float:
        """计算钙电流"""
        channels = self.ion_channels['ca_channels']
        e_ca = 132.0  # mV
        g_ca = channels['density'] * channels['activation']**2
        return -g_ca * (self.membrane_potential - e_ca)
    
    def _update_metabolism(self, dt: float, spike_occurred: bool):
        """更新代谢状态"""
        
        # ATP消耗
        base_consumption = 0.001 * dt  # 基础代谢
        spike_consumption = 0.01 if spike_occurred else 0.0
        
        self.atp_level -= (base_consumption + spike_consumption)
        
        # ATP生产（需要葡萄糖和氧气）
        if self.glucose_level > 0.1 and self.oxygen_level > 0.1:
            atp_production = 0.005 * dt * min(self.glucose_level, self.oxygen_level)
            self.atp_level += atp_production
            
            # 消耗葡萄糖和氧气
            self.glucose_level -= atp_production * 0.1
            self.oxygen_level -= atp_production * 0.2
        
        # 限制在合理范围内
        self.atp_level = np.clip(self.atp_level, 0.0, 2.0)
        self.glucose_level = np.clip(self.glucose_level, 0.0, 2.0)
        self.oxygen_level = np.clip(self.oxygen_level, 0.0, 2.0)