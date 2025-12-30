"""
多类型神经元模型实现
Multi-Type Neuron Models Implementation

实现多类型神经元模型：HH、AdEx、Izhikevich、multi-compartment
包含胶质细胞和神经调质动力学
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from .neuron_base import NeuronBase, NeuronType

# Compatibility alias used by unit tests/integrations.
BaseNeuronModel = NeuronBase

# 兼容无 numba 环境
try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# 使用统一的NeuronType枚举，删除重复定义

@dataclass
class CompartmentParameters:
    """室参数"""
    length: float = 100.0        # μm
    diameter: float = 2.0        # μm
    Ra: float = 150.0           # Ω·cm
    Cm: float = 1.0             # μF/cm²
    Rm: float = 30000.0         # Ω·cm²
    
    # 离子通道密度
    Na_density: float = 120.0    # mS/cm²
    K_density: float = 36.0      # mS/cm²
    Ca_density: float = 0.0      # mS/cm²
    h_density: float = 0.0       # mS/cm²
    
    # 突触参数
    synapse_density: float = 1.0  # synapses/μm²

@dataclass
class IonChannelState:
    """离子通道状态"""
    # HH钠通道
    m: float = 0.0  # 激活
    h: float = 1.0  # 失活
    
    # HH钾通道
    n: float = 0.0  # 激活
    
    # 钙通道
    ca_m: float = 0.0
    ca_h: float = 1.0
    
    # h电流
    h_m: float = 0.0
    
    # 钙浓度
    ca_i: float = 50e-6  # mM



class LIFNeuron(NeuronBase):
    """Leaky Integrate-and-Fire神经元（统一接口实现）"""
    
    def __init__(self, neuron_id: int, params: Dict[str, Any]):
        # 设置LIF特定参数
        self.tau_m = params.get('tau_m', 20.0)
        self.V_rest = params.get('V_rest', -70.0)
        self.V_thresh = params.get('V_thresh', -50.0)
        self.V_reset = params.get('V_reset', -70.0)
        self.t_ref = params.get('t_ref', 2.0)
        self.R_m = params.get('R_m', 10.0)
        
        # 调用基类初始化
        super().__init__(neuron_id, NeuronType.LIF, params)
        
        # LIF神经元特定状态
        self.refractory_counter = 0  # 不应期计数器
    
    def update(self, dt: float, current_time: float = 0.0) -> bool:
        """更新LIF神经元（返回是否发放）。"""
        self._current_time = current_time
        input_current = self.I_ext + self.I_syn
        if self.neuromodulation:
            input_current += sum(self.neuromodulation.values()) * 10.0
        # 不应期处理
        if self.refractory_counter > 0:
            self.refractory_counter -= dt
            if self.refractory_counter <= 0:
                self.voltage = self.V_reset
                self.refractory_counter = 0
            self.is_spiking = False
            return False
        
        # LIF动力学方程
        dV_dt = (self.V_rest - self.voltage + self.R_m * input_current) / self.tau_m
        self.voltage += dV_dt * dt
        
        # 检查是否发放脉冲
        if self.voltage >= self.V_thresh:
            self.is_spiking = True
            self.last_spike_time = current_time
            self.spike_times.append(current_time)
            self.refractory_counter = self.t_ref
            self.voltage = self.V_reset
            return True
        else:
            self.is_spiking = False

        return False
    
    def reset(self) -> None:
        """重置神经元状态"""
        self.voltage = self.V_rest
        self.refractory_counter = 0
        self.is_spiking = False
        self.spike_times.clear()
        self.last_spike_time = -float('inf')
        self.I_ext = 0.0
        self.I_syn = 0.0

class HodgkinHuxleyNeuron(NeuronBase):
    """Hodgkin-Huxley神经元（统一接口实现）"""
    
    def __init__(self, neuron_id: int, params: Dict[str, Any]):
        default_params = {
            'C_m': 1.0,
            'g_Na': 120.0,
            'g_K': 36.0,
            'g_L': 0.3,
            'E_Na': 50.0,
            'E_K': -77.0,
            'E_L': -54.4,
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(neuron_id, NeuronType.HODGKIN_HUXLEY, merged_params)
        self._prev_V = self.V
    
    def update(self, dt: float, current_time: float = 0.0) -> bool:
        """更新HH神经元（返回是否发放）。"""
        self._current_time = current_time
        input_current = self.I_ext + self.I_syn
        
        # 计算门控变量的alpha和beta
        V = self.V
        eps = 1e-9
        denom_m = 1.0 - np.exp(-(V + 40.0) / 10.0)
        denom_n = 1.0 - np.exp(-(V + 55.0) / 10.0)
        alpha_m = 0.1 * (V + 40.0) / (denom_m if abs(denom_m) > eps else eps)
        beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
        
        alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
        
        alpha_n = 0.01 * (V + 55.0) / (denom_n if abs(denom_n) > eps else eps)
        beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
        
        # 更新门控变量
        dm = (alpha_m * (1.0 - self.m) - beta_m * self.m) * dt
        dh = (alpha_h * (1.0 - self.h) - beta_h * self.h) * dt
        dn = (alpha_n * (1.0 - self.n) - beta_n * self.n) * dt
        
        self.m += dm
        self.h += dh
        self.n += dn
        
        # 计算电流
        I_Na = self.g_Na * (self.m ** 3) * self.h * (V - self.E_Na)
        I_K = self.g_K * (self.n ** 4) * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)
        
        # 更新膜电位
        dV = (input_current - I_Na - I_K - I_L) / self.C_m
        self.V = V + dV * dt
        
        # 检查发放
        spiked = bool(self._prev_V < 0.0 and self.V >= 0.0)
        self.is_spiking = spiked
        if spiked:
            self.last_spike_time = current_time
            self.spike_times.append(current_time)
        self._prev_V = self.V
        return spiked
    
    def reset(self):
        """重置状态"""
        self.V = -65.0
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        self.I_ext = 0.0
        self.I_syn = 0.0
        self.is_spiking = False
        self.spike_times.clear()
        self.last_spike_time = -float('inf')
        self._prev_V = self.V

class AdExNeuron(NeuronBase):
    """Adaptive Exponential Integrate-and-Fire神经元（统一接口实现）"""
    
    def __init__(self, neuron_id: int, params: Dict[str, Any]):
        default_params = {
            # Scaled AdEx defaults: tuned so that ~200 units of input current
            # elicit spiking in typical unit tests.
            'C': 100.0,
            'g_L': 5.0,
            'E_L': -70.0,
            'V_T': -50.0,
            'Delta_T': 2.0,
            'a': 4.0,
            'tau_w': 144.0,
            'b': 80.5,
            'V_thresh': -40.0,
            'V_reset': -70.0,
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__(neuron_id, NeuronType.ADAPTIVE_EXPONENTIAL, merged_params)
        self.w = 0.0
    
    def update(self, dt: float, current_time: float) -> bool:
        """更新AdEx神经元"""
        
        # 总输入电流
        I_total = self.I_ext + self.I_syn
        
        # 神经调质调节适应强度
        adaptation_modulation = 1.0 + 0.3 * self.neuromodulation.get('serotonin', 0.0)
        
        # 指数项
        if self.V < self.V_thresh:
            exp_term = self.Delta_T * np.exp((self.V - self.V_T) / self.Delta_T)
        else:
            exp_term = 0.0
        
        # 更新膜电位
        dV = (-self.g_L * (self.V - self.E_L) + self.g_L * exp_term - self.w + I_total) / self.C
        self.V += dV * dt
        
        # 更新适应变量
        dw = (self.a * (self.V - self.E_L) - self.w) / self.tau_w * adaptation_modulation
        self.w += dw * dt
        
        # 检查发放
        if self.V >= self.V_thresh:
            self.V = self.V_reset
            self.w += self.b
            self.last_spike_time = current_time
            self.spike_times.append(current_time)
            return True
        
        return False
    
    def reset(self):
        """重置状态"""
        self.V = self.E_L
        self.w = 0.0
        self.I_ext = 0.0
        self.I_syn = 0.0

class IzhikevichNeuron(NeuronBase):
    """Izhikevich神经元（统一接口实现）"""
    
    def __init__(self, neuron_id: int, params: Dict[str, Any]):
        # 确保包含所有必要参数
        default_params = {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0, 'V_thresh': 30.0}
        merged_params = {**default_params, **params}
        super().__init__(neuron_id, NeuronType.IZHIKEVICH, merged_params)
        
        # 状态变量
        self.u = self.b * self.V  # 恢复变量
    
    def update(self, dt: float, current_time: float) -> bool:
        """更新Izhikevich神经元"""
        
        # 总输入电流
        I_total = self.I_ext + self.I_syn
        
        # 神经调质调节
        modulation = sum(self.neuromodulation.values())
        I_total += modulation * 10.0  # 调节强度
        
        # Izhikevich方程
        dv = (0.04 * self.V**2 + 5 * self.V + 140 - self.u + I_total) * dt
        du = self.a * (self.b * self.V - self.u) * dt
        
        self.V += dv
        self.u += du
        
        # 检查发放
        if self.V >= self.V_thresh:
            self.V = self.c
            self.u += self.d
            self.last_spike_time = current_time
            self.spike_times.append(current_time)
            return True
        
        return False
    
    def reset(self):
        """重置状态"""
        self.V = self.c
        self.u = self.b * self.V
        self.I_ext = 0.0
        self.I_syn = 0.0
        self.spike_times = []

class MultiCompartmentNeuron(NeuronBase):
    """多室神经元模型（统一接口实现）"""
    
    def __init__(self, neuron_id: int, params: Dict[str, Any]):
        # 提前初始化容器，避免基类初始化期间 reset 访问未定义属性
        self.compartments = {}
        self.connections = []  # (source_comp, target_comp, conductance)
        self.ion_channels = {}
        super().__init__(neuron_id, NeuronType.MULTI_COMPARTMENT, params)
        
        # 创建室
        self._create_compartments(params.get('morphology', {}))
        
        # 离子通道状态
        self.ion_channels = {comp_name: IonChannelState() 
                           for comp_name in self.compartments.keys()}
        self.reset()
    
    def _create_compartments(self, morphology: Dict[str, Any]):
        """创建室结构"""
        
        # 默认形态学
        default_morphology = {
            'soma': CompartmentParameters(length=20.0, diameter=20.0),
            'basal_dendrite': CompartmentParameters(length=200.0, diameter=2.0),
            'apical_dendrite': CompartmentParameters(length=400.0, diameter=3.0),
            'axon': CompartmentParameters(length=1000.0, diameter=1.0, Na_density=500.0)
        }
        
        # 合并用户定义的形态学
        for comp_name, default_params in default_morphology.items():
            user_params = morphology.get(comp_name, {})
            # Back-compat: allow a single "dendrite" definition to parameterize the basal dendrite.
            if not user_params and comp_name == "basal_dendrite" and isinstance(morphology, dict):
                user_params = morphology.get("dendrite", {})
            
            # 创建室参数
            comp_params = CompartmentParameters(**{
                **default_params.__dict__,
                **user_params
            })
            
            self.compartments[comp_name] = {
                'params': comp_params,
                'V': -70.0,  # 膜电位
                'area': np.pi * comp_params.diameter * comp_params.length,  # μm²
                'synapses': []  # 突触列表
            }
        
        # Provide a "dendrite" alias expected by lightweight tests/configs.
        if isinstance(morphology, dict) and "dendrite" in morphology and "dendrite" not in self.compartments:
            if "basal_dendrite" in self.compartments:
                self.compartments["dendrite"] = self.compartments["basal_dendrite"]

        # 创建室间连接
        self.connections = [
            ('soma', 'basal_dendrite', 0.1),
            ('soma', 'apical_dendrite', 0.1),
            ('soma', 'axon', 0.2)
        ]
    
    def update(self, dt: float, current_time: float) -> bool:
        """更新多室神经元"""
        
        spike_occurred = False
        
        # 更新每个室
        for comp_name, comp_data in self.compartments.items():
            params = comp_data['params']
            
            # 计算离子电流
            I_ion = self._calculate_ion_currents(comp_name, comp_data['V'])
            
            # 计算室间电流
            I_axial = self._calculate_axial_currents(comp_name)
            
            # 计算突触电流
            I_syn = sum(synapse.get_current() for synapse in comp_data['synapses'])
            
            # 外部电流（仅胞体）
            I_ext = self.I_ext if comp_name == 'soma' else 0.0
            
            # 总电流
            I_total = I_ext + I_syn + I_axial - I_ion
            
            # 更新膜电位
            C_m = params.Cm * comp_data['area'] * 1e-8  # 转换为pF
            dV = I_total / C_m
            comp_data['V'] += dV * dt
            
            # 检查胞体发放
            if comp_name == 'soma' and comp_data['V'] > 0:
                spike_occurred = True
                comp_data['V'] = -70.0  # 重置
                self.last_spike_time = current_time
                self.spike_times.append(current_time)
        
        # 更新离子通道状态
        self._update_ion_channels(dt)
        
        return spike_occurred
    
    def _calculate_ion_currents(self, comp_name: str, V: float) -> float:
        """计算离子电流"""
        
        comp_params = self.compartments[comp_name]['params']
        ion_state = self.ion_channels[comp_name]
        area = self.compartments[comp_name]['area']
        
        # 钠电流
        g_Na = comp_params.Na_density * area * 1e-8  # nS
        I_Na = g_Na * ion_state.m**3 * ion_state.h * (V - 50.0)
        
        # 钾电流
        g_K = comp_params.K_density * area * 1e-8  # nS
        I_K = g_K * ion_state.n**4 * (V + 77.0)
        
        # 漏电流
        g_L = area * 1e-8 / comp_params.Rm  # nS
        I_L = g_L * (V + 70.0)
        
        return I_Na + I_K + I_L
    
    def _calculate_axial_currents(self, comp_name: str) -> float:
        """计算室间轴向电流"""
        
        I_axial = 0.0
        V_comp = self.compartments[comp_name]['V']
        
        for source, target, g_axial in self.connections:
            if source == comp_name:
                V_target = self.compartments[target]['V']
                I_axial += g_axial * (V_target - V_comp)
            elif target == comp_name:
                V_source = self.compartments[source]['V']
                I_axial += g_axial * (V_source - V_comp)
        
        return I_axial
    
    def _update_ion_channels(self, dt: float):
        """更新离子通道状态"""
        
        for comp_name, ion_state in self.ion_channels.items():
            V = self.compartments[comp_name]['V']
            
            # HH门控变量
            alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
            beta_m = 4.0 * np.exp(-(V + 65) / 18)
            
            alpha_h = 0.07 * np.exp(-(V + 65) / 20)
            beta_h = 1 / (1 + np.exp(-(V + 35) / 10))
            
            alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
            beta_n = 0.125 * np.exp(-(V + 65) / 80)
            
            # 更新门控变量
            ion_state.m += (alpha_m * (1 - ion_state.m) - beta_m * ion_state.m) * dt
            ion_state.h += (alpha_h * (1 - ion_state.h) - beta_h * ion_state.h) * dt
            ion_state.n += (alpha_n * (1 - ion_state.n) - beta_n * ion_state.n) * dt
    
    def add_synapse_to_compartment(self, compartment_name: str, synapse):
        """向指定室添加突触"""
        if compartment_name in self.compartments:
            self.compartments[compartment_name]['synapses'].append(synapse)
    
    def get_compartment_voltage(self, compartment_name: str) -> float:
        """获取指定室的膜电位"""
        return self.compartments.get(compartment_name, {}).get('V', -70.0)
    
    def reset(self):
        """重置状态"""
        for comp_data in self.compartments.values():
            comp_data['V'] = -70.0
        
        for ion_state in self.ion_channels.values():
            ion_state.m = 0.05
            ion_state.h = 0.6
            ion_state.n = 0.32
        
        self.I_ext = 0.0
        self.I_syn = 0.0

class GlialCell(NeuronBase):
    """胶质细胞基类（统一接口实现）"""
    
    def __init__(self, neuron_id: int, cell_type: NeuronType, params: Dict[str, Any]):
        super().__init__(neuron_id, cell_type, params)
        
        # 胶质细胞特有属性
        self.territory_radius = params.get('territory_radius', 50.0)  # μm
        self.connected_neurons = []
        self.metabolic_state = 1.0
        self.last_spike_time = 0.0
        self.spike_times = []
    
    def update(self, dt: float, current_time: float) -> bool:
        """更新胶质细胞基类（默认实现）"""
        # 胶质细胞不发放动作电位
        return False
    
    def reset(self):
        """重置胶质细胞状态"""
        # 重置基本状态
        self.I_ext = 0.0
        self.I_syn = 0.0
        self.spike_times = []

class Astrocyte(GlialCell):
    """星形胶质细胞"""
    
    def __init__(self, neuron_id: int, params: Dict[str, Any]):
        super().__init__(neuron_id, NeuronType.ASTROCYTE, params)
        
        # 钙动力学参数
        self.Ca_cyt = 0.1      # μM
        self.Ca_ER = 2.0       # μM
        self.IP3 = 0.1         # μM
        
        # 代谢参数
        self.glucose = 5.0     # mM
        self.lactate = 1.0     # mM
        self.glutamate_uptake = 0.0
        
        # 钙波传播
        self.calcium_wave_speed = 20.0  # μm/s
        self.wave_threshold = 0.5       # μM
    
    def update(self, dt: float, current_time: float) -> bool:
        """更新星形胶质细胞"""
        
        # 钙动力学（简化Li-Rinzel模型）
        J_leak = 0.02 * (self.Ca_ER - self.Ca_cyt)
        J_pump = 0.9 * self.Ca_cyt**2 / (self.Ca_cyt**2 + 0.4**2)
        J_channel = 3.0 * self.IP3**3 * self.Ca_cyt**2 / ((self.IP3**3 + 0.3**3) * (self.Ca_cyt**2 + 0.2**2))
        
        # 更新钙浓度
        dCa_cyt = (J_leak + J_channel - J_pump) * dt
        dCa_ER = -(J_leak + J_channel - J_pump) * dt
        dIP3 = (0.16 - 0.08 * self.IP3) * dt
        
        self.Ca_cyt += dCa_cyt
        self.Ca_ER += dCa_ER
        self.IP3 += dIP3
        
        # 限制在生理范围
        self.Ca_cyt = np.clip(self.Ca_cyt, 0.05, 5.0)
        self.Ca_ER = np.clip(self.Ca_ER, 0.5, 10.0)
        self.IP3 = np.clip(self.IP3, 0.01, 2.0)
        
        # 谷氨酸摄取
        neuron_activity = sum(1 for neuron in self.connected_neurons 
                            if hasattr(neuron, 'last_spike_time') and 
                            current_time - neuron.last_spike_time < 10.0)
        
        self.glutamate_uptake = 0.1 * neuron_activity
        
        # 代谢支持
        self._update_metabolism(dt, neuron_activity)
        
        # 调节连接的神经元
        for neuron in self.connected_neurons:
            if hasattr(neuron, 'glia_modulation'):
                neuron.glia_modulation = self.Ca_cyt * 0.1
        
        return False  # 胶质细胞不发放动作电位
    
    def _update_metabolism(self, dt: float, neuron_activity: float):
        """更新代谢状态"""
        
        # 葡萄糖消耗
        glucose_consumption = 0.1 * neuron_activity * dt
        self.glucose -= glucose_consumption
        
        # 乳酸生产
        lactate_production = 0.05 * neuron_activity * dt
        self.lactate += lactate_production
        
        # 限制在生理范围
        self.glucose = np.clip(self.glucose, 1.0, 10.0)
        self.lactate = np.clip(self.lactate, 0.5, 5.0)
    
    def connect_neuron(self, neuron):
        """连接神经元"""
        if neuron not in self.connected_neurons:
            self.connected_neurons.append(neuron)
    
    def reset(self):
        """重置状态"""
        self.Ca_cyt = 0.1
        self.Ca_ER = 2.0
        self.IP3 = 0.1
        self.glucose = 5.0
        self.lactate = 1.0
        self.glutamate_uptake = 0.0

class Microglia(GlialCell):
    """小胶质细胞"""
    
    def __init__(self, neuron_id: int, params: Dict[str, Any]):
        # 激活状态
        self.activation_level = 0.0  # 0-1
        self.surveillance_mode = True
        
        # 炎症因子
        self.cytokines = {
            'TNF_alpha': 0.0,
            'IL_1beta': 0.0,
            'IL_6': 0.0
        }
        
        # 吞噬活动
        self.phagocytic_activity = 0.0

        # Call base initialisation after attributes exist (NeuronBase calls reset()).
        super().__init__(neuron_id, NeuronType.MICROGLIA, params)
    
    def update(self, dt: float, current_time: float) -> bool:
        """更新小胶质细胞"""
        
        # 检测神经元损伤信号
        damage_signals = 0.0
        for neuron in self.connected_neurons:
            if hasattr(neuron, 'V') and neuron.V < -80.0:  # 异常低电位
                damage_signals += 1.0
        
        # 更新激活水平
        if damage_signals > 0:
            self.activation_level = min(1.0, self.activation_level + 0.1 * dt)
            self.surveillance_mode = False
        else:
            self.activation_level = max(0.0, self.activation_level - 0.05 * dt)
            if self.activation_level < 0.1:
                self.surveillance_mode = True
        
        # 释放炎症因子
        if self.activation_level > 0.5:
            for cytokine in self.cytokines:
                self.cytokines[cytokine] += self.activation_level * 0.1 * dt
        else:
            for cytokine in self.cytokines:
                self.cytokines[cytokine] *= 0.95  # 衰减
        
        # 吞噬活动
        self.phagocytic_activity = self.activation_level * 0.5
        
        return False
    
    def reset(self):
        """重置状态"""
        self.activation_level = 0.0
        self.surveillance_mode = True
        for cytokine in self.cytokines:
            self.cytokines[cytokine] = 0.0
        self.phagocytic_activity = 0.0

class NeuromodulatorSystem:
    """神经调质系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 神经调质浓度
        self.concentrations = {
            'dopamine': 0.0,
            'serotonin': 0.0,
            'acetylcholine': 0.0,
            'norepinephrine': 0.0,
            'histamine': 0.0
        }
        
        # 释放参数
        self.release_rates = {
            'dopamine': 0.01,
            'serotonin': 0.005,
            'acetylcholine': 0.02,
            'norepinephrine': 0.01,
            'histamine': 0.001
        }
        
        # 清除参数
        self.clearance_rates = {
            'dopamine': 0.1,
            'serotonin': 0.05,
            'acetylcholine': 0.5,  # 快速水解
            'norepinephrine': 0.08,
            'histamine': 0.02
        }
    
    def update(self, dt: float, neural_activity: Dict[str, float]):
        """更新神经调质浓度"""
        
        for modulator in self.concentrations:
            # 基于神经活动的释放
            activity = neural_activity.get(modulator, 0.0)
            release = self.release_rates[modulator] * activity * dt
            
            # 清除
            clearance = self.clearance_rates[modulator] * self.concentrations[modulator] * dt
            
            # 更新浓度
            self.concentrations[modulator] += release - clearance
            self.concentrations[modulator] = max(0.0, self.concentrations[modulator])
    
    def apply_to_neurons(self, neurons: List[NeuronBase]):
        """将神经调质应用到神经元"""
        
        for neuron in neurons:
            for modulator, concentration in self.concentrations.items():
                neuron.apply_neuromodulation(modulator, concentration)

def _normalize_cell_type_tag(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().lower()
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        try:
            return str(enum_value).strip().lower()
        except Exception:
            pass
    enum_name = getattr(value, "name", None)
    if enum_name is not None:
        try:
            return str(enum_name).strip().lower()
        except Exception:
            pass
    try:
        return str(value).strip().lower()
    except Exception:
        return ""


def _infer_neuron_type_from_params(params: Dict[str, Any]) -> NeuronType:
    """Infer a concrete neuron model from `cell_type`/population hints."""

    if not isinstance(params, dict):
        return NeuronType.LIF

    pop = params.get("population_type")
    pop_tag = _normalize_cell_type_tag(pop)
    if pop_tag in {"inhibitory", "inh", "interneuron"}:
        return NeuronType.IZHIKEVICH
    if pop_tag in {"excitatory", "exc", "pyramidal", "projection"}:
        return NeuronType.ADAPTIVE_EXPONENTIAL

    cell = params.get("cell_type")
    cell_tag = _normalize_cell_type_tag(cell)

    if any(token in cell_tag for token in ("astrocyte", "bergmann", "glia", "opg")):
        return NeuronType.ASTROCYTE
    if "microglia" in cell_tag:
        return NeuronType.MICROGLIA

    if any(
        token in cell_tag
        for token in (
            "interneuron",
            "pv",
            "parvalbumin",
            "sst",
            "somatostatin",
            "vip",
            "lamp5",
            "basket",
            "chandelier",
            "martinotti",
            "gaba",
        )
    ):
        return NeuronType.IZHIKEVICH

    if any(
        token in cell_tag
        for token in (
            "pyramidal",
            "glutamate",
            "excit",
            "cortex_rs",
            "cortex_ib",
            "hippocampus_rs",
            "hippocampus_ib",
            "thalamus_tc",
        )
    ):
        return NeuronType.ADAPTIVE_EXPONENTIAL

    if any(token in cell_tag for token in ("dopamin", "seroton", "acetylch", "norepinephrine", "histamine")):
        params.setdefault("tau_w", 400.0)
        params.setdefault("a", 2.0)
        return NeuronType.ADAPTIVE_EXPONENTIAL

    is_inhibitory = params.get("is_inhibitory")
    if isinstance(is_inhibitory, bool):
        return NeuronType.IZHIKEVICH if is_inhibitory else NeuronType.ADAPTIVE_EXPONENTIAL

    return NeuronType.LIF

def create_neuron(neuron_type: NeuronType | str, neuron_id: int, params: Dict[str, Any]) -> NeuronBase:
    """神经元工厂函数"""

    if isinstance(neuron_type, str):
        normalized = neuron_type.strip().lower()
        if normalized in {"auto", "cell_type", "celltype"}:
            normalized = _infer_neuron_type_from_params(params).value
        # 将域特异的细胞名称映射到已实现的基础模型
        aliases = {
            "thalamic_relay": "lif",
            "fast_spiking": "lif",
            "reticular": "lif",
            "matrix": "lif",
            "core": "lif",
            # CellType-like tags (enums/strings) -> abstract neuron role tags.
            "pyramidal_l2_3": "pyramidal",
            "pyramidal_l23": "pyramidal",
            "pyramidal_l2/3": "pyramidal",
            "pyramidal_l5a": "pyramidal",
            "pyramidal_l5b": "pyramidal",
            "pyramidal_l6": "pyramidal",
            "pv_interneuron": "interneuron",
            "parvalbumin_interneuron": "interneuron",
            "sst_interneuron": "interneuron",
            "somatostatin_interneuron": "interneuron",
            "vip_interneuron": "interneuron",
            "vasoactive_intestinal_peptide_interneuron": "interneuron",
            "lamp5_interneuron": "interneuron",
            "basket_cell": "interneuron",
            "chandelier_cell": "interneuron",
            "martinotti_cell": "interneuron",
            "protoplasmic_astrocyte": "astrocyte",
            "fibrous_astrocyte": "astrocyte",
            "astrocyte_protoplasmic": "astrocyte",
            "astrocyte_fibrous": "astrocyte",
            "microglia_ramified": "microglia",
            "microglia_activated": "microglia",
            "excitatory": "pyramidal",
            "inhibitory": "interneuron",
        }
        normalized = aliases.get(normalized, normalized)
        try:
            neuron_type = NeuronType(normalized)
        except ValueError:
            try:
                neuron_type = NeuronType[neuron_type.strip().upper()]
            except KeyError as exc:
                raise ValueError(f'不支持的神经元类型: {neuron_type}') from exc

    neuron_classes = {
        NeuronType.LIF: LIFNeuron,
        NeuronType.HODGKIN_HUXLEY: HodgkinHuxleyNeuron,
        NeuronType.ADAPTIVE_EXPONENTIAL: AdExNeuron,
        NeuronType.IZHIKEVICH: IzhikevichNeuron,
        NeuronType.MULTI_COMPARTMENT: MultiCompartmentNeuron,
        NeuronType.PYRAMIDAL: AdExNeuron,
        NeuronType.INTERNEURON: IzhikevichNeuron,
        NeuronType.ASTROCYTE: Astrocyte,
        NeuronType.MICROGLIA: Microglia
    }
    
    if neuron_type not in neuron_classes:
        raise ValueError(f"不支持的神经元类型: {neuron_type}")
    
    return neuron_classes[neuron_type](neuron_id, params)

def get_default_parameters(neuron_type: NeuronType) -> Dict[str, Any]:
    """获取默认参数"""
    
    defaults = {
        NeuronType.LIF: {
            'tau_m': 20.0, 'V_rest': -70.0, 'V_thresh': -50.0,
            'V_reset': -70.0, 't_ref': 2.0, 'R_m': 10.0
        },
        NeuronType.HODGKIN_HUXLEY: {
            'C_m': 1.0, 'g_Na': 120.0, 'g_K': 36.0, 'g_L': 0.3,
            'E_Na': 50.0, 'E_K': -77.0, 'E_L': -54.4
        },
        NeuronType.ADAPTIVE_EXPONENTIAL: {
            'C': 100.0, 'g_L': 5.0, 'E_L': -70.0, 'V_T': -50.0,
            'Delta_T': 2.0, 'a': 4.0, 'tau_w': 144.0, 'b': 80.5,
            'V_thresh': -40.0, 'V_reset': -70.0
        },
        NeuronType.IZHIKEVICH: {
            'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0, 'V_thresh': 30.0
        },
        NeuronType.ASTROCYTE: {
            'territory_radius': 50.0
        },
        NeuronType.MICROGLIA: {
            'territory_radius': 30.0
        }
    }
    
    return defaults.get(neuron_type, {})
