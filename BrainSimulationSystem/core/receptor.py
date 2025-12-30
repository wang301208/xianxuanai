"""
详细的突触后受体模型
Detailed Postsynaptic Receptor Model
"""
import logging
import numpy as np

# 兼容无 scipy 环境
try:
    from scipy.special import expit
except ImportError:
    def expit(x):
        return 1.0 / (1.0 + np.exp(-x))

from .synapse_types import ReceptorType, ReceptorKinetics

class DetailedReceptor:
    """
    详细受体模型，模拟单个受体的动力学过程，包括结合、门控和失敏。
    """
    
    def __init__(self, receptor_type: ReceptorType, density: float, kinetics: ReceptorKinetics):
        """
        初始化一个详细的受体模型。

        Args:
            receptor_type (ReceptorType): 受体的类型。
            density (float): 受体在突触后膜上的密度 (receptors/μm²)。
            kinetics (ReceptorKinetics): 描述受体动力学特性的参数。
        """
        self.receptor_type = receptor_type
        self.density = density
        self.kinetics = kinetics
        
        # 初始化受体状态 (所有受体最初都处于关闭状态)
        self.closed_state = density
        self.open_state = 0.0
        self.bound_state = 0.0
        self.desensitized_state = 0.0
        
        # 初始化电流和电导
        self.current = 0.0
        self.conductance = 0.0
        
        self.logger = logging.getLogger(f"Receptor_{receptor_type.value}")
    
    def update(self, dt: float, neurotransmitter_conc: float, membrane_voltage: float) -> float:
        """
        在每个时间步更新受体状态和计算产生的电流。

        Args:
            dt (float): 时间步长 (ms)。
            neurotransmitter_conc (float): 突触间隙中的神经递质浓度 (mM)。
            membrane_voltage (float): 突触后膜电位 (mV)。

        Returns:
            float: 通过该受体产生的突触后电流 (pA)。
        """
        # 1. 状态转换速率计算
        binding_rate = self.kinetics.kon * neurotransmitter_conc * self.closed_state
        unbinding_rate = self.kinetics.koff * self.bound_state
        opening_rate = self.kinetics.alpha * self.bound_state
        closing_rate = self.kinetics.beta * self.open_state
        desensitization_rate = self.kinetics.desensitization_rate * self.bound_state
        recovery_rate = self.kinetics.recovery_rate * self.desensitized_state
        
        # 2. 状态变量更新 (使用欧拉法)
        d_closed = -binding_rate + unbinding_rate + recovery_rate
        d_bound = binding_rate - unbinding_rate - opening_rate - desensitization_rate
        d_open = opening_rate - closing_rate
        d_desensitized = desensitization_rate - recovery_rate
        
        self.closed_state += d_closed * dt
        self.bound_state += d_bound * dt
        self.open_state += d_open * dt
        self.desensitized_state += d_desensitized * dt
        
        # 3. 确保状态变量非负且总和守恒
        total = self.closed_state + self.bound_state + self.open_state + self.desensitized_state
        if total > 0:
            factor = self.density / total
            self.closed_state = max(0, self.closed_state * factor)
            self.bound_state = max(0, self.bound_state * factor)
            self.open_state = max(0, self.open_state * factor)
            self.desensitized_state = max(0, self.desensitized_state * factor)
        
        # 4. 计算总电导
        self.conductance = self.open_state * self.kinetics.single_channel_conductance
        
        # 5. 应用电压依赖性调节
        if self.kinetics.voltage_dependence:
            voltage_factor = self._calculate_voltage_dependence(membrane_voltage)
            self.conductance *= voltage_factor
        
        if self.kinetics.mg_block:
            mg_factor = self._calculate_mg_block(membrane_voltage)
            self.conductance *= mg_factor
        
        # 6. 计算最终电流 (I = g * (V_m - E_rev))
        self.current = self.conductance * (membrane_voltage - self.kinetics.reversal_potential)
        
        return self.current
    
    def _calculate_voltage_dependence(self, voltage: float) -> float:
        """
        计算电压依赖性因子，通常用于NMDA受体。
        使用Sigmoid函数模拟电压依赖性。
        """
        return expit((voltage + 60) / 10)
    
    def _calculate_mg_block(self, voltage: float) -> float:
        """
        计算镁离子对NMDA受体的阻断因子。
        这是一个经验公式，描述了镁离子在不同膜电位下对通道的阻断程度。
        """
        mg_conc = 1.0  # 假设细胞外镁离子浓度为 1.0 mM
        return 1.0 / (1.0 + (mg_conc / 3.57) * np.exp(-0.062 * voltage))