"""
胶质细胞模拟模块

实现星形胶质细胞钙波模型和代谢耦合功能
"""

import numpy as np
from typing import Dict, List

class Astrocyte:
    """星形胶质细胞模型 (基于De Pittà 2009双池模型)"""
    
    def __init__(self, params: Dict = None):
        self.params = params or {
            'IP3_production': 0.16,  # μM/s
            'IP3_degradation': 0.08,  # /s
            'Ca_ER_leak': 0.02,      # /s
            'SERCA_pump': 0.9,       # μM/s
            'delta_Ca': 0.2,         # 耦合强度
            'tau_Ca': 2.0,           # s
            'tau_IP3': 10.0          # s
        }
        
        # 状态变量
        self.Ca_cyt = 0.1    # 胞质钙浓度 (μM)
        self.Ca_ER = 2.0      # 内质网钙浓度 (μM)
        self.IP3 = 0.1        # IP3浓度 (μM)
        
        # 连接参数
        self.connected_neurons = []  # 关联的神经元
        
    def update(self, dt: float):
        """更新胶质细胞状态"""
        # 计算钙流 (简化版Li-Rinzel模型)
        J_leak = self.params['Ca_ER_leak'] * (self.Ca_ER - self.Ca_cyt)
        J_pump = self.params['SERCA_pump'] * self.Ca_cyt**2 / (self.Ca_cyt**2 + 0.4**2)
        J_channel = 3.0 * self.IP3**3 * self.Ca_cyt**2 / ((self.IP3**3 + 0.3**3) * (self.Ca_cyt**2 + 0.2**2))
        
        # 更新浓度
        dCa_cyt = (J_leak + J_channel - J_pump) * dt
        dCa_ER = -(J_leak + J_channel - J_pump) * dt
        dIP3 = (self.params['IP3_production'] - self.params['IP3_degradation'] * self.IP3) * dt
        
        self.Ca_cyt += dCa_cyt
        self.Ca_ER += dCa_ER
        self.IP3 += dIP3
        
        # 神经调质效应
        for neuron in self.connected_neurons:
            neuron.glia_modulation = self.Ca_cyt * self.params['delta_Ca']

class MetabolicCoupling:
    """代谢耦合系统 (ATP/葡萄糖动态)"""
    
    def __init__(self):
        self.ATP = 2.0       # mM
        self.glucose = 5.0   # mM
        self.lactate = 1.0    # mM
        
    def update(self, neuron_activity: float, dt: float):
        """更新代谢状态"""
        # 简化的神经元-胶质细胞乳酸穿梭模型
        glucose_consumption = 0.1 * neuron_activity
        lactate_production = 0.05 * neuron_activity
        ATP_production = 2.0 * glucose_consumption
        
        self.glucose -= glucose_consumption * dt
        self.lactate += lactate_production * dt
        self.ATP = self.ATP * 0.99 + ATP_production * dt