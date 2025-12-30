"""
神经调质系统模块

实现多巴胺、5-HT等神经调质的扩散和作用机制
"""

from enum import Enum
from typing import List
import numpy as np

class ModulatorType(Enum):
    DOPAMINE = 1
    SEROTONIN = 2
    NOREPINEPHRINE = 3
    ACETYLCHOLINE = 4

class Neuromodulator:
    """神经调质扩散和作用模型"""
    
    def __init__(self, mod_type: ModulatorType):
        self.type = mod_type
        self.concentration = 0.0  # 基准浓度 (μM)
        self.receptors = {
            'D1': 0.5 if mod_type == ModulatorType.DOPAMINE else 0.0,
            '5HT2A': 0.5 if mod_type == ModulatorType.SEROTONIN else 0.0
        }
        
        # 扩散参数
        self.diffusion_coef = 0.01  # mm²/s
        self.degradation_rate = 0.1  # /s
        
    def update_diffusion(self, sources: List[float], dt: float):
        """更新调质浓度 (简化扩散模型)"""
        # 源项 + 扩散 - 降解
        new_conc = (sum(sources) + 
                   self.diffusion_coef * np.random.normal(0, 0.1) - 
                   self.degradation_rate * self.concentration)
        self.concentration = max(0, self.concentration + new_conc * dt)
        
    def modulate_synapse(self, synapse):
        """调节突触参数"""
        if self.type == ModulatorType.DOPAMINE:
            # 多巴胺能调节 (D1受体增强LTP)
            synapse.alpha_LTP *= (1 + 0.5 * self.receptors['D1'] * self.concentration)
            
        elif self.type == ModulatorType.SEROTONIN:
            # 5-HT调节 (2A受体增强LTD)
            synapse.alpha_LTD *= (1 + 0.3 * self.receptors['5HT2A'] * self.concentration)

class ModulatorNetwork:
    """全脑调质网络"""
    
    def __init__(self):
        self.dopamine = Neuromodulator(ModulatorType.DOPAMINE)
        self.serotonin = Neuromodulator(ModulatorType.SEROTONIN)
        self.norepinephrine = Neuromodulator(ModulatorType.NOREPINEPHRINE)
        self.ach = Neuromodulator(ModulatorType.ACETYLCHOLINE)
        
    def update(self, dt: float):
        """更新所有调质系统"""
        self.dopamine.update_diffusion([], dt)
        self.serotonin.update_diffusion([], dt) 
        self.norepinephrine.update_diffusion([], dt)
        self.ach.update_diffusion([], dt)
