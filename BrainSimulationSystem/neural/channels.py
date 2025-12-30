"""
增强版离子通道实现
包含钙动态调节系统
"""

import numpy as np
from typing import Optional

class DynamicCalciumChannel:
    """支持细胞内钙反馈的电压门控钙通道"""
    def __init__(self):
        self.conductance = 1.2  # 最大电导(mS/cm²)
        self.activation = 0.0
        self.inactivation = 1.0
        self.ca_intracellular = 0.0  # 胞内钙浓度(μM)
        self.ca_extracellular = 2000  # 胞外钙浓度(μM)
        
    def update_gates(self, voltage: float, dt: float = 0.01):
        """更新门控状态"""
        # 钙依赖性调节因子
        ca_inhibition = 1 / (1 + (self.ca_intracellular / 0.3)**2)
        
        # 电压依赖激活
        alpha_m = 0.05 * (voltage + 20) / (1 - np.exp(-(voltage + 20)/10)) * ca_inhibition
        beta_m = 0.1 * np.exp(-(voltage + 50)/80)
        
        # 钙依赖性失活
        alpha_h = 0.002 * np.exp(-(voltage + 50)/80)
        beta_h = 0.01 / (1 + np.exp(-(voltage + 50)/10))
        
        # 更新门控变量
        self.activation += (alpha_m*(1-self.activation) - beta_m*self.activation) * dt
        self.inactivation += (alpha_h*(1-self.inactivation) - beta_h*self.inactivation) * dt
        
        # 更新钙浓度 (钙电流驱动)
        if self.activation > 0.2:
            self.ca_intracellular += 0.05 * self.activation * (self.ca_extracellular - self.ca_intracellular)
        self.ca_intracellular *= 0.98  # 钙泵清除

class CalciumPoolManager:
    """细胞内钙库系统"""
    def __init__(self):
        self.ER_store = 100.0  # 内质网钙储存(μM)
        self.cytoplasm = 0.1    # 胞浆基础钙浓度
        
    def CICR_release(self, ip3_level: float) -> float:
        """钙诱导钙释放"""
        release_amount = min(ip3_level * 5, self.ER_store * 0.3)
        self.ER_store -= release_amount
        return release_amount
    
    def reuptake(self, dt: float):
        """钙泵回收"""
        pumped = min(0.5 * dt, self.cytoplasm)
        self.cytoplasm -= pumped
        self.ER_store += pumped * 0.8  # 80%回收效率