"""
树突计算高级实现
支持非线性空间整合与时间累积
"""

import numpy as np
from typing import List

class DendriticBranch:
    """生物可信的树突分支模型"""
    def __init__(self):
        self.synaptic_weights = []  # 突触权重列表
        self.vm = -70.0             # 局部膜电位(mV)
        self.nmda_ratio = 0.3       # NMDA受体占比
        self.spatial_const = 0.5    # 空间衰减常数
        self.temporal_window = 5    # 时间积分窗口(ms)
        
    def integrate_spatial(self, inputs: List[float]) -> float:
        """非线性空间整合"""
        # AMPA快速成分 (线性部分)
        ampa = sum(w * x for w, x in zip(self.synaptic_weights, inputs))
        
        # NMDA慢速成分 (电压依赖性非线性)
        nmda_gate = 1 / (1 + np.exp(-(self.vm + 30)/10))  # 镁离子阻断
        nmda = sum(w * x * nmda_gate 
                  for w, x in zip(self.synaptic_weights, inputs))
        
        # 整合并激活 (sigmoid非线性)
        total = ampa + self.nmda_ratio * nmda
        return 2 / (1 + np.exp(-total)) - 1  # 双曲正切变体
    
    def integrate_temporal(self, spike_train: List[float]) -> float:
        """时间积分 (带衰减)"""
        if len(spike_train) > self.temporal_window:
            window = spike_train[-self.temporal_window:]
        else:
            window = spike_train
            
        # 双指数核积分
        time_points = np.arange(len(window))
        kernel = np.exp(-time_points/2) - np.exp(-time_points/5)
        return np.dot(window, kernel)
    
    def update_potential(self, current: float, dt: float):
        """更新局部膜电位"""
        self.vm += current * dt / 10  # 电容取10μF/cm²
        self.vm = max(-80, min(50, self.vm))  # 电压钳制